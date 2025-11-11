/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_atomic_int64 : require

#include "shaders_scene.h"
#include "traverse_device_host.h"

layout(local_size_x = TRAVERSAL_WORKGROUP_SIZE) in;

layout(push_constant, scalar) uniform SortInstancesConstant_
{
  SortInstancesConstant pc;
};

uint div_up(uint n, uint d)
{
  return (n + d - 1) / d;
}

void main()
{
  const uint32_t instanceId = gl_GlobalInvocationID.x;
  if(instanceId >= pc.instancesSize)
    return;

  uint32_t      rootIndex   = 0;
  InstanceArray instances   = InstanceArray(pc.instances);
  Instance      instance    = instances.array[instanceId];
  MeshArray     meshes      = MeshArray(pc.meshes);
  Mesh          mesh        = meshes.array[instance.meshIndex];
  NodeArray     nodes       = NodeArray(mesh.nodesAddress);
  Node          root        = nodes.array[rootIndex];
  NodeRange     rootChilren = decodeChildren(root.childrenOrClusters);

  // DANGER: assuming the first child is LOD0 root - in practice it was when written.
  // Ideally, there would be a more reliable mechanism to get the mesh's bounding sphere.
  Node lod0Root   = nodes.array[rootChilren.childOffset + 0];
  vec4 lod0Sphere = vec4(lod0Root.bsx, lod0Root.bsy, lod0Root.bsz, lod0Root.bsw);

  vec3  cameraInObjectSpace = vec3(instance.transformInv * vec4(pc.traversalParams.viewPosition, 1.0));
  float distance            = length(cameraInObjectSpace - lod0Sphere.xyz);

  // Bubble sort into list of instances to traverse using a 64bit atomicMin
  SortingMeshInstancesArray sortingMeshInstances = SortingMeshInstancesArray(pc.sortingMeshInstances);
  uint32_t                  key                  = uint32_t(distance * pc.traversalParams.distanceToUNorm32);
  uint64_t                  keyValue             = (uint64_t(key) << 32) | instanceId;

  // All threads load their mesh's nearest array into local cache
  // No branching ensures uniform execution and avoids dependent loads in the search
  uint64_t cachedNearest[TRAVERSAL_NEAREST_INSTANCE_COUNT];
  for(uint32_t i = 0; i < TRAVERSAL_NEAREST_INSTANCE_COUNT; ++i)
  {
    cachedNearest[i] = sortingMeshInstances.array[instance.meshIndex].nearest[i];
  }

  // Search for insertion position (implicitly checks if keyValue < worstValue)
  // If no valid position found, startIdx will be TRAVERSAL_NEAREST_INSTANCE_COUNT
  uint32_t startIdx = TRAVERSAL_NEAREST_INSTANCE_COUNT;
  for(uint32_t i = 0; i < TRAVERSAL_NEAREST_INSTANCE_COUNT; ++i)
  {
    // Would be nice to filter out instances based on pointInLimacon(), but this
    // would be a non-commutative operation that would break sorting, introduce
    // temporal instability and wrong results. Consider an instance that is
    // filtered out because it's outside the limacon of an existing instance.
    // Then the existing instance is replaced with a closer one, but the closer
    // one's limacon may not exclude those already filtered out.
    // TODO: this would be possible with greedy selection, but that would
    // require multiple passes of all instances. Might be worth trying.
    if(keyValue < cachedNearest[i])
    {
      startIdx = i;
      break;
    }
  }

  // Atomic bubble sort only if a valid insertion position was found
  // Starting from the found position reduces average atomic operations
  if(startIdx < TRAVERSAL_NEAREST_INSTANCE_COUNT)
  {
    for(uint32_t i = startIdx; i < TRAVERSAL_NEAREST_INSTANCE_COUNT; ++i)
    {
      uint64_t old = atomicMin(sortingMeshInstances.array[instance.meshIndex].nearest[i], keyValue);
      if(old >= keyValue)
        keyValue = old;
    }
  }
}
