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

#ifndef TRAVERSE_PER_INSTANCE
#error "Must define TRAVERSE_PER_INSTANCE"
#endif

#include "shaders_scene.h"
#include "traverse_device_host.h"

layout(local_size_x = TRAVERSAL_WORKGROUP_SIZE) in;

layout(set = 0, binding = BTraversalConstants, scalar) uniform TraverseConstants_
{
  TraversalConstants pc;
};

uint div_up(uint n, uint d)
{
  return (n + d - 1) / d;
}

bool pointInLimacon(vec3 center, float radius, vec3 focal, vec3 point)
{
  focal -= center;
  point -= center;
  vec3 focalToPoint = point - focal;
  float focalToPointLength = max(1e-6f, length(focalToPoint));
  return 2.0f * (radius - dot(focalToPoint, focal) / focalToPointLength) - focalToPointLength >= 0.0f;
}

void main()
{
#if TRAVERSE_PER_INSTANCE
  const uint32_t instanceId = gl_GlobalInvocationID.x;
  if(instanceId >= pc.itemsSize)
    return;
  const uint32_t itemId    = instanceId;
  InstanceArray  instances = InstanceArray(pc.instancesAddress);
  Instance       instance  = instances.array[instanceId];
  MeshArray      meshes    = MeshArray(pc.meshesAddress);
  Mesh           mesh      = meshes.array[instance.meshIndex];
#else
  const uint32_t meshId = gl_GlobalInvocationID.x;
  if(meshId >= pc.itemsSize)
    return;
  const uint32_t itemId = meshId;
  MeshArray      meshes = MeshArray(pc.meshesAddress);
  Mesh           mesh   = meshes.array[meshId];
#endif

  // Get the root node for each mesh
  uint32_t  rootIndex   = 0;
  NodeArray nodes       = NodeArray(mesh.nodesAddress);
  Node      root        = nodes.array[rootIndex];
  NodeRange rootChilren = decodeChildren(root.childrenOrClusters);

  // Create jobs for all children of the root node
  JobStatusArray jobStatus     = JobStatusArray(pc.jobStatusAddress);
  NodeQueueArray nodeQueue     = NodeQueueArray(pc.nodeQueueAddress);
  uint32_t       numChildren   = rootChilren.childCountMinusOne + 1;
  uint32_t       newJobBatches = div_up(numChildren, NODE_BATCH_SIZE);
  atomicAdd(jobStatus.array[0].remaining, int(newJobBatches));
  int writeIndex = atomicAdd(jobStatus.array[0].nodeQueue.write, int(newJobBatches));
  for(uint32_t batchIndex = 0; batchIndex < newJobBatches; ++batchIndex)
    nodeQueue.array[writeIndex + batchIndex] = encodeNodeJob(itemId, rootIndex, batchIndex);

#if IS_RASTERIZATION
#else

  // zero the output cluster count for each instance
  ClusterBLASInfoNVArray blasInput = ClusterBLASInfoNVArray(pc.blasInputAddress);
  blasInput.array[itemId].clusterReferencesCount  = 0;
  blasInput.array[itemId].clusterReferencesStride = 8 /* sizeof(VkDeviceAddress) */;

#if TRAVERSE_PER_INSTANCE
  // BLAS allocation now happens in traverse_write_selected_clusters.comp.glsl
  // for per-instance traversal
  blasInput.array[itemId].clusterReferences = 0;
#else
  uint64_t instanceClustersOffset = atomicAdd(jobStatus.array[0].blasInputClustersAlloc, int(mesh.residentClusterCount));
  blasInput.array[itemId].clusterReferences = pc.blasInputClustersAddress + instanceClustersOffset * 8 /* sizeof(VkDeviceAddress) */;
#endif

  // Copy the data needed for traversal for the nearest few instances after
  // sorting
#if !TRAVERSE_PER_INSTANCE
  // DANGER: assuming the first child is LOD0 root - in practice it was when written.
  // Ideally, there would be a more reliable mechanism to get the mesh's bounding sphere.
  Node lod0Root   = nodes.array[rootChilren.childOffset + 0];
  vec4 lod0Sphere = vec4(lod0Root.bsx, lod0Root.bsy, lod0Root.bsz, lod0Root.bsw);

  SortingMeshInstancesArray sortingMeshInstances = SortingMeshInstancesArray(pc.sortingMeshInstances);
  MeshInstancesArray        meshInstances        = MeshInstancesArray(pc.meshInstances);
  meshInstances.array[meshId].lastInstanceIsRadius = uint8_t(0);
  int nextIndex = 0;
  float closestDistanceLimit; // After this, all instances are guaranteed to produce less detail than those already added
  for(int i = 0; i < TRAVERSAL_NEAREST_INSTANCE_COUNT; ++i)
  {
    uint32_t instanceIndex = uint32_t(sortingMeshInstances.array[meshId].nearest[i] & 0xffffffff);  // index is stored in the lower 4 bytes
    if(instanceIndex != 0xffffffff)
    {
      InstanceArray instances   = InstanceArray(pc.instancesAddress);
      Instance      instance    = instances.array[instanceIndex];
      vec3  cameraInObjectSpace = vec3(instance.transformInv * vec4(pc.traversalParams.viewPosition, 1.0));
      
      // Filter out instances that are outside any 3D limacon defined by closer
      // instance-cameras (i.e. camera positions in object space).
      bool instanceCanIncreaseDetail = false;
      bool furtherInstanceCanIncreaseDetail = true;
      if(i == 0)
      {
        closestDistanceLimit = length(cameraInObjectSpace) + lod0Sphere.w * 2.0f;
        instanceCanIncreaseDetail = true;
      }
      else if(length(cameraInObjectSpace) < closestDistanceLimit)
      {
        for(int j = 0; j < nextIndex && j < TRAVERSAL_NEAREST_INSTANCE_COUNT; ++j)
        {
          if(pointInLimacon(lod0Sphere.xyz, lod0Sphere.w, meshInstances.array[meshId].camerasInObjectSpace[j], cameraInObjectSpace))
          {
            instanceCanIncreaseDetail = true;
            break;
          }
        }
      }
      else
        furtherInstanceCanIncreaseDetail = false;  // once outside closestDistanceLimit, no more instances need be checked

      if(instanceCanIncreaseDetail)
      {
        meshInstances.array[meshId].camerasInObjectSpace[nextIndex] = cameraInObjectSpace;
#if TRAVERSAL_CONSERVATIVE_FALLBACK
        if(i == TRAVERSAL_NEAREST_INSTANCE_COUNT - 1 && furtherInstanceCanIncreaseDetail)
        {
          // If this is the last instance sorted and we haven't been able to
          // filter it out, there may be any number of instances surrounding the
          // camera at this distance, so traverse for the worst case. I.e.
          // direction independent LOD that probably loads the entire level.
          // Guess: this will frequently cause mid-range clumps of instances to
          // be mildly over-detailed. These would be near discrete LOD capable
          // anyway. Provided TRAVERSAL_NEAREST_INSTANCE_COUNT is large enough,
          // should be unlikely to cause catastrophic issues when the camera is
          // very close to instances, which is where continuous LOD works best.
          // IMO it would be safer to accept quality loss than suddenly
          // streaming in entire high detail LOD levels.
          meshInstances.array[meshId].camerasInObjectSpace[nextIndex] = vec3(length(cameraInObjectSpace), 0, 0);
          meshInstances.array[meshId].lastInstanceIsRadius = uint8_t(1);
        }
#endif
        ++nextIndex;
      }
    }
  }
  meshInstances.array[meshId].instanceCount = uint8_t(nextIndex);

  TraverseStatsRef traverseStats = TraverseStatsRef(pc.traverseStatsAddress);
  atomicAdd(traverseStats.d.instancesTraversed, nextIndex);
  atomicMax(traverseStats.d.maxInstancesPerMesh, nextIndex);
#endif
#endif
}
