/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#extension GL_EXT_debug_printf : require

#include "traverse_device_host.h"

#ifndef TRAVERSE_PER_INSTANCE
#error "Must define TRAVERSE_PER_INSTANCE"
#endif

#if IS_RASTERIZATION
#error "Not for rasterization"
#endif

layout(local_size_x = TRAVERSAL_WORKGROUP_SIZE) in;

layout(set = 0, binding = BTraversalConstants, scalar) uniform TraverseConstants_
{
  TraversalConstants pc;
};

void main()
{
  uint32_t    selectedClusterIndex      = gl_GlobalInvocationID.x;
  JobStatusArray jobStatus = JobStatusArray(pc.jobStatusAddress);

  uint32_t selectedClusterCount = min(jobStatus.array[0].selectedClusterAlloc, pc.maxSelectedClusters);
  if(selectedClusterIndex >= selectedClusterCount)
    return;

  SelectedClusterArray   selectedClusters = SelectedClusterArray(pc.selectedClusters);
  SelectedCluster        selectedCluster  = selectedClusters.array[selectedClusterIndex];
  uint32_t               itemId           = selectedCluster.objectIndex;
  ClusterBLASInfoNVArray blasInput        = ClusterBLASInfoNVArray(pc.blasInputAddress);

  uint           blasWriteIndex           = atomicAdd(blasInput.array[itemId].clusterReferencesCount, 1);
  MutUint64Array blasInputClusters        = MutUint64Array(blasInput.array[itemId].clusterReferences);
  blasInputClusters.array[blasWriteIndex] = selectedCluster.clusterAddress;
}
