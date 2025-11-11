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

#ifndef TRAVERSE_DEVICE_HOST_H
#define TRAVERSE_DEVICE_HOST_H

#include "shaders_buffer_ref.h"
#include "shaders_glm.h"
#include "shaders_lod_structs.h"
#include "shaders_scene.h"

#define TRAVERSAL_WORKGROUP_SIZE 64

#ifdef __cplusplus
namespace shaders {
#endif  // __cplusplus

const int BTraversalConstants = 0;
const int BTraversalHiZTex    = 1;

#define NODE_BATCH_SIZE 8     // Power of two in range [1, 32]
#define CLUSTER_BATCH_SIZE 8  // Power of two in range [8, 32]

// Jobs are encoded as 64bit integers, otherwise the scene would be limited to
// just ~2^15 instances.
#define INSTANCE_BITS 32
#define NODE_BITS 26
#define CLUSTER_BATCH_BITS 5
#define MAX_CLUSTERS_PER_NODE 256
#define TRAVERSAL_MAX_INSTANCES (1llu << INSTANCE_BITS)
#define TRAVERSAL_MAX_NODES (1llu << NODE_BITS)

#ifndef __cplusplus
struct NodeJob
{
  uint64_t ready;        // bits: 1
  uint64_t parentIndex;  // bits: NODE_BITS
  uint64_t batchIndex;   // bits: 5
  uint64_t objectIndex;  // bits: INSTANCE_BITS
};
#else
using EncodedNodeJob = uint64_t;
#endif

#ifndef __cplusplus
struct ClusterJob
{
  uint64_t ready;        // bits: 1
  uint64_t parentIndex;  // bits: NODE_BITS
  uint64_t batchIndex;   // bits: CLUSTER_BATCH_INDEX_BITS
  uint64_t objectIndex;  // bits: INSTANCE_BITS
};
#else
using EncodedClusterJob = uint64_t;
#endif

#ifndef __cplusplus
uint64_t encodeNodeJob(uint64_t objectIndex, uint64_t parentIndex, uint64_t batchIndex)
{
  uint64_t ready  = 1;
  uint64_t result = 0;
  pushBits(result, ready, 1);
  pushBits(result, parentIndex, NODE_BITS);
  pushBits(result, batchIndex, 5);
  pushBits(result, objectIndex, INSTANCE_BITS);
  return result;
}

NodeJob decodeNodeJob(uint64_t encoded)
{
  NodeJob result;
  result.objectIndex = popBits(encoded, INSTANCE_BITS);
  result.batchIndex  = popBits(encoded, 5);
  result.parentIndex = popBits(encoded, NODE_BITS);
  uint64_t ready     = popBits(encoded, 1);
  return result;
}

uint64_t encodeClusterJob(uint64_t objectIndex, uint64_t parentIndex, uint64_t batchIndex)
{
  uint64_t ready  = 1;
  uint64_t result = 0;
  pushBits(result, ready, 1);
  pushBits(result, parentIndex, NODE_BITS);
  pushBits(result, batchIndex, 5);
  pushBits(result, objectIndex, INSTANCE_BITS);
  return result;
}

ClusterJob decodeClusterJob(uint64_t encoded)
{
  ClusterJob result;
  result.objectIndex = popBits(encoded, INSTANCE_BITS);
  result.batchIndex  = popBits(encoded, 5);
  result.parentIndex = popBits(encoded, NODE_BITS);
  uint64_t ready     = popBits(encoded, 1);
  return result;
}
#endif

struct QueueStatus
{
  int read;
  int write;
};

struct JobStatus
{
  QueueStatus nodeQueue;
  QueueStatus clusterQueue;

  // Total remaining work items in all queues. Batch counts, not
  // individual jobs (1 count for 4 jobs per node queue item and 1 count
  // for 8 jobs per cluster queue item).
  int remaining;

  // Only used for per-instance traversal to allocate items in
  // 'selectedClusters' before grouping into blasInputClustersAddress. Ends up
  // with the same value as blasInputClustersAlloc.
  int selectedClusterAlloc;

  // Allocates ranges of clusters in 'blasInputClustersAddress'
  int blasInputClustersAlloc;
};

struct TraversalParams
{
  mat4     viewTransform;
  vec3     viewPosition;
  float    distanceToUNorm32;
  float    errorOverDistanceThreshold;
  uint32_t useOcclusion;
  mat4     hizViewProj;
  vec4     hizSizeFactors;
  vec2     hizViewport;
  float    hizSizeMax;
};

#define TRAVERSAL_NEAREST_INSTANCE_COUNT 8

// When the above instance count is exhausted, fall back to a straight distance
// from the object space origin check, i.e. omnidirectional LOD.
#define TRAVERSAL_CONSERVATIVE_FALLBACK 1

struct TraverseStats
{
  uint32_t triangleCount;
  uint32_t instancesTraversed;
  uint32_t maxInstancesPerMesh;
  uint32_t errorNodeQueueOverflow;
  uint32_t errorClusterQueueOverflow;
  uint32_t errorTraversalIncomplete;
  uint32_t errorEmptyTraversalResults;
  uint32_t errorBlasInputOverflow;
};

struct MeshInstances
{
  // Rather than provide the instance transform, it's cheaper to compute the
  // camera's position in object space in traverse_init.comp.glsl. This is
  // possible because the LOD metric is purely distance based.
  vec3 camerasInObjectSpace[TRAVERSAL_NEAREST_INSTANCE_COUNT];

  // Number of valid items in camerasInObjectSpace to be considered during
  // traversal.
  uint8_t instanceCount;

  // When non-zero, assume the x component of camerasInObjectSpace[instanceCount
  // - 1] is a radius around the root bounding sphere that remaining cameras
  // must be outside
  uint8_t lastInstanceIsRadius;
};

// This avoids the allocation problem of not knowing the number of clusters that
// will be selected during traversal. We first write all selected clusters to a
// global buffer, arbitrarily sized at launch, and then linearize them per
// instance or mesh. For per-instance traversal, out-of-memory can still occur
// if too many instances are near the camera, in which case they fall back to
// the min-detail BLAS.
struct SelectedCluster
{
  uint32_t objectIndex;  // instance or mesh, depending on TRAVERSE_PER_INSTANCE
  uint64_t clusterAddress;
};

struct SortingMeshInstances
{
  uint64_t nearest[TRAVERSAL_NEAREST_INSTANCE_COUNT];  // {sortKey, instanceIndex} pairs updated with atomics
};

// VkDispatchIndirectCommand
struct DispatchIndirect
{
  uint32_t x;
  uint32_t y;
  uint32_t z;
};

struct TraversalConstants
{
  TraversalParams traversalParams;

  DEVICE_ADDRESS(Mesh) meshesAddress;
  DEVICE_ADDRESS(Instance) instancesAddress;
  DEVICE_ADDRESS(EncodedNodeJob) nodeQueueAddress;
  DEVICE_ADDRESS(EncodedClusterJob) clusterQueueAddress;
  DEVICE_ADDRESS(JobStatus) jobStatusAddress;
  DEVICE_ADDRESS(TraverseStats) traverseStatsAddress;

  DEVICE_ADDRESS(ClusterBLASInfoNV) blasInputAddress;
  DEVICE_ADDRESS(uint64_t) blasInputClustersAddress;

  DEVICE_ADDRESS(DrawCluster) drawClustersAddress;
  DEVICE_ADDRESS(DrawMeshTasksIndirect) drawMeshTasksIndirectAddress;
  DEVICE_ADDRESS(DrawStats) drawStatsAddress;

  DEVICE_ADDRESS(MeshInstances) meshInstances;                // null if TRAVERSE_PER_INSTANCE
  DEVICE_ADDRESS(SortingMeshInstances) sortingMeshInstances;  // null if TRAVERSE_PER_INSTANCE

  DEVICE_ADDRESS(SelectedCluster) selectedClusters;                        // null if not TRAVERSE_PER_INSTANCE
  DEVICE_ADDRESS(DispatchIndirect) writeSelectedClustersDispatchIndirect;  // null if not TRAVERSE_PER_INSTANCE
  uint32_t maxSelectedClusters;

  uint32_t nodeQueueSize;
  uint32_t clusterQueueSize;
  uint32_t itemsSize;  // Instances or Meshes depending on TRAVERSE_PER_INSTANCE
  uint32_t drawClustersSize;
};

struct SortInstancesConstant
{
  TraversalParams traversalParams;
  DEVICE_ADDRESS(Instance) instances;
  DEVICE_ADDRESS(Mesh) meshes;
  DEVICE_ADDRESS(SortingMeshInstances) sortingMeshInstances;
  uint32_t instancesSize;
};

struct WriteInstancesConstant
{
  DEVICE_ADDRESS(Instance) instances;
  DEVICE_ADDRESS(uint64_t) meshBlasAddresses;
  DEVICE_ADDRESS(InstanceInfo) tlasInfos;
  uint32_t instancesSize;
};

#ifndef __cplusplus
DECL_BUFFER_REF(NodeArray, Node);
DECL_MUTABLE_BUFFER_REF(JobStatusArray, JobStatus);
DECL_MUTABLE_BUFFER_REF(ClusterBLASInfoNVArray, ClusterBLASInfoNV);
DECL_MUTABLE_BUFFER_REF(TraversalParamsArray, TraversalParams);
DECL_MUTABLE_BUFFER_REF(MeshInstancesArray, MeshInstances);
DECL_MUTABLE_BUFFER_REF(SortingMeshInstancesArray, SortingMeshInstances);
DECL_MUTABLE_BUFFER_REF(SelectedClusterArray, SelectedCluster);
DECL_MUTABLE_BUFFER_REF(DispatchIndirectArray, DispatchIndirect);
DECL_BUFFER_REF(NodeQueueArray, uint64_t);
DECL_BUFFER_REF(ClusterQueueArray, uint64_t);
DECL_MUTABLE_BUFFER_REF(DrawClusterArray, DrawCluster);
DECL_MUTABLE_BUFFER_REF(InstanceInfoArray, InstanceInfo);
DECL_BUFFER_REF_SINGLE_BASE(TraverseStatsRef, , TraverseStats, 4)  // TODO: validate alignment like array refs
#endif

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif  // TRAVERSE_DEVICE_HOST_H
