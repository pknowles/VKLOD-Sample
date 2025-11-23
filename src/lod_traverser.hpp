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

#pragma once

#include <acceleration_structures.hpp>
#include <debug_range_summary.hpp>
#include <optional>
#include <queue>
#include <sample_glsl_compiler.hpp>
#include <sample_profiler.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <traverse_device_host.h>

// Returns parameters that will give the lowest detail mesh
inline shaders::TraversalParams initialTraversalParams()
{
  return shaders::TraversalParams{
      .viewTransform     = glm::translate(glm::mat4(1.0f), {0, 0, 1e10f}),
      .viewPosition      = glm::vec3(0, 0, 1e10f),
      .distanceToUNorm32 = 1.0f,
      .errorOverDistanceThreshold = 1.0f,
      .useOcclusion               = 0,
      .hizViewProj    = glm::translate(glm::mat4(1.0f), {0, 0, 1e10f}),
      .hizSizeFactors = glm::vec4(1.0f),
      .hizViewport    = glm::vec2(1.0f),
      .hizSizeMax     = 1.0f,
  };
}

template <class T>
class ReadbackQueue
{
public:
  class Buffer
  {
  public:
    Buffer(const vko::Device& device, vko::vma::Allocator& allocator)
        : m_device(device,
                   1,
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                       | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   allocator)
        , m_host(device,
                 1,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 allocator)
        , m_mapping(m_host.map())
    {
    }

    const vkobj::Buffer<T>& deviceBuffer() const { return m_device; }

    template <vko::device_and_commands DeviceAndCommands>
    void copyToHost(DeviceAndCommands&                   device,
                    std::optional<vkobj::SemaphoreValue> semaphoreValue,
                    VkCommandBuffer                      cmd)
    {
      VkBufferCopy copy{.srcOffset = 0, .dstOffset = 0, .size = sizeof(T)};
      device.vkCmdCopyBuffer(cmd, m_device, m_host, 1, &copy);
      m_ready = semaphoreValue;
#ifndef NDEBUG
      m_copiedToHost = true;
#endif
    }

    template <vko::device_and_commands DeviceAndCommands>
    std::optional<T> tryRead(DeviceAndCommands& device)
    {
#ifndef NDEBUG
      assert(m_copiedToHost);
#endif
      // DANGER: this object gets reused. If the user forgets to call
      // copyToHost(), m_ready may be stale.
      if(m_ready && !m_ready->isSignaled(device))
        return std::nullopt;
      return std::optional(m_mapping.span()[0]);
    }

    void reset()
    {
#ifndef NDEBUG
      m_copiedToHost = false;
#endif
    }

  private:
    vkobj::Buffer<T>                     m_device;
    vkobj::Buffer<T>                     m_host;
    vkobj::BufferMapping<T>              m_mapping;
    std::optional<vkobj::SemaphoreValue> m_ready;
#ifndef NDEBUG
    bool m_copiedToHost = false;  // purely for state debugging
#endif
  };

  Buffer getFreeBuffer(const vko::Device& device, vko::vma::Allocator& allocator)
  {
    if(!m_free.empty())
    {
      Buffer buffer = std::move(m_free.front());
      m_free.pop();
      return buffer;
    }
    return Buffer(device, allocator);
  }

  void queueBuffer(Buffer&& buffer)
  {
    m_pending.push(std::move(buffer));
    assert(m_pending.size() <= 10);
  }

  template <vko::device_and_commands DeviceAndCommands>
  std::optional<T> latest(DeviceAndCommands& device)
  {
    while(!m_pending.empty())
    {
      auto result = m_pending.front().tryRead(device);
      if(!result)
        break;

      m_latest = result;
      m_pending.front().reset();
      m_free.push(std::move(m_pending.front()));
      m_pending.pop();
    }
    return m_latest;
  }

private:
  std::queue<Buffer> m_free;
  std::queue<Buffer> m_pending;
  std::optional<T>   m_latest;
};

// Encapsulate LOD hierarchy traversal, which computes which clusters to render
// for a given camera view for all instances. Holds intermediate job queue data
// and traversal compute shaders.
// The *instance* traverser produces a BLAS for each instance to match its LOD.
// This has a substantial performance and memory cost and doesn't save much for
// raytracing since the BVH for the highest detail instance must be built either
// way. See the *mesh* traverser below.
class LodInstanceTraverser
{
public:
  LodInstanceTraverser(const vko::Device&   device,
                       vko::vma::Allocator& allocator,
                       vkobj::Staging&      staging,
                       VkQueue              queue,
                       SampleGlslCompiler&  glslCompiler,
                       VkCommandPool        initPool,
                       VkQueue              initQueue,
                       uint32_t             initQueueFamilyIndex,
                       const Scene&         scene,
                       const SceneVK&       sceneVk,
                       uint32_t             maxTotalClusterGroups);
  void         traverseAndBuildBVH(const vko::Device&              device,
                                   vko::vma::Allocator&            allocator,
                                   const shaders::TraversalParams& traversalParams,
                                   const SceneVK&                  sceneVk,
                                   SampleProfiler&                 profiler,
                                   vkobj::SemaphoreValue           submitSemaphore,
                                   VkCommandBuffer                 cmd);
  VkDeviceSize traversalMemory() const
  {
    return m_nodeQueue.sizeBytes() + m_clusterQueue.sizeBytes() + m_jobStatus.sizeBytes();
  }
  VkAccelerationStructureKHR tlas() const { return m_tlas->output(); }
  VkDeviceSize blasDeviceMemory() const { return m_blas.deviceMemory(); }
  VkDeviceSize tlasDeviceMemory() const
  {
    return m_tlas.value().sizeBytes() + m_tlasInfos.sizeBytes();
  }
  template <vko::device_and_commands DeviceAndCommands>
  std::optional<shaders::TraverseStats> stats(DeviceAndCommands& device)
  {
    return m_stats.latest(device);
  }

private:
  void traverse(const vko::Device&              device,
                vko::vma::Allocator&            allocator,
                const shaders::TraversalParams& traversalParams,
                const SceneVK&                  sceneVk,
                const vkobj::Buffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>& blasInput,
                const vkobj::Buffer<VkDeviceAddress>& blasInputClusters,
                SampleProfiler*                       profiler,
                vkobj::SemaphoreValue                 submitSemaphore,
                VkCommandBuffer                       cmd);

  // UBO for traversal constants becuase they're too big for push constants
  vkobj::Buffer<shaders::TraversalConstants> m_traversalConstants;

  // Single descriptor set for all traversal shaders
  vkobj::SingleDescriptorSet m_uboDescriptorSet;

  // Compute shader to initialize the work queues
  vkobj::SimpleComputePipeline<> m_traverseInitPipeline;

  // Main traversal compute shader
  vkobj::SimpleComputePipeline<> m_traversePipeline;

  // Post-traversal verification. The bottom level acceleration structure
  // build cannot be given zero clusters. If Traversal did not produce any
  // clusters, default to the first (which should be lowest detail anyway).
  // TODO: change traversal so this is not needed
  vkobj::SimpleComputePipeline<> m_traverseVerifyPipeline;

  // Single-thread launch to write selectedClusterAlloc to the DispatchIndirect
  // buffer. This could almost be done in traverse_verify.comp.glsl if it
  // weren't for it possibly writting more selected clusters to handle empty
  // traversal output (e.g. in case of out of memory).
  vkobj::SimpleComputePipeline<> m_writeIndirectSizePipeline;

  // Compute shader to write the selected clusters to their individual BLAS input lists
  vkobj::SimpleComputePipeline<> m_writeSelectedClustersPipeline;

  vkobj::Buffer<shaders::EncodedClusterJob> m_clusterQueue;
  BlasArray                                 m_blas;
  vkobj::Buffer<shaders::SelectedCluster>   m_selectedClusters;
  vkobj::Buffer<shaders::DispatchIndirect> m_writeSelectedClustersDispatchIndirect;
  vkobj::Buffer<shaders::EncodedNodeJob> m_nodeQueue;
  vkobj::Buffer<shaders::JobStatus>      m_jobStatus;

  vkobj::Buffer<VkAccelerationStructureInstanceKHR> m_tlasInfos;
  std::optional<Tlas>                   m_tlas;  // optional for delayed init
  ReadbackQueue<shaders::TraverseStats> m_stats;
};

// Encapsulate LOD hierarchy traversal, which computes which clusters to render
// for a given camera view for all instances. Holds intermediate job queue data
// and traversal compute shaders.
// The *mesh* traverser produces a BLAS for each mesh to match the highest LOD
// of the closest few instances. The same BLAS is used for all instances, which
// saves time and memory for raytracing at the cost of over-detailed instances
// in the distance. The trace performance of this is insignificant.
class LodMeshTraverser
{
public:
  LodMeshTraverser(const vko::Device&   device,
                   vko::vma::Allocator& allocator,
                   vkobj::Staging&      staging,
                   VkQueue              queue,
                   SampleGlslCompiler&  glslCompiler,
                   VkCommandPool        initPool,
                   VkQueue              initQueue,
                   uint32_t             initQueueFamilyIndex,
                   const Scene&         scene,
                   const SceneVK&       sceneVk,
                   uint32_t             maxTotalClusterGroups);

  void         traverseAndBuildBVH(const vko::Device&              device,
                                   vko::vma::Allocator&            allocator,
                                   const shaders::TraversalParams& traversalParams,
                                   const SceneVK&                  sceneVk,
                                   SampleProfiler&                 profiler,
                                   vkobj::SemaphoreValue           submitSemaphore,
                                   VkCommandBuffer                 cmd);
  VkDeviceSize traversalMemory() const
  {
    return m_nodeQueue.sizeBytes() + m_clusterQueue.sizeBytes() + m_jobStatus.sizeBytes();
  }
  VkAccelerationStructureKHR tlas() const { return m_tlas->output(); }
  VkDeviceSize blasDeviceMemory() const { return m_blas.deviceMemory(); }
  VkDeviceSize tlasDeviceMemory() const
  {
    return m_tlas.value().sizeBytes() + m_tlasInfos.sizeBytes();
  }
  template <vko::device_and_commands DeviceAndCommands>
  std::optional<shaders::TraverseStats> stats(DeviceAndCommands& device)
  {
    return m_stats.latest(device);
  }

private:
  void traverse(const vko::Device&              device,
                vko::vma::Allocator&            allocator,
                const shaders::TraversalParams& traversalParams,
                const SceneVK&                  sceneVk,
                const vkobj::Buffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>& blasInput,
                const vkobj::Buffer<VkDeviceAddress>& blasInputClusters,
                SampleProfiler*                       profiler,
                vkobj::SemaphoreValue                 submitSemaphore,
                VkCommandBuffer                       cmd);

  vkobj::Buffer<shaders::TraversalConstants> m_traversalConstants;
  vkobj::SingleDescriptorSet                 m_uboDescriptorSet;

  // Compute shader to find the k nearest instances per mesh
  vkobj::SimpleComputePipeline<shaders::SortInstancesConstant> m_traverseSortInstances;

  // Compute shader to initialize the work queues
  vkobj::SimpleComputePipeline<> m_traverseInit;

  // Main traversal compute shader
  vkobj::SimpleComputePipeline<> m_traverse;

  // Post-traversal verification. The bottom level acceleration structure
  // build cannot be given zero clusters. If Traversal did not produce any
  // clusters, default to the first (which should be lowest detail anyway).
  // TODO: change traversal so this is not needed
  vkobj::SimpleComputePipeline<> m_traverseVerify;

  // Compute shader to assign BLAS addresses to each instance
  vkobj::SimpleComputePipeline<shaders::WriteInstancesConstant> m_instanceWriter;

  vkobj::Buffer<shaders::EncodedNodeJob>            m_nodeQueue;
  vkobj::Buffer<shaders::EncodedClusterJob>         m_clusterQueue;
  vkobj::Buffer<shaders::JobStatus>                 m_jobStatus;
  vkobj::Buffer<shaders::MeshInstances>             m_meshInstances;
  vkobj::Buffer<shaders::SortingMeshInstances>      m_sortingMeshInstances;
  vkobj::Buffer<VkDeviceAddress>                    m_blasAddresses;
  BlasArray                                         m_blas;
  vkobj::Buffer<VkAccelerationStructureInstanceKHR> m_tlasInfos;
  std::optional<Tlas>                   m_tlas;  // optional for delayed init
  ReadbackQueue<shaders::TraverseStats> m_stats;

  const vko::Device* m_device = nullptr;
};

namespace shaders {

inline std::ostream& operator<<(std::ostream& os, const shaders::MeshInstances& x)
{
  using numerical_chars::operator<<;
  using ::operator<<;
  os << "MeshInstances{\n";
  for(int i = 0; i < TRAVERSAL_NEAREST_INSTANCE_COUNT; ++i)
  {
    if(i != 0)
      os << ", ";
    os << "  [" << i << "]={\n";
    os << "    cameraInObjectSpace " << x.camerasInObjectSpace[i] << "\n";
    os << "  }";
    os << "  instanceCount " << x.instanceCount << "\n";
    os << "  lastInstanceIsRadius " << x.lastInstanceIsRadius << "\n";
  }
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const shaders::SortingMeshInstances& x)
{
  using numerical_chars::operator<<;
  os << "SortingMeshInstances{\n";
  for(int i = 0; i < TRAVERSAL_NEAREST_INSTANCE_COUNT; ++i)
  {
    os << (i == 0 ? "  " : ", ");
    os << "[" << i << "]={\n";
    os << "    nearest w0 " << (x.nearest[i] >> 32) << "\n";
    os << "    nearest w1 " << (x.nearest[i] & 0xffffffff) << "\n";
    os << "  }";
  }
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const shaders::SelectedCluster& x)
{
  using numerical_chars::operator<<;
  os << "SelectedCluster{\n";
  os << "    objectIndex " << x.objectIndex << "\n";
  os << "    clusterAddress " << x.clusterAddress << "\n";
  os << "}";
  return os;
}

}  // namespace shaders
