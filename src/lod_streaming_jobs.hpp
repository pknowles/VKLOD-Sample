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

#include <any>
#include <deque>
#include <memory_resource>
#include <sample_glsl_compiler.hpp>
#include <sample_raytracing_objects.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <shaders_stream.h>
#include <unordered_set>
#include <vulkan/vulkan_core.h>

namespace streaming {

// Container to track ClusterGroupGeometryVk and Allocation since they are
// each created at different times. This is the granulartiy of streaming,
// split into two separate allocations.
struct ClusterGroupVk
{
  ClusterGroupGeometryVk geometry;
  PoolMemory             clasAddresses;
  ClusterGroupVk(ClusterGroupGeometryVk&& geometry_, PoolMemory&& clasAddresses_)
      : geometry(std::move(geometry_))
      , clasAddresses(std::move(clasAddresses_))
  {
  }
  ClusterGroupVk(ClusterGroupGeometryVk&& geometry_)
      : geometry(std::move(geometry_))
  {
  }
};

// Separate types so that a function can take a specific program as a parameter
class MakeRequestsProgram : public vkobj::SimpleComputePipeline<shaders::BuildRequestsConstants>
{
public:
  MakeRequestsProgram(const vko::Device& device, SampleGlslCompiler& glslCompiler)
      : vkobj::SimpleComputePipeline<shaders::BuildRequestsConstants>(device, glslCompiler, "stream_make_requests.comp.glsl")
  {
  }
};

class ModifyGroupsProgram : public vkobj::SimpleComputePipeline<shaders::StreamGroupsConstants>
{
public:
  ModifyGroupsProgram(const vko::Device& device, SampleGlslCompiler& glslCompiler)
      : vkobj::SimpleComputePipeline<shaders::StreamGroupsConstants>(device, glslCompiler, "stream_modify_groups.comp.glsl")
  {
  }
};

class FillClasInputProgram : public vkobj::SimpleComputePipeline<shaders::FillClasInputConstants>
{
public:
  FillClasInputProgram(const vko::Device& device, SampleGlslCompiler& glslCompiler)
      : vkobj::SimpleComputePipeline<shaders::FillClasInputConstants>(device, glslCompiler, "stream_fill_clas_input.comp.glsl")
  {
  }
};

class PackClasProgram : public vkobj::SimpleComputePipeline<shaders::PackClasConstants>
{
public:
  PackClasProgram(const vko::Device& device, SampleGlslCompiler& glslCompiler)
      : vkobj::SimpleComputePipeline<shaders::PackClasConstants>(device, glslCompiler, "stream_pack_clas.comp.glsl")
  {
  }
};

// A reusable GPU buffer of cluster groups indices to either load or unload.
// These are passed in a queue from the render thread to the streaming thread.
// The streaming thread will block until the data is ready.
class RequestList
{
public:
  RequestList(const vko::Device&   device,
              vko::vma::Allocator& allocator,
              vkobj::Staging&      staging,
              VkQueue              queue,
              uint32_t             maxRequests);
  void gather(const vko::Device&          device,
              MakeRequestsProgram&        program,
              vko::DeviceBuffer<uint8_t>& groupNeededFlags,
              vko::SemaphoreValue         promisedSubmitSemaphoreState,
              VkCommandBuffer             cmd);
  void download(const vko::Device&                  device,
                vkobj::Staging&                     staging,
                std::vector<shaders::GroupRequest>& result)
  {
    result.clear();

    // First download the request list count
    uint32_t count;
    {
      // Download the single StreamRequestCounts struct
      auto countFuture =
          vko::download(staging, device, vko::BufferSpan(m_requestCounts).subspan(0, 1));

      memoryBarrier(device, staging.commandBuffer(), VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT);
      device.vkCmdFillBuffer(staging.commandBuffer(), m_requestCounts,
                             offsetof(shaders::StreamRequestCounts, requestsCount),
                             sizeof(shaders::StreamRequestCounts::requestsCount),
                             0);  // zero the count for buffer reuse

      // Submit with custom wait semaphores for synchronization with render thread
      std::array waits{m_readySemaphore.value().waitSemaphoreInfo(VK_PIPELINE_STAGE_2_TRANSFER_BIT)};
      std::array<VkSemaphoreSubmitInfo, 0> noSignals{};
      staging.submit(waits, noSignals);

      // Take the min. requestsCount may overshoot due to parallel atomics. Note
      // the future.get() here will block until the above submission completes.
      count = std::min(m_maxRequests, countFuture.get(device)[0].requestsCount);
    }

    // Download the data, if there is any
    if(count > 0u)
    {
      result.resize(count);
      size_t offset         = 0;
      auto   downloadHandle = vko::downloadForEach(
          staging, device, vko::BufferSpan(m_requests).subspan(0, count),
          [&result, &offset](VkDeviceSize, std::span<const shaders::GroupRequest> chunk) {
            std::ranges::copy(chunk, result.begin() + std::ptrdiff_t(offset));
            offset += chunk.size();
          });
      staging.submit();
      downloadHandle.wait(device);
    }
  }
  size_t memoryUsage() const
  {
    return m_requests.sizeBytes() + m_requestCounts.sizeBytes();
  }

private:
  uint32_t m_maxRequests;
  std::optional<vkobj::SemaphoreValue> m_readySemaphore;  // render queue to streaming thread sync
  vko::DeviceBuffer<shaders::GroupRequest> m_requests;
  vko::DeviceBuffer<shaders::StreamRequestCounts> m_requestCounts;  // single element
};

// A reusable GPU buffer of cluster groups to insert in or remove from the
// renderable data. These are passed in a queue from the streaming thread to the
// render thread. They will not be added to the queue until ready, so no
// synchronization in that direction is needed and the render thread will not be
// interrupted.
class GroupModsList
{
public:
  GroupModsList(const vko::Device& device, vko::vma::Allocator& allocator, uint32_t maxLoadUnloads);
  template <typename StagingAllocator>
  shaders::StreamGroupModsList write(const vko::Device& device,
                                     StagingAllocator&  staging,
                                     std::span<const shaders::LoadGroup> loads,
                                     std::span<const shaders::UnloadGroup> unloads,
                                     std::span<const ClusteredMesh> meshes,
                                     std::span<const uint32_t> meshInstanceCounts)
  {
    // Make sure this object is not still being read from by the GPU after a
    // previous call to modifyGroups()
    if(m_reuseSemaphore)
      m_reuseSemaphore->wait(device);

    // Write the counts, which will be used to dispatch compute threads
    m_groupModsList.loadGroupCount   = uint32_t(loads.size());
    m_groupModsList.unloadGroupCount = uint32_t(unloads.size());

    // Write the LoadGroup and UnloadGroup arrays that contain references to
    // streamed data that must be added or removed
    if(!loads.empty())
      vko::upload(staging, device, loads, m_loads);
    if(!unloads.empty())
      vko::upload(staging, device, unloads, m_unloads);

    // Record running totals of what was loaded and unloaded. This may be used to
    // determine worst case allocations needed
    m_clusterCountDelta         = 0;
    m_instanceClusterCountDelta = 0;
    for(const shaders::LoadGroup& load : loads)
    {
      assert(load.groupData.clusterCount
             == meshes[load.meshIndex].groupClusterRanges[load.groupIndex].count);
      m_clusterCountDelta += load.groupData.clusterCount;
      m_instanceClusterCountDelta +=
          load.groupData.clusterCount * meshInstanceCounts[load.meshIndex];
    }
    for(const shaders::UnloadGroup& unload : unloads)
    {
      uint32_t groupClusterCount =
          meshes[unload.meshIndex].groupClusterRanges[unload.groupIndex].count;
      m_clusterCountDelta -= groupClusterCount;
      m_instanceClusterCountDelta -=
          groupClusterCount * meshInstanceCounts[unload.meshIndex];
      assert(unload.groupIndex
             != uint32_t(meshes[unload.meshIndex].groupGeneratedGroups.size() - 1));  // should never unload root pages
    }

    // Debugging - check there are no duplicates between loads/unloads
#if !defined(NDEBUG)
    std::set<std::pair<uint32_t, uint32_t>> loadids;
    std::set<std::pair<uint32_t, uint32_t>> unloadids;
    for(const shaders::LoadGroup& load : loads)
      loadids.insert({load.meshIndex, load.groupIndex});
    for(const shaders::UnloadGroup& unload : unloads)
      unloadids.insert({unload.meshIndex, unload.groupIndex});
    for(const shaders::LoadGroup& load : loads)
      assert(unloadids.count({load.meshIndex, load.groupIndex}) == 0);
    for(const shaders::UnloadGroup& unload : unloads)
      assert(loadids.count({unload.meshIndex, unload.groupIndex}) == 0);
#endif

    // Returns the uploaded structures, which contain cluster geometry and can be used
    // to fill CLAS build input structures
    return m_groupModsList;
  }

  // Called on the render thread to patch pointers (meshPointers) to cluster
  // groups. These may include writing pointers to new groups or zeroing
  // pointers to those just unloaded. May return objects to free after the
  // commands have been completed and the semaphore has been signalled.
  // NOTE: unloadGarbageSemaphore is only valid when unloadGarbage is not empty
  void modifyGroups(const vko::Device&            device,
                    ModifyGroupsProgram&          program,
                    vkobj::Buffer<shaders::Mesh>& meshPointers,
                    vkobj::SemaphoreValue         promisedSubmitSemaphoreState,
                    VkCommandBuffer               cmd,
                    uint64_t& totalResidentClusters,  // TODO: clean up too many params
                    uint64_t& totalResidentInstanceClusters);

  size_t memoryUsage() const
  {
    return m_loads.sizeBytes() + m_unloads.sizeBytes();
  }

private:
  vko::DeviceBuffer<shaders::LoadGroup>   m_loads;
  vko::DeviceBuffer<shaders::UnloadGroup> m_unloads;
  std::optional<vkobj::SemaphoreValue> m_reuseSemaphore;  // render queue to streaming thread sync
  shaders::StreamGroupModsList m_groupModsList;
  int32_t                      m_clusterCountDelta         = 0;
  int32_t                      m_instanceClusterCountDelta = 0;
};

enum Result : uint32_t
{
  eSuccess,       // keep loading/unloading
  eDelay,         // out of memory or no need to unload, delay until later
  eStopAndRetry,  // pipeline full, retry later
};

// A class to manage two queues:
// 1. Top level requests. This may fill up, and drain over time. If a
//    load/unload is no longer needed, the request may be skipped within this
//    queue. Order is not important.
// 2. Dependency expanded requests. Loads/unloads must happen in order and
//    cannot be skipped.
class RequestDependencyPipeline
{
public:
  RequestDependencyPipeline(size_t globalGroupCount)
      : m_globalGroupsExpected(globalGroupCount, false)
      , m_globalGroupsNeeded(globalGroupCount, false)
      , m_globalGroupsLoaded(globalGroupCount, false)
  {
  }

  // Insert a batch of requests from the render thread
  void queueRequests(std::span<const shaders::GroupRequest> requests);

  // Extract a single batch of loads/unloads, making sure cluster groups are
  // always loaded and unloaded in order of dependencies.
  // TODO: while an individaul RequestList cannot have a load and unload for
  // the same page, m_topLevelRequests can. It is then possible that a batch
  // is created with both a load and an unload for the same page. This could
  // result in cracks in a mesh.
  void dequeueLoadUnloadBatch(const Scene& scene,
                              const std::function<Result(uint32_t, uint32_t)>& emitLoad,
                              const std::function<Result(uint32_t, uint32_t)>& emitUnload);

  uint32_t pendingRequests() const { return m_pendingRequests; }

private:
  // Depth first search. Related to topological sort
  Result loadGroupDependenciesRecursive(offset_span<offset_span<uint32_t>> meshGroupGeneratedGroups,
                                        uint32_t meshGroupOffset,
                                        uint32_t meshIndex,
                                        uint32_t meshGroupIndex,
                                        const std::function<Result(uint32_t, uint32_t)>& emitLoad);
  Result unloadGroupDependenciesRecursive(
      offset_span<offset_span<uint32_t>> meshGroupGeneratedGroups,
      offset_span<offset_span<uint32_t>> meshGroupGeneratingGroups,
      uint32_t                           meshGroupOffset,
      uint32_t                           meshIndex,
      uint32_t                           meshGroupIndex,
      const std::function<Result(uint32_t, uint32_t)>& emitUnload);

  std::vector<bool> m_globalGroupsExpected;  // Allow new request to shortcut queued requests in m_topLevelRequests
  std::vector<bool> m_globalGroupsNeeded;  // Pin top level requests so that an orphaned dependency doesn't unload them
  std::vector<bool> m_globalGroupsLoaded;  // Keep track of what load events have actually been issued
  std::unordered_set<uint32_t> m_batchUnloads;  // Don't reload dependencies if they were unloaded in the same batch
  std::deque<shaders::GroupRequest>  m_topLevelRequests;
  std::vector<shaders::GroupRequest> m_delayedRequests;
  bool                               m_useDelayedRequests = false;
  uint32_t                           m_pendingRequests    = 0;
};

}  // namespace streaming

inline std::ostream& operator<<(std::ostream& os, const shaders::GroupRequest& x)
{
  os << "GroupRequest{";
  os << "globalGroup " << x.decoded.globalGroup << " ";
  os << "load " << x.decoded.load;
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const shaders::LoadGroup& x)
{
  os << "LoadGroup{";
  os << "meshIndex " << x.meshIndex << " ";
  os << "groupIndex " << x.groupIndex << " ";
  os << "groupData " << x.groupData;
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const shaders::UnloadGroup& x)
{
  os << "UnloadGroup{";
  os << "meshIndex " << x.meshIndex << " ";
  os << "groupIndex " << x.groupIndex;
  os << "}";
  return os;
}
