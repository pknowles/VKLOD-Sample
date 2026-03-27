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

#include <lod_streaming_jobs.hpp>
#include <sample_vulkan_objects.hpp>
#include <shaders_scene.h>
#include <vulkan/vulkan_core.h>

namespace streaming {

RequestList::RequestList(const vko::Device&   device,
                         vko::vma::Allocator& allocator,
                         vkobj::Staging&      staging,
                         VkQueue              queue,
                         uint32_t             maxRequests)
    : m_maxRequests(maxRequests)
    , m_requests(device,
                 maxRequests,
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                     | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 allocator)
    , m_requestCounts([&]() {
      // Initialize m_requestCounts contents
      shaders::StreamRequestCounts init{
          .requestsCount = 0,
          .requestsSize  = maxRequests,
      };
      auto buffer = vko::upload(staging, device, allocator, std::vector{init},
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                    | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
      memoryBarrier(device, staging.commandBuffer(), VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      staging.submit();
      // TODO: can this wait be moved to the caller? do we even need it?
      vko::check(device.vkQueueWaitIdle(queue));
      return buffer;
    }())
{
}

void RequestList::gather(const vko::Device&          device,
                         MakeRequestsProgram&        program,
                         vko::DeviceBuffer<uint8_t>& groupNeededFlags,
                         vko::SemaphoreValue promisedSubmitSemaphoreState,
                         VkCommandBuffer     cmd)
{
  shaders::BuildRequestsConstants constants{
      .groupNeededFlagsAddress    = groupNeededFlags,
      .streamRequestCountsAddress = m_requestCounts,
      .requestsAddress            = m_requests,
      .groupCount                 = uint32_t(groupNeededFlags.size()),
  };
  memoryBarrier(device, cmd, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  device.vkCmdPushConstants(cmd, program.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(constants), &constants);
  device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, program.pipeline);
  device.vkCmdDispatch(cmd, div_ceil(uint32_t(groupNeededFlags.size()), uint32_t(STREAM_WORKGROUP_SIZE)),
                       1, 1);
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  m_readySemaphore = promisedSubmitSemaphoreState;
}

GroupModsList::GroupModsList(const vko::Device& device, vko::vma::Allocator& allocator, uint32_t maxLoadUnloads)
    : m_loads(device,
              maxLoadUnloads,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
              allocator)
    , m_unloads(device,
                maxLoadUnloads,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                allocator)
{
  m_groupModsList.loadGroupsAddress   = m_loads;
  m_groupModsList.loadGroupCount      = 0;
  m_groupModsList.unloadGroupsAddress = m_unloads;
  m_groupModsList.unloadGroupCount    = 0;
}

void GroupModsList::modifyGroups(const vko::Device&            device,
                                 ModifyGroupsProgram&          program,
                                 vkobj::Buffer<shaders::Mesh>& meshPointers,
                                 vkobj::SemaphoreValue promisedSubmitSemaphoreState,
                                 VkCommandBuffer cmd,
                                 uint64_t&       totalResidentClusters,
                                 uint64_t&       totalResidentInstanceClusters)
{
  // Update running totals
  assert(int64_t(totalResidentClusters) >= -m_clusterCountDelta);
  assert(int64_t(totalResidentInstanceClusters) >= -m_instanceClusterCountDelta);
  totalResidentClusters =
      uint64_t(std::max<int64_t>(0, int64_t(totalResidentClusters) + m_clusterCountDelta));
  totalResidentInstanceClusters = uint64_t(std::max<int64_t>(
      0, int64_t(totalResidentInstanceClusters) + m_instanceClusterCountDelta));

  // Launch compute kernels to process loads and unloads
  memoryBarrier(device, cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  if(m_groupModsList.loadGroupCount)
  {
    shaders::StreamGroupsConstants loadConstants{
        .meshesAddress = meshPointers,
        .mods          = m_groupModsList,
        .load          = 1u,
    };
    device.vkCmdPushConstants(cmd, program.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(loadConstants), &loadConstants);
    device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, program.pipeline);
    device.vkCmdDispatch(cmd, div_ceil(m_groupModsList.loadGroupCount, uint32_t(STREAM_WORKGROUP_SIZE)),
                         1, 1);
  }
  if(m_groupModsList.unloadGroupCount)
  {
    shaders::StreamGroupsConstants unloadConstants{
        .meshesAddress = meshPointers,
        .mods          = m_groupModsList,
        .load          = 0,
    };
    device.vkCmdPushConstants(cmd, program.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(unloadConstants), &unloadConstants);
    device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, program.pipeline);
    device.vkCmdDispatch(cmd, div_ceil(m_groupModsList.unloadGroupCount, uint32_t(STREAM_WORKGROUP_SIZE)),
                         1, 1);
  }
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Record the command buffer's semaphore so this object doesn't get written to
  // before the above compute shaders are finished
  m_reuseSemaphore = promisedSubmitSemaphoreState;
}

void RequestDependencyPipeline::queueRequests(std::span<const shaders::GroupRequest> requests)
{
  if(requests.empty())
    return;

  // Keep track of the most recent state at the back of the queue so short
  // pulses can be ignored
  for(const shaders::GroupRequest& request : requests)
  {
    m_globalGroupsExpected[request.decoded.globalGroup] = bool(request.decoded.load);
    if(m_globalGroupsNeeded[request.decoded.globalGroup] != bool(request.decoded.load))
    {
      m_pendingRequests++;
    }
    else
    {
      assert(m_pendingRequests > 0);
      m_pendingRequests--;
    }
  }

  // Add all the requests to the main queue. This is allowed to grow
  // indefinitely
  m_topLevelRequests.insert(m_topLevelRequests.end(), requests.begin(), requests.end());

  if(!m_delayedRequests.empty())
    m_useDelayedRequests = true;
}

void RequestDependencyPipeline::dequeueLoadUnloadBatch(
    const Scene&                                     scene,
    const std::function<Result(uint32_t, uint32_t)>& emitLoad,
    const std::function<Result(uint32_t, uint32_t)>& emitUnload)
{
  m_batchUnloads.clear();
  while(!m_topLevelRequests.empty()
        || (m_useDelayedRequests && !m_delayedRequests.empty()))
  {
    if(m_topLevelRequests.empty())
    {
      assert(m_useDelayedRequests && !m_delayedRequests.empty());
      m_topLevelRequests.insert(m_topLevelRequests.end(),
                                m_delayedRequests.begin(), m_delayedRequests.end());
      m_delayedRequests.clear();
      m_useDelayedRequests = false;
      assert(!m_topLevelRequests.empty());
    }

    // Get the next top level request, but don't consume it just yet
    shaders::GroupRequest request = m_topLevelRequests.front();

    // Skip processing requests made outdated by newer ones from
    // queueRequests().
    if(m_globalGroupsExpected[request.decoded.globalGroup]
       != bool(request.decoded.load))
    {
      m_topLevelRequests.pop_front();
      // m_pendingRequests should already be adjusted for this
      continue;
    }

    // With the above filter we now also need to ignore inevitable duplicate
    // requests that are still in the queue
    if(m_globalGroupsNeeded[request.decoded.globalGroup] == bool(request.decoded.load))
    {
      m_topLevelRequests.pop_front();
      // m_pendingRequests should already be adjusted for this
      continue;
    }

    // TODO: avoid upper_bound binary search
    auto offsetIt =
        std::ranges::upper_bound(scene.meshGroupOffsets, request.decoded.globalGroup) - 1;
    uint32_t meshGroupOffset = *offsetIt;
    uint32_t meshIndex = uint32_t(offsetIt - scene.meshGroupOffsets.begin());
    uint32_t meshGroupIndex = request.decoded.globalGroup - *offsetIt;
    Result   result;
    if(request.decoded.load)
    {
      result = loadGroupDependenciesRecursive(scene.meshes[meshIndex].groupGeneratedGroups,
                                              meshGroupOffset, meshIndex,
                                              meshGroupIndex, emitLoad);

      // Pin the page to prevent automatically unloading it due to a lost
      // dependency.
      if(result != Result::eDelay)
        m_globalGroupsNeeded[request.decoded.globalGroup] = true;
    }
    else
    {
      // Un-pin this group as needed to allow it and dependencies to unload
      m_globalGroupsNeeded[request.decoded.globalGroup] = false;
      result = unloadGroupDependenciesRecursive(scene.meshes[meshIndex].groupGeneratedGroups,
                                                scene.meshes[meshIndex].groupGeneratingGroups,
                                                meshGroupOffset, meshIndex,
                                                meshGroupIndex, emitUnload);
      if(result == Result::eDelay)
        m_globalGroupsNeeded[request.decoded.globalGroup] =
            true;  // restore the pin as it wasn't unloaded and we'll try again later
    }

    if(result == Result::eSuccess || result == Result::eDelay)
    {
      m_topLevelRequests.pop_front();
      if(result == Result::eDelay)
        m_delayedRequests.push_back(request);
      else
        --m_pendingRequests;
    }
    else
    {
      assert(result == Result::eStopAndRetry);

      // Reset m_globalGroupsNeeded so the retry passes the initial filter
      m_globalGroupsNeeded[request.decoded.globalGroup] = !bool(request.decoded.load);
      break;
    }
  }
}

Result RequestDependencyPipeline::loadGroupDependenciesRecursive(
    offset_span<offset_span<uint32_t>>               meshGroupGeneratedGroups,
    uint32_t                                         meshGroupOffset,
    uint32_t                                         meshIndex,
    uint32_t                                         meshGroupIndex,
    const std::function<Result(uint32_t, uint32_t)>& emitLoad)
{
  Result result = Result::eSuccess;

  // Nothing to do if the group is already loaded
  if(m_globalGroupsLoaded[meshGroupOffset + meshGroupIndex])
    return result;

  // Load the group dependencies first to guarantee dependency order.
  for(uint32_t dependency : meshGroupGeneratedGroups[meshGroupIndex])
  {
    result = loadGroupDependenciesRecursive(meshGroupGeneratedGroups, meshGroupOffset,
                                            meshIndex, dependency, emitLoad);
    if(result != Result::eSuccess)
      return result;
  }

  // Don't emit a load if just unloaded in this batch. Top level pulse requests
  // like this are filtered out by m_globalGroupsExpected, but a dependency can
  // be unloaded and immediately reloaded by another top level request. The
  // reverse cannot happen as top level requests are pinned.
  if(m_batchUnloads.count(meshGroupOffset + meshGroupIndex) > 0)
    return Result::eStopAndRetry;

  // Emit load op for this group
  result = emitLoad(meshIndex, meshGroupIndex);
  if(result == Result::eSuccess)
    m_globalGroupsLoaded[meshGroupOffset + meshGroupIndex] = true;
  return result;
}

Result RequestDependencyPipeline::unloadGroupDependenciesRecursive(
    offset_span<offset_span<uint32_t>>               meshGroupGeneratedGroups,
    offset_span<offset_span<uint32_t>>               meshGroupGeneratingGroups,
    uint32_t                                         meshGroupOffset,
    uint32_t                                         meshIndex,
    uint32_t                                         meshGroupIndex,
    const std::function<Result(uint32_t, uint32_t)>& emitUnload)
{
  Result result = Result::eSuccess;

  // Cannot implicitly unload directly requested group
  if(m_globalGroupsNeeded[meshGroupOffset + meshGroupIndex])
    return result;

  // Abort if something depends on this group
  for(auto& groupDependentOnThis : meshGroupGeneratingGroups[meshGroupIndex])
  {
    if(m_globalGroupsLoaded[meshGroupOffset + groupDependentOnThis])
      return result;
  }

  // Emit unload op for this group if it is currently loaded
  if(m_globalGroupsLoaded[meshGroupOffset + meshGroupIndex])
    result = emitUnload(meshIndex, meshGroupIndex);

  // Search for orphaned dependencies and unload them even if this group was
  // already unloaded. This is to support Result::eStopAndRetry.
  if(result == Result::eSuccess)
  {
    m_globalGroupsLoaded[meshGroupOffset + meshGroupIndex] = false;
    m_batchUnloads.insert(meshGroupOffset + meshGroupIndex);

    // Check if any dependencies can now be unloaded. Recurse after to guarantee
    // dependency order.
    for(uint32_t dependency : meshGroupGeneratedGroups[meshGroupIndex])
    {
      result = unloadGroupDependenciesRecursive(meshGroupGeneratedGroups,
                                                meshGroupGeneratingGroups, meshGroupOffset,
                                                meshIndex, dependency, emitUnload);
      if(result != Result::eSuccess)
        return result;
    }
  }

  return result;
}

}  // namespace streaming
