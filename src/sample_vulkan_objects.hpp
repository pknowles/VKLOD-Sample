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

#include <sample_allocation.hpp>
#include <utility>
#include <vko/allocator.hpp>
#include <vko/bound_buffer.hpp>
#include <vko/handles.hpp>
#include <vko/staging_memory.hpp>
#include <vko/timeline_queue.hpp>
#include <vulkan/vulkan_core.h>

namespace vkobj {

using vko::AccelerationStructureKHR;
using vko::CommandPool;
using vko::ComputePipeline;
using vko::GraphicsPipeline;
using vko::PipelineLayout;
using vko::ShaderModule;

using vko::SemaphoreValue;
using vko::TimelineQueue;
using Semaphore = vko::TimelineSemaphore;

using ByteBuffer = vko::DeviceBuffer<std::byte>;

template <class T>
using Buffer = vko::DeviceBuffer<T>;

using Staging = vko::StagingStream<vko::vma::RecyclingStagingPool<vko::Device>>;
using DedicatedStaging = vko::StagingStream<vko::DedicatedStagingPool<vko::Device>>;

// Trivial wrapper to give a compile error when mixing up types after a
// copy/paste error. Allows the code to be more explicit about what is a host
// and device pointer with a little type safety.
template <class T>
struct DeviceAddress
{
  using value_type         = T;
  explicit DeviceAddress() = default;
  template <vko::device_and_commands DeviceAndCommands>
  DeviceAddress(const vko::BoundBuffer<T>& buffer, const DeviceAndCommands& device)
      : address(buffer.address(device))
  {
  }
  DeviceAddress(const vko::DeviceBuffer<T>& buffer)
      : address(buffer.address())
  {
  }
  explicit DeviceAddress(VkDeviceAddress raw)  // Avoid non-typed constructor if possible
      : address(raw)
  {
  }
  VkDeviceAddress address = 0xffffffffffffffffull;
  explicit        operator VkDeviceAddress() const { return address; }
  explicit        operator bool() const { return address != 0; }
  friend inline std::ostream& operator<<(std::ostream& os, const DeviceAddress& a)
  {
    return os << a.address;
  }
  int64_t operator-(const DeviceAddress& other)
  {
    // similar behaviour to pointer arithmetic where subtraction gives the
    // distance in elements
    int64_t size = address - other.address;
    assert(size % sizeof(T) == 0);
    return size / sizeof(T);
  }
};

template <class T, class U>
  requires(sizeof(T) == sizeof(U))
DeviceAddress<T> deviceReinterpretCast(DeviceAddress<U> address)
{
  return DeviceAddress<T>{VkDeviceAddress(address)};
}

template <class T>
T* translateOffset(DeviceAddress<T> offsetAddress, void* hostBase)
{
  if(!hostBase)
    return nullptr;
  return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(hostBase)
                              + offsetAddress.address);
}

template <class T>
DeviceAddress<T> translateOffset(DeviceAddress<T> offsetAddress, VkDeviceAddress deviceBase)
{
  if(!deviceBase)
    return DeviceAddress<T>{0};
  return DeviceAddress<T>{offsetAddress.address + deviceBase};
}

using ByteBufferMapping = vko::BufferMapping<std::byte, vko::vma::Allocator>;

template <class T>
using BufferMapping = vko::BufferMapping<T, vko::vma::Allocator>;

using CommandBuffer = vko::CommandBuffer;

// Single, recording, primary command buffer with no inheritance info.
class BuildingCommandBuffer
{
public:
  BuildingCommandBuffer()                                       = delete;
  BuildingCommandBuffer(BuildingCommandBuffer&& other) noexcept = default;
  BuildingCommandBuffer& operator=(BuildingCommandBuffer&& other) noexcept = default;

  template <vko::device_and_commands DeviceAndCommands>
  BuildingCommandBuffer(DeviceAndCommands&        device,
                        VkCommandPool             pool,
                        VkCommandBufferUsageFlags flags = 0)
      : m_cmd(device, nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY)
      , m_vkEndCommandBuffer(device.vkEndCommandBuffer)
  {
    VkCommandBufferBeginInfo beginInfo = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = flags,
        .pInheritanceInfo = nullptr,
    };
    vko::check(device.vkBeginCommandBuffer(m_cmd, &beginInfo));
  }
  operator VkCommandBuffer() const { return m_cmd; }
  explicit operator bool() const { return m_cmd != VK_NULL_HANDLE; }

  CommandBuffer&& endRecording()
  {
    if(m_cmd.engaged())
      vko::check(m_vkEndCommandBuffer(m_cmd));
    return std::move(m_cmd);
  }

  // Verify endRecording() was called and command buffer was moved from.
  // It's a bug if a command buffer is destroyed during recording.
  ~BuildingCommandBuffer()
  {
    // engaged() now correctly returns false when empty (moved from)
    assert(!m_cmd.engaged());
  }

private:
  CommandBuffer          m_cmd;
  PFN_vkEndCommandBuffer m_vkEndCommandBuffer = nullptr;
};

class ReadyCommandBuffer
{
public:
  ReadyCommandBuffer() = delete;
  explicit ReadyCommandBuffer(BuildingCommandBuffer&& cmd)
      : m_cmd(cmd.endRecording())
  {
  }
  template <vko::device_and_commands DeviceAndCommands>
  void submit(DeviceAndCommands& device, VkQueue queue) const
  {
    vko::submit(device, queue, {}, m_cmd, {});
  }
  template <vko::device_and_commands DeviceAndCommands>
  void submitAfter(DeviceAndCommands& device, VkQueue queue, std::span<VkSemaphoreSubmitInfo> waits) const
  {
    vko::submit(device, queue, waits, m_cmd, {});
  }
  template <vko::device_and_commands DeviceAndCommands>
  void submitAndSignal(DeviceAndCommands&               device,
                       VkQueue                          queue,
                       std::span<VkSemaphoreSubmitInfo> signals) const
  {
    vko::submit(device, queue, {}, m_cmd, signals);
  }
  explicit operator bool() const
  {
    return static_cast<VkCommandBuffer>(m_cmd) != VK_NULL_HANDLE;
  }
  VkCommandBuffer object() const { return m_cmd; }

private:
  CommandBuffer m_cmd;
};

// A command buffer that submits itself to the queue when it goes out of scope
template <class Queue>
class ImmediateCommandBuffer;

// Specialization for VkQueue
template <>
class ImmediateCommandBuffer<VkQueue>
{
public:
  template <vko::device_and_commands DeviceAndCommands>
  explicit ImmediateCommandBuffer(DeviceAndCommands& device, VkCommandPool pool, VkQueue queue)
      : m_device(&device)
      , m_cmd(device, pool, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
      , m_queue(queue)
  {
  }
  template <vko::device_and_commands DeviceAndCommands>
  explicit ImmediateCommandBuffer(DeviceAndCommands&  device,
                                  VkCommandPool       pool,
                                  vko::TimelineQueue& queue)      = delete;
  ImmediateCommandBuffer(const ImmediateCommandBuffer& other)     = delete;
  ImmediateCommandBuffer(ImmediateCommandBuffer&& other) noexcept = default;
  ImmediateCommandBuffer& operator=(const ImmediateCommandBuffer& other) = delete;
  ImmediateCommandBuffer& operator=(ImmediateCommandBuffer&& other) noexcept
  {
    submit();
    m_device = other.m_device;
    m_cmd    = std::move(other.m_cmd);
    m_queue  = other.m_queue;
    other.m_queue = VK_NULL_HANDLE;  // Prevent double-submit when other's destructor runs
    return *this;
  }
  ~ImmediateCommandBuffer() { submit(); }
  operator VkCommandBuffer() const { return m_cmd; }
  void submit()
  {
    if(m_queue)
    {
      ReadyCommandBuffer recorded(std::move(m_cmd));
      recorded.submit(*m_device, m_queue);
      vko::check(m_device->vkQueueWaitIdle(m_queue));
      m_queue = VK_NULL_HANDLE;
    }
  }

private:
  const vko::Device*    m_device = nullptr;
  BuildingCommandBuffer m_cmd;
  VkQueue               m_queue = VK_NULL_HANDLE;
};

// deduction guide
template <vko::device_and_commands DeviceAndCommands>
ImmediateCommandBuffer(DeviceAndCommands&, VkCommandPool, VkQueue)
    -> ImmediateCommandBuffer<VkQueue>;

// Specialization for TimelineQueue
// TODO: consolidate
template <>
class ImmediateCommandBuffer<TimelineQueue>
{
public:
  template <vko::device_and_commands DeviceAndCommands>
  ImmediateCommandBuffer(DeviceAndCommands&    device,
                         VkCommandPool         pool,
                         TimelineQueue&        queue,
                         VkPipelineStageFlags2 stageMask)
      : m_device(&device)
      , m_cmd(device, pool, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
      , m_queue(&queue)
      , m_submitPromise(queue.submitPromise())
      , m_stageMask(stageMask)
  {
  }
  ImmediateCommandBuffer(const ImmediateCommandBuffer& other)     = delete;
  ImmediateCommandBuffer(ImmediateCommandBuffer&& other) noexcept = default;
  ImmediateCommandBuffer& operator=(const ImmediateCommandBuffer& other) = delete;
  ImmediateCommandBuffer& operator=(ImmediateCommandBuffer&& other) noexcept
  {
    submit();
    m_device    = other.m_device;
    m_cmd       = std::move(other.m_cmd);
    m_queue     = other.m_queue;
    m_stageMask = other.m_stageMask;
    return *this;
  }
  ~ImmediateCommandBuffer() { submit(); }
  operator VkCommandBuffer() const { return m_cmd; }
  void submit()
  {
    if(m_queue)
    {
      ReadyCommandBuffer recorded(std::move(m_cmd));
      m_queue->submit(*m_device, {}, recorded.object(), std::move(m_submitPromise), m_stageMask);
      vko::check(m_device->vkQueueWaitIdle(*m_queue));
      m_queue = nullptr;
    }
  }
  vko::SemaphoreValue submitSemaphore()
  {
    return m_submitPromise.futureValue();
  }

private:
  const vko::Device*    m_device = nullptr;
  BuildingCommandBuffer m_cmd;
  vko::TimelineQueue*   m_queue = nullptr;
  vko::SubmitPromise    m_submitPromise;
  VkPipelineStageFlags2 m_stageMask = 0;
};

// deduction guide
template <vko::device_and_commands DeviceAndCommands>
ImmediateCommandBuffer(DeviceAndCommands&, VkCommandPool, TimelineQueue&, VkPipelineStageFlags2)
    -> ImmediateCommandBuffer<TimelineQueue>;

// Vulkan objects common to a thread doing vulkan work. This includes a
// non-thread safe StagingMemoryManager in the allocator.
struct Context
{
  Context() = delete;
  Context(const vko::Instance& instance_,
          const vko::Device&   device_,
          VkPhysicalDevice     physicalDevice_,
          vko::vma::Allocator& memAllocator,
          vko::TimelineQueue&  queue_)
      : instance(instance_)
      , device(device_)
      , physicalDevice(physicalDevice_)
      , commandPool(device_,
                    VkCommandPoolCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                        .queueFamilyIndex = queue_.familyIndex(),
                    })
      , queue(queue_)
      , queueFamilyIndex(queue_.familyIndex())  // TODO: remove
      , allocator(memAllocator)
      , staging(queue.get(),
                vko::vma::RecyclingStagingPool<vko::Device>(device_,
                                                            memAllocator,
                                                            /*minPools=*/2,
                                                            /*maxPools=*/8,
                                                            /*poolSize=*/16 << 20))  // 16MB pools
  {
  }

  std::reference_wrapper<const vko::Instance> instance;
  std::reference_wrapper<const vko::Device>   device;
  VkPhysicalDevice                            physicalDevice;
  vkobj::CommandPool commandPool;  // reusable single-shot command buffers per thread
  std::reference_wrapper<vko::TimelineQueue> queue;  // owned by this thread (note, there are better designs)
  uint32_t                                    queueFamilyIndex;
  std::reference_wrapper<vko::vma::Allocator> allocator;
  vkobj::Staging                              staging;
};

// Scoped push/pop wrapper for NVTX range markers
class NvtxRange
{
public:
  [[nodiscard]] explicit NvtxRange([[maybe_unused]] const char* name)
  {
#ifdef NVTX3
    nvtxRangePush(name);
#else
    (void)name;
#endif
  }

  [[nodiscard]] explicit NvtxRange([[maybe_unused]] const char* name,
                                   [[maybe_unused]] uint32_t    color)
  {
#ifdef NVTX3
    nvtxRangePushEx(nvtxRangeStartEx(
        &(nvtxEventAttributes_t){.version       = NVTX_VERSION,
                                 .size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE,
                                 .colorType     = NVTX_COLOR_ARGB,
                                 .color         = color,
                                 .messageType   = NVTX_MESSAGE_TYPE_ASCII,
                                 .message.ascii = name}));
#else
    (void)name;
    (void)color;
#endif
  }

  ~NvtxRange()
  {
#ifdef NVTX3
    nvtxRangePop();
#endif
  }

  NvtxRange(const NvtxRange&)            = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

// Scoped wrapper for Vulkan debug utils labels
template <class CommandBuffer = vko::CyclingCommandBuffer<>>
class ScopedDebugLabel
{
public:
  template <vko::device_and_commands DeviceAndCommands>
  [[nodiscard]] explicit ScopedDebugLabel(DeviceAndCommands& device,
                                          CommandBuffer&     cmd,
                                          const std::string& label,
                                          std::array<float, 4> color = {{1.0f, 1.0f, 1.0f, 1.0f}})
      : m_cmd(cmd)
      , m_vkCmdBeginDebugUtilsLabelEXT(device.vkCmdBeginDebugUtilsLabelEXT)
      , m_vkCmdEndDebugUtilsLabelEXT(device.vkCmdEndDebugUtilsLabelEXT)
      , m_nvtxRange(label.c_str())  // mark the CPU work too
  {
    VkDebugUtilsLabelEXT labelInfo{
        .sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pNext      = nullptr,
        .pLabelName = label.c_str(),
        .color      = {color[0], color[1], color[2], color[3]},
    };
    m_vkCmdBeginDebugUtilsLabelEXT(cmd, &labelInfo);
  }
  ~ScopedDebugLabel()
  {
    if(m_cmd && m_vkCmdEndDebugUtilsLabelEXT)
    {
      m_vkCmdEndDebugUtilsLabelEXT(m_cmd);
    }
  }
  ScopedDebugLabel(const ScopedDebugLabel&)            = delete;
  ScopedDebugLabel& operator=(const ScopedDebugLabel&) = delete;

private:
  CommandBuffer&                   m_cmd;
  PFN_vkCmdBeginDebugUtilsLabelEXT m_vkCmdBeginDebugUtilsLabelEXT = nullptr;
  PFN_vkCmdEndDebugUtilsLabelEXT   m_vkCmdEndDebugUtilsLabelEXT   = nullptr;
  NvtxRange                        m_nvtxRange;
};

}  // namespace vkobj

template <vko::device_and_commands DeviceAndCommands>
inline void memoryBarrier(DeviceAndCommands&   device,
                          VkCommandBuffer      cmd,
                          VkAccessFlags        srcAccess,
                          VkAccessFlags        dstAccess,
                          VkPipelineStageFlags srcStage,
                          VkPipelineStageFlags dstStage)
{
  VkMemoryBarrier memoryBarrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                   .pNext = nullptr,
                                   .srcAccessMask = srcAccess,
                                   .dstAccessMask = dstAccess};
  device.vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &memoryBarrier, 0,
                              nullptr, 0, nullptr);
}
