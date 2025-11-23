/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Pre VK_NV_cluster_acceleration_structure BLAS and TLAS objects copied and cut
// down from https://github.com/nvpro-samples/vk_raytrace_displacement

#pragma once

#include <sample_vulkan_objects.hpp>
#include <span>
#include <vulkan/vulkan_core.h>

namespace rt {

// VkAccelerationStructureBuildSizesInfoKHR wrapper, a dependency of the main
// AccelerationStructure.
class AccelerationStructureSizes
{
public:
  template <vko::device_and_commands DeviceAndCommands>
  AccelerationStructureSizes(const DeviceAndCommands&             device,
                             VkAccelerationStructureTypeKHR       type,
                             VkBuildAccelerationStructureFlagsKHR flags,
                             std::span<const VkAccelerationStructureGeometryKHR> geometries,
                             std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos)
      : m_sizeInfo{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
                   .pNext                     = nullptr,
                   .accelerationStructureSize = 0,
                   .updateScratchSize         = 0,
                   .buildScratchSize          = 0}
  {
    assert(geometries.size() == rangeInfos.size());
    VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .pNext = nullptr,
        .type  = type,
        .flags = flags,
        .mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .srcAccelerationStructure = VK_NULL_HANDLE,
        .dstAccelerationStructure = VK_NULL_HANDLE,
        .geometryCount            = static_cast<uint32_t>(geometries.size()),
        .pGeometries              = geometries.data(),
        .ppGeometries             = nullptr,
        .scratchData              = {.deviceAddress = 0},
    };
    std::vector<uint32_t> primitiveCounts(rangeInfos.size());
    std::transform(rangeInfos.begin(), rangeInfos.end(), primitiveCounts.begin(),
                   [](const VkAccelerationStructureBuildRangeInfoKHR& rangeInfo) {
                     return rangeInfo.primitiveCount;
                   });
    device.vkGetAccelerationStructureBuildSizesKHR(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildGeometryInfo, primitiveCounts.data(), &m_sizeInfo);
  }
  const VkAccelerationStructureBuildSizesInfoKHR& operator*() const
  {
    return m_sizeInfo;
  }
  VkAccelerationStructureBuildSizesInfoKHR& operator*() { return m_sizeInfo; }
  const VkAccelerationStructureBuildSizesInfoKHR* operator->() const
  {
    return &m_sizeInfo;
  }
  VkAccelerationStructureBuildSizesInfoKHR* operator->() { return &m_sizeInfo; }

private:
  VkAccelerationStructureBuildSizesInfoKHR m_sizeInfo;
};

// VkAccelerationStructureKHR wrapper including a Buffer that holds backing
// memory for the vulkan object itself and the built acceleration structure.
// This can be a top or bottom level acceleration structure depending on the
// 'type' passed to the constructor. To use the acceleration structure it must
// first be given to BuiltAccelerationStructure.
class BuiltAccelerationStructure;
class AccelerationStructure
{
public:
  AccelerationStructure() = delete;
  template <vko::device_and_commands DeviceAndCommands, vko::allocator Allocator = vko::vma::Allocator>
  AccelerationStructure(const DeviceAndCommands&       device,
                        Allocator&                     allocator,
                        VkAccelerationStructureTypeKHR type,
                        const VkAccelerationStructureBuildSizesInfoKHR& size,
                        VkAccelerationStructureCreateFlagsKHR           flags)
      : m_type(type)
      , m_sizes(size)
      , m_buffer(device,
                 m_sizes.accelerationStructureSize,
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                     | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 allocator)
      , m_accelerationStructure(device,
                                VkAccelerationStructureCreateInfoKHR{
                                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                                    .pNext       = nullptr,
                                    .createFlags = flags,
                                    .buffer      = m_buffer,
                                    .offset      = 0,
                                    .size = m_sizes.accelerationStructureSize,
                                    .type = m_type,
                                    .deviceAddress = 0,
                                })
  {
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .pNext                 = nullptr,
        .accelerationStructure = m_accelerationStructure,
    };
    m_address = device.vkGetAccelerationStructureDeviceAddressKHR(device, &addressInfo);
  }
  const VkAccelerationStructureTypeKHR& type() const { return m_type; }
  const VkAccelerationStructureBuildSizesInfoKHR& sizes() const
  {
    return m_sizes;
  }

private:
  // Use the C++ type system to hide access to the object until it is built with
  // BuiltAccelerationStructure. This adds a little compile-time state checking.
  friend class BuiltAccelerationStructure;
  VkAccelerationStructureKHR object() const { return m_accelerationStructure; }
  const VkDeviceAddress&     address() const { return m_address; }

  VkAccelerationStructureTypeKHR           m_type;
  VkAccelerationStructureBuildSizesInfoKHR m_sizes;
  vkobj::ByteBuffer                        m_buffer;
  vkobj::AccelerationStructureKHR          m_accelerationStructure;
  VkDeviceAddress                          m_address;
};

// An AccelerationStructure that is guaranteed to have had
// vkCmdBuildAccelerationStructuresKHR called on it. An AccelerationStructure
// must be std::move()d into the constructor, along with the inputs for the
// build. An update() call is also provided, which uses
// VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR and writes the updated data
// into the same object. Note that it is up to the caller to make sure to submit
// the command buffer, filled as a side effect of the constructor, before
// submitting operations that use the built acceleration structure.
class BuiltAccelerationStructure
{
public:
  template <vko::device_and_commands DeviceAndCommands>
  BuiltAccelerationStructure(const DeviceAndCommands& device,
                             AccelerationStructure&&  accelerationStructure,
                             VkBuildAccelerationStructureFlagsKHR flags,
                             std::span<const VkAccelerationStructureGeometryKHR> geometries,
                             std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos,
                             VkDeviceAddress scratchBufferAddress,
                             VkCommandBuffer cmd)
      : m_accelerationStructure(std::move(accelerationStructure))
  {
    build(device, flags, geometries, rangeInfos, false, scratchBufferAddress, cmd);
  }

  template <vko::device_and_commands DeviceAndCommands>
  void build(const DeviceAndCommands&                            device,
             VkBuildAccelerationStructureFlagsKHR                flags,
             std::span<const VkAccelerationStructureGeometryKHR> geometries,
             std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos,
             bool            update,
             VkDeviceAddress scratchBufferAddress,
             VkCommandBuffer cmd)
  {
    assert(geometries.size() == rangeInfos.size());
    assert(!update || !!(flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR));
    VkBuildAccelerationStructureModeKHR mode =
        update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR :
                 VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .pNext = nullptr,
        .type  = m_accelerationStructure.type(),
        .flags = flags,
        .mode  = mode,
        .srcAccelerationStructure = update ? m_accelerationStructure.object() : VK_NULL_HANDLE,
        .dstAccelerationStructure = m_accelerationStructure.object(),
        .geometryCount            = static_cast<uint32_t>(geometries.size()),
        .pGeometries              = geometries.data(),
        .ppGeometries             = nullptr,
        .scratchData              = {.deviceAddress = scratchBufferAddress},
    };
    auto rangeInfosPtr = rangeInfos.data();
    device.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildGeometryInfo, &rangeInfosPtr);

    // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
    // is finished before starting the next one.
    VkMemoryBarrier barrier{
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext         = nullptr,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                0, 1, &barrier, 0, nullptr, 0, nullptr);
  }

  operator VkAccelerationStructureKHR() const
  {
    return m_accelerationStructure.object();
  }
  VkDeviceSize sizeBytes() const
  {
    return m_accelerationStructure.sizes().accelerationStructureSize;
  }
  VkDeviceAddress address() const { return m_accelerationStructure.address(); }

private:
  AccelerationStructure m_accelerationStructure;
};

}  // namespace rt
