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

#include "vko/device_address.hpp"
#include <acceleration_structures.hpp>
#include <nvh/nsightevents.h>
#include <sample_allocation.hpp>
#include <sample_vulkan_objects.hpp>
#include <stdexcept>
#include <vulkan/vulkan_core.h>

BlasArray::BlasArray(const vko::Device&   device,
                     vko::vma::Allocator& allocator,
                     uint32_t             blasCount,
                     uint32_t             maxClustersPerMesh,
                     uint32_t             maxTotalClusters)
    : m_maxClustersPerMesh(maxClustersPerMesh)
{
  // Allocate BLAS input buffer, a per-BLAS list of CLAS addresses
  m_blasInfos.emplace(device, blasCount,
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_deviceMemory += m_blasInfos->sizeBytes();

  // Call common init and resize code. There will be no garbage on the first
  // call.
  std::ignore = resize(device, allocator, maxTotalClusters);
}

BlasArray::OldBuffers BlasArray::resize(const vko::Device&   device,
                                        vko::vma::Allocator& allocator,
                                        uint32_t             maxTotalClusters)
{
  printf("Reallocating BLAS: %u\n", maxTotalClusters);
  assert(maxTotalClusters > 0);
  m_maxTotalClusters = maxTotalClusters;

  // First, query the driver for the worst case memory usage given the new
  // cluster count
  // DANGER: keep in sync with SceneBlas::cmdBuild()
  VkClusterAccelerationStructureClustersBottomLevelInputNV custerBlasInput{
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV,
      .pNext                = nullptr,
      .maxTotalClusterCount = m_maxTotalClusters,
      .maxClusterCountPerAccelerationStructure =
          std::min(m_maxTotalClusters, m_maxClustersPerMesh),  // Far bigger than we should ever generate
  };
  VkClusterAccelerationStructureInputInfoNV inputs = {
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
      .pNext = nullptr,
      .maxAccelerationStructureCount = uint32_t(m_blasInfos->size()),
      .flags                         = m_buildFlags,
      .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV,
      .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
      .opInput =
          VkClusterAccelerationStructureOpInputNV{
              .pClustersBottomLevel = &custerBlasInput,
          },
  };
  VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
      .pNext = nullptr,
      .accelerationStructureSize = 0,
      .updateScratchSize         = 0,
      .buildScratchSize          = 0,
  };
  device.vkGetClusterAccelerationStructureBuildSizesNV(device, &inputs, &sizesInfo);

  // Allocate a buffer for the input CLAS addresses, linearized for all BLASes
  OldBuffers result;
  if(m_clasAddresses)
    m_deviceMemory -= m_clasAddresses->sizeBytes();
  result.m_clasAddresses = std::move(m_clasAddresses);
  m_clasAddresses.emplace(device, m_maxTotalClusters,
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                              | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_deviceMemory += m_clasAddresses->sizeBytes();

  // Allocate space for all output BLASes, i.e. the acceleration structure data
  if(m_blas)
    m_deviceMemory -= m_blas->sizeBytes();
  result.m_blas = std::move(m_blas);
  m_blas.emplace(device, sizesInfo.accelerationStructureSize,
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_deviceMemory += m_blas->sizeBytes();
  assert(m_blas->address() % 128 == 0);  // TODO: verify alignment from VkPhysicalDeviceAccelerationStructurePropertiesKHR?

  // Scratch space to build BLASes with a single indirect call
  if(m_blasScratchBuffer)
    m_deviceMemory -= m_blasScratchBuffer->sizeBytes();
  result.m_blasScratchBuffer = std::move(m_blasScratchBuffer);
  m_blasScratchBuffer.emplace(device, sizesInfo.buildScratchSize,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                  | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_deviceMemory += m_blasScratchBuffer->sizeBytes();
  return result;
}

void BlasArray::cmdBuild(const vko::Device&              device,
                         VkCommandBuffer                 cmd,
                         vkobj::Buffer<VkDeviceAddress>& outputBlasAddresses)
{
  cmdBuild(device, cmd, vko::DeviceSpan(outputBlasAddresses).regionKhr());
}

void BlasArray::cmdBuild(const vko::Device& device,
                         VkCommandBuffer    cmd,
                         vkobj::Buffer<VkAccelerationStructureInstanceKHR>& outputTlasInfos)
{
  VkDeviceAddress addr = outputTlasInfos.address();
  cmdBuild(device, cmd,
           VkStridedDeviceAddressRegionKHR{
               .deviceAddress = addr + offsetof(VkAccelerationStructureInstanceKHR, accelerationStructureReference),
               .stride = VkDeviceSize(sizeof(VkAccelerationStructureInstanceKHR)),
               .size = outputTlasInfos.size()
                       * VkDeviceSize(sizeof(VkAccelerationStructureInstanceKHR)),
           });
}

void BlasArray::cmdBuild(const vko::Device& device, VkCommandBuffer cmd, VkStridedDeviceAddressRegionKHR addresses)
{
  // DANGER: keep in sync with SceneBlas::resize()
  assert(m_clasAddresses->size() == size_t(m_maxTotalClusters));
  VkClusterAccelerationStructureClustersBottomLevelInputNV custerBlasInput{
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV,
      .pNext                = nullptr,
      .maxTotalClusterCount = m_maxTotalClusters,
      .maxClusterCountPerAccelerationStructure =
          std::min(m_maxTotalClusters, m_maxClustersPerMesh),  // Far bigger than we should ever generate
  };
  VkClusterAccelerationStructureInputInfoNV input = {
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
      .pNext = nullptr,
      .maxAccelerationStructureCount = uint32_t(m_blasInfos->size()),
      .flags                         = m_buildFlags,
      .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV,
      .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
      .opInput =
          VkClusterAccelerationStructureOpInputNV{
              .pClustersBottomLevel = &custerBlasInput,
          },
  };
  assert(m_blasInfos->size() == addresses.size / addresses.stride
         && addresses.size % addresses.stride == 0);
  VkClusterAccelerationStructureCommandsInfoNV blasCommandsInfo = {
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
      .pNext = nullptr,
      .input = input,
      .dstImplicitData = m_blas->address(),  // implicit meaning the driver will populate dstAddressesArray for us
      .scratchData       = m_blasScratchBuffer->address(),
      .dstAddressesArray = addresses,
      .dstSizesArray =
          VkStridedDeviceAddressRegionKHR{
              .deviceAddress = 0,
              .stride        = sizeof(uint32_t),
              .size          = 0,
          },
      .srcInfosArray = vko::DeviceSpan(*m_blasInfos).regionKhr(),
      .srcInfosCount = 0 /* optional device/dynamic size, but we want everything */,
      .addressResolutionFlags = 0,
  };
  device.vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &blasCommandsInfo);

  // memoryBarrier helper function - need to find its signature
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                          VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR};
  device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);
}

struct TlasInput
{
  TlasInput(const vkobj::Buffer<VkAccelerationStructureInstanceKHR>& tlasInfo)
     : geometry{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .pNext        = nullptr,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .geometry =
            VkAccelerationStructureGeometryDataKHR{
                .instances =
                    {
                        .sType           = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                        .pNext           = nullptr,
                        .arrayOfPointers = VK_FALSE,
                        .data =
                            VkDeviceOrHostAddressConstKHR{
                                .deviceAddress = tlasInfo.address(),
                            },
                    },
            },
        .flags = 0,
    },
    rangeInfo{
        .primitiveCount  = uint32_t(tlasInfo.size()),
        .primitiveOffset = 0,
        .firstVertex     = 0,
        .transformOffset = 0,
    }
  {
  }
  VkAccelerationStructureGeometryKHR       geometry;
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo;
};

inline Tlas makeTlas(const vko::Device&   device,
                     vko::vma::Allocator& allocator,
                     const vkobj::Buffer<VkAccelerationStructureInstanceKHR>& tlasInfo,
                     VkCommandBuffer                      initCmd,
                     VkBuildAccelerationStructureFlagsKHR buildFlags)
{
  // We want to record a command buffer that updates the acceleration structure,
  // so we do a regular build first, i.e.
  // VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR now and
  // VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR in cmdUpdate()
  VkAccelerationStructureCreateFlagsKHR createFlags = 0;
  TlasInput                             tlasInput(tlasInfo);
  VkBuildAccelerationStructureFlagsKHR  fullBuildFlags =
      buildFlags | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
      | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR;
  rt::AccelerationStructureSizes sizes(
      device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, fullBuildFlags,
      std::span(&tlasInput.geometry, 1), std::span(&tlasInput.rangeInfo, 1));
  vko::DeviceBuffer<std::byte> scratch(
      device, std::max(sizes->buildScratchSize, sizes->updateScratchSize),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
          | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);

  rt::BuiltAccelerationStructure builtTlas(
      device,
      rt::AccelerationStructure(device, allocator, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                *sizes, createFlags),
      fullBuildFlags, std::span(&tlasInput.geometry, 1),
      std::span(&tlasInput.rangeInfo, 1), scratch.address(), initCmd);

  // Barrier for immediate update afterwards
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR};
  device.vkCmdPipelineBarrier(initCmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);

  return Tlas(&device, std::move(builtTlas), std::move(scratch), buildFlags);
}

Tlas::Tlas(const vko::Device&                                       device,
           vko::vma::Allocator&                                     allocator,
           const vkobj::Buffer<VkAccelerationStructureInstanceKHR>& tlasInfo,
           VkCommandBuffer                                          initCmd,
           VkBuildAccelerationStructureFlagsKHR                     buildFlags)
    : Tlas(makeTlas(device, allocator, tlasInfo, initCmd, buildFlags))
{
  m_device = &device;  // Ensure device is set even when using delegating constructor
}

void Tlas::cmdUpdate(const vko::Device& device,
                     const vkobj::Buffer<VkAccelerationStructureInstanceKHR>& tlasInfo,
                     VkCommandBuffer cmd,
                     bool            rebuild)
{
  VkAccelerationStructureGeometryKHR instancesGeometry = {
      .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .pNext        = nullptr,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry =
          VkAccelerationStructureGeometryDataKHR{
              .instances =
                  {
                      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                      .pNext           = nullptr,
                      .arrayOfPointers = VK_FALSE,
                      .data =
                          VkDeviceOrHostAddressConstKHR{
                              .deviceAddress = tlasInfo.address(),
                          },
                  },
          },
      .flags = 0,
  };
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo{
      .primitiveCount  = uint32_t(tlasInfo.size()),
      .primitiveOffset = 0,
      .firstVertex     = 0,
      .transformOffset = 0,
  };
  VkBuildAccelerationStructureFlagsKHR buildFlags =
      m_buildFlags | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
      | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR;

  // Record a build or update into cmd
  m_tlas.build(*m_device, buildFlags, std::span(&instancesGeometry, 1),
               std::span(&rangeInfo, 1), !rebuild /* update */,
               m_tlasScratchBuffer.address(), cmd);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR};
  device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0,
                              1, &barrier, 0, nullptr, 0, nullptr);
}

ClasStaging::ClasStaging(const vko::Instance& instance,
                         const vko::Device&   device,
                         vko::vma::Allocator& allocator,
                         VkPhysicalDevice     physicalDevice,
                         const Scene&         scene,
                         uint32_t             maxGroupsPerBuild,
                         uint32_t             maxClustersPerBuild,
                         uint32_t             positionTruncateBits,
                         VkBuildAccelerationStructureFlagsKHR buildFlags)
    : m_maxClustersPerBuild(maxClustersPerBuild)
    , m_buildFlags(buildFlags)
    , m_groupTotalClasSizes(device,
                            maxClustersPerBuild,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                            allocator)
    , m_groupTotalClasSizesMapping(m_groupTotalClasSizes.map())
{
  // Query alignment requirements
  m_clasProperties = VkPhysicalDeviceClusterAccelerationStructurePropertiesNV{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV,
      .pNext                              = nullptr,
      .maxVerticesPerCluster              = 0xffffffffu,
      .maxTrianglesPerCluster             = 0xffffffffu,
      .clusterScratchByteAlignment        = 0xffffffffu,
      .clusterByteAlignment               = 0xffffffffu,
      .clusterTemplateByteAlignment       = 0xffffffffu,
      .clusterBottomLevelByteAlignment    = 0xffffffffu,
      .clusterTemplateBoundsByteAlignment = 0xffffffffu,
      .maxClusterGeometryIndex            = 0xffffffffu,
  };
  VkPhysicalDeviceProperties2 props2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                        .pNext      = &m_clasProperties,
                                        .properties = {}};
  instance.vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

  // Compute conservative CLAS sizes
  assert(scene.counts.maxClusterTriangleCount <= m_clasProperties.maxTrianglesPerCluster);
  assert(scene.counts.maxClusterVertexCount <= m_clasProperties.maxVerticesPerCluster);
  m_triangleClusterInput = VkClusterAccelerationStructureTriangleClusterInputNV{
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV,
      .pNext                         = nullptr,
      .vertexFormat                  = VK_FORMAT_R32G32B32_SFLOAT,
      .maxGeometryIndexValue         = 0,
      .maxClusterUniqueGeometryCount = 0,
      .maxClusterTriangleCount       = scene.counts.maxClusterTriangleCount,
      .maxClusterVertexCount         = scene.counts.maxClusterVertexCount,
      .maxTotalTriangleCount = maxClustersPerBuild * scene.counts.maxClusterTriangleCount,
      .maxTotalVertexCount = maxClustersPerBuild * scene.counts.maxClusterVertexCount,
      .minPositionTruncateBitCount = positionTruncateBits,
  };
  VkClusterAccelerationStructureInputInfoNV buildClasInputInfo = {
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
      .pNext = nullptr,
      .maxAccelerationStructureCount = maxClustersPerBuild,
      .flags                         = m_buildFlags,
      .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
      .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
      .opInput = VkClusterAccelerationStructureOpInputNV{.pTriangleClusters = &m_triangleClusterInput},
  };
  m_buildClasSizesInfo = VkAccelerationStructureBuildSizesInfoKHR{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
      .pNext = nullptr,
      .accelerationStructureSize = 0xffffffffffffffffull /* output */,
      .updateScratchSize         = 0xffffffffffffffffull /* output */,
      .buildScratchSize          = 0xffffffffffffffffull /* output */,
  };
  device.vkGetClusterAccelerationStructureBuildSizesNV(device, &buildClasInputInfo,
                                                       &m_buildClasSizesInfo);

  // Compute scratch memory for a move and compaction operation to per-group
  // CLAS buffers. CLASes are built to a conservatively sized output. To
  // conserve memory, the acceleration structures are moved and compacted
  // after the build.
  VkClusterAccelerationStructureMoveObjectsInputNV moveInfo = {
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV,
      .pNext = nullptr,
      .type  = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV,
      .noMoveOverlap = VK_TRUE,  // not in-place, copying to separate buffers
      .maxMovedBytes = uint32_t(m_buildClasSizesInfo.accelerationStructureSize),
  };
  VkClusterAccelerationStructureInputInfoNV moveObjectsInputInfo = {
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
      .pNext = nullptr,
      .maxAccelerationStructureCount = maxClustersPerBuild,
      .flags                         = m_buildFlags,
      .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV,
      .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV,
      .opInput = VkClusterAccelerationStructureOpInputNV{.pMoveObjects = &moveInfo},
  };
  m_moveClasSizesInfo = VkAccelerationStructureBuildSizesInfoKHR{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
      .pNext = nullptr,
      .accelerationStructureSize = 0xffffffffffffffffull /* output */,
      .updateScratchSize         = 0xffffffffffffffffull /* output */,
      .buildScratchSize          = 0xffffffffffffffffull /* output */,
  };
  device.vkGetClusterAccelerationStructureBuildSizesNV(device, &moveObjectsInputInfo,
                                                       &m_moveClasSizesInfo);

  // This can happen if a validation layers error callback chooses to skip the call
  if(m_buildClasSizesInfo.accelerationStructureSize == 0xffffffffffffffffull)
    throw std::runtime_error("vkGetClusterAccelerationStructureBuildSizesNV() failed");

  // Temporary buffers to build just one batch of acceleration structures
  VkDeviceSize scratchSize = std::max(m_buildClasSizesInfo.buildScratchSize,
                                      m_moveClasSizesInfo.updateScratchSize);  // TODO: buildScratchSize or buildScratchSize to move?
  m_clasData.emplace(device, m_buildClasSizesInfo.accelerationStructureSize,
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                         | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_clasAddresses.emplace(device, maxClustersPerBuild,
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                              | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_clasPackedAddresses.emplace(device, maxClustersPerBuild,
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                    | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_groupClasAllocNext.emplace(device, maxGroupsPerBuild,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                   | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_groupClasBaseAddresses.emplace(
      device, maxGroupsPerBuild,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
          | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_clasInfo.emplace(device, maxClustersPerBuild,
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                         | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_clasSizes.emplace(device, maxClustersPerBuild,
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_clasScratch.emplace(device, scratchSize,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,  // TODO: align to VkPhysicalDeviceClusterAccelerationStructurePropertiesNV::clusterScratchByteAlignment
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_loadClusterLoadGroups.emplace(device, maxClustersPerBuild,
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                      | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_loadGroupClusterOffsets.emplace(device, maxGroupsPerBuild,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                        | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
}

void ClasStaging::buildClas(vkobj::Staging&    staging,
                            const vko::Device& device,
                            streaming::FillClasInputProgram& fillClasInputProgram,
                            streaming::PackClasProgram&  packClasProgram,
                            shaders::StreamGroupModsList uploadedMods,
                            std::span<const uint32_t> loadClusterLoadGroupsHost,
                            std::span<const uint32_t> loadGroupClusterOffsetsHost,
                            uint32_t totalClusters)
{
  vkobj::NvtxRange buildClasRange("ClasStaging::buildClas");
  assert(loadClusterLoadGroupsHost.size() <= maxClustersPerBuild());
  assert(m_groupTotalClasSizesHost.size() == 0);
  uint32_t newGroupCount = uploadedMods.loadGroupCount;

  auto& cmd = staging.commandBuffer();

  // Barrier from uploading data in ClusterGroupGeometryVk() and
  // GroupModsList::write() to its direct use in the acceleration data
  // structure build. Added for completeness; probably reduncant with
  // subsequent barriers.
  memoryBarrier(device, cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

  // Temporary data for per-cluster threads to look up their group and cluster
  // index within the group
  vko::upload(staging, device, loadClusterLoadGroupsHost, *m_loadClusterLoadGroups);
  vko::upload(staging, device, loadGroupClusterOffsetsHost, *m_loadGroupClusterOffsets);
  memoryBarrier(device, cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Have the GPU populate
  // VkClusterAccelerationStructureBuildTriangleClusterInfoNV CLAS
  // input structures since addresses are already resident
  shaders::FillClasInputConstants fillClasInputConstants{
      .loadGroupsAddress = uploadedMods.loadGroupsAddress,
      .clasInfoAddress = vkobj::deviceReinterpretCast<shaders::ClusterCLASInfoNV>(
          vkobj::DeviceAddress<VkClusterAccelerationStructureBuildTriangleClusterInfoNV>(*m_clasInfo)),
      .loadClusterLoadGroupsAddress   = *m_loadClusterLoadGroups,
      .loadGroupClusterOffsetsAddress = *m_loadGroupClusterOffsets,
      .clusterCount                   = totalClusters,
      .positionTruncateBits = m_triangleClusterInput.minPositionTruncateBitCount,
  };
  device.vkCmdPushConstants(cmd, fillClasInputProgram.pipelineLayout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            sizeof(fillClasInputConstants), &fillClasInputConstants);
  device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           fillClasInputProgram.pipeline);
  device.vkCmdDispatch(cmd, div_ceil(totalClusters, uint32_t(STREAM_WORKGROUP_SIZE)), 1, 1);
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

  // Build all cluster acceleration structures in one indirect call
  VkClusterAccelerationStructureInputInfoNV buildClasInputInfo = {
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
      .pNext = nullptr,
      .maxAccelerationStructureCount = totalClusters,  // NOTE: set the exact CLAS count (sizes were based on the max.)
      .flags = m_buildFlags,
      .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
      .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
      .opInput = VkClusterAccelerationStructureOpInputNV{.pTriangleClusters = &m_triangleClusterInput},
  };
  VkClusterAccelerationStructureCommandsInfoNV buildClasCmdInfo = {
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
      .pNext = nullptr,
      .input = buildClasInputInfo,
      .dstImplicitData = m_clasData->address(),  // implicit meaning the driver will populate dstAddressesArray for us
      .scratchData       = m_clasScratch->address(),
      .dstAddressesArray = vko::DeviceSpan(*m_clasAddresses).regionKhr(),
      .dstSizesArray     = vko::DeviceSpan(*m_clasSizes).regionKhr(),
      .srcInfosArray     = vko::DeviceSpan(*m_clasInfo).regionKhr(),
      .srcInfosCount = 0 /* optional device/dynamic size, but we want everything */,
      .addressResolutionFlags = 0,
  };
  device.vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &buildClasCmdInfo);
  memoryBarrier(device, cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

  // Compute compacted per-group destination address offsets. Clear
  // the buffer first so all offsets start at zero.
  device.vkCmdFillBuffer(cmd, *m_groupClasAllocNext, 0, sizeof(uint32_t) * newGroupCount, 0);
  memoryBarrier(device, cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  shaders::PackClasConstants packClasConstants{
      .loadGroupsAddress              = uploadedMods.loadGroupsAddress,
      .loadClusterLoadGroupsAddress   = *m_loadClusterLoadGroups,
      .loadGroupClusterOffsetsAddress = *m_loadGroupClusterOffsets,
      .clasSizesAddress               = *m_clasSizes,
      .groupClasAllocNextAddress      = *m_groupClasAllocNext,  // output
      .packedClasAddressesAddress     = vkobj::DeviceAddress<uint64_t>(0llu),
      .groupClasBaseAddressesAddress  = vkobj::DeviceAddress<uint64_t>(0llu),
      .clusterCount                   = totalClusters,
  };
  device.vkCmdPushConstants(cmd, packClasProgram.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(packClasConstants), &packClasConstants);
  device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, packClasProgram.pipeline);
  device.vkCmdDispatch(cmd, div_ceil(totalClusters, uint32_t(STREAM_WORKGROUP_SIZE)), 1, 1);
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

  // Download the packed sizes. This won't happen immediately. The pointer
  // becomes valid after the command buffer has been executed. Note that we
  // use a dedicated staging buffer, not the allocator's staging buffers,
  // because the buffer and its mapped pointer needs to remain valid until the
  // call to compactClas().
  {
    VkBufferCopy bufferCopy{.srcOffset = 0, .dstOffset = 0, .size = sizeof(uint32_t) * newGroupCount};
    device.vkCmdCopyBuffer(cmd, *m_groupClasAllocNext, m_groupTotalClasSizes, 1, &bufferCopy);
    m_groupTotalClasSizesHost =
        m_groupTotalClasSizesMapping.span().subspan(0, newGroupCount);
  }
}

void ClasStaging::compactClas(vkobj::Staging&              staging,
                              const vko::Device&           device,
                              PoolAllocator&               memoryPool,
                              vkobj::SemaphoreValue        readySemaphoreState,
                              streaming::PackClasProgram&  packClasProgram,
                              shaders::StreamGroupModsList uploadedMods,
                              uint32_t                     totalClusters,
                              std::vector<PoolMemory>&     newClases)
{
  vkobj::NvtxRange compactClasRange("ClasStaging::compactClas");
  // Make sure buildClas() has completed before reading
  // clasSizes.groupTotalClasSizesHost
  readySemaphoreState.wait(device);

  size_t newGroupCount = m_groupTotalClasSizesHost.size();
  assert(newGroupCount != 0);

  // Use staging's command buffer for all work
  auto& cmd = staging.commandBuffer();

  // Allocate per-group CLAS data and upload the allocated address
  assert(newClases.empty());
  vko::upload(staging, device,
              vko::BufferSpan(*m_groupClasBaseAddresses).subspan(0, newGroupCount),
              [&](VkDeviceSize elemOffset, std::span<VkDeviceAddress> groupClasBaseAddressesHost) {
                auto sizesSubspan =
                    std::span(m_groupTotalClasSizesHost)
                        .subspan(elemOffset, groupClasBaseAddressesHost.size());
                for(size_t i = 0; i < groupClasBaseAddressesHost.size(); ++i)
                {
                  newClases.emplace_back(PoolMemory(memoryPool, sizesSubspan[i],
                                                    m_clasProperties.clusterByteAlignment));
                  groupClasBaseAddressesHost[i] = newClases.back();
                }
              });
  memoryBarrier(device, cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Add the allocated base addresses to the offsets, writing both:
  // - packedClasAddresses for vulkan to move the CLASes
  // - shaders::ClusterGroup::clasAddressesAddress array for
  //   rendering
  // The groupClasAddressOffsets is cleared again to re-compute
  // offsets from the base address. An alternative would be to write
  // the offsets earlier and append the base address at the cost of
  // an additional fetch.
  device.vkCmdFillBuffer(cmd, *m_groupClasAllocNext, 0, sizeof(uint32_t) * newGroupCount, 0);
  memoryBarrier(device, cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  shaders::PackClasConstants packClasConstants{
      .loadGroupsAddress              = uploadedMods.loadGroupsAddress,
      .loadClusterLoadGroupsAddress   = *m_loadClusterLoadGroups,
      .loadGroupClusterOffsetsAddress = *m_loadGroupClusterOffsets,
      .clasSizesAddress               = *m_clasSizes,
      .groupClasAllocNextAddress      = *m_groupClasAllocNext,
      .packedClasAddressesAddress = *m_clasPackedAddresses,  // one of two outputs
      .groupClasBaseAddressesAddress = *m_groupClasBaseAddresses,  // new input
      .clusterCount                  = totalClusters,
  };
  device.vkCmdPushConstants(cmd, packClasProgram.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(packClasConstants), &packClasConstants);
  device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, packClasProgram.pipeline);
  device.vkCmdDispatch(cmd, div_ceil(totalClusters, uint32_t(STREAM_WORKGROUP_SIZE)), 1, 1);
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

  // Copy and tightly pack CLASes from the temporary array
  // to per-group allocations
  static_assert(sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV)
                == sizeof(VkDeviceSize));
  VkClusterAccelerationStructureMoveObjectsInputNV moveInfo = {
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV,
      .pNext = nullptr,
      .type  = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV,
      .noMoveOverlap = true,  // not in-place, copying to separate buffers
      .maxMovedBytes = uint32_t(m_buildClasSizesInfo.accelerationStructureSize),
  };
  VkClusterAccelerationStructureInputInfoNV moveObjectsInputInfo = {
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
      .pNext = nullptr,
      .maxAccelerationStructureCount = totalClusters,  // NOTE: set the exact CLAS count (sizes were based on the max.)
      .flags  = m_buildFlags,
      .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV,
      .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV,
      .opInput = VkClusterAccelerationStructureOpInputNV{.pMoveObjects = &moveInfo},
  };
  VkClusterAccelerationStructureCommandsInfoNV moveClasCmdInfo = {
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
      .pNext           = nullptr,
      .input           = moveObjectsInputInfo,
      .dstImplicitData = 0,  // dstAddressesArray is already populated
      .scratchData     = m_clasScratch->address(),
      .dstAddressesArray = vko::DeviceSpan(*m_clasPackedAddresses).regionKhr(),
      .dstSizesArray =
          VkStridedDeviceAddressRegionKHR{
              .deviceAddress = 0,
              .stride        = sizeof(uint32_t),
              .size          = 0,
          },
      .srcInfosArray = vko::DeviceSpan(*m_clasAddresses).regionKhr(),
      .srcInfosCount = 0 /* optional indirect size, but we already have totalClusters host side */,
      .addressResolutionFlags = 0,
  };
  device.vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &moveClasCmdInfo);
  memoryBarrier(device, cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);

  m_groupTotalClasSizesHost = {};  // Marker for out of order buildClas() + compactClas()
}
