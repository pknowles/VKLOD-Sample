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

#include <algorithm>
#include <condition_variable>
#include <debug_range_summary.hpp>
#include <iostream>
#include <lod_streaming_jobs.hpp>
#include <ostream>
#include <sample_glsl_compiler.hpp>
#include <sample_producer_consumer.hpp>
#include <sample_raytracing_objects.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <unordered_set>
#include <vulkan/vulkan_core.h>

inline std::ostream& operator<<(std::ostream& os,
                                const VkClusterAccelerationStructureBuildTriangleClusterInfoNV& x)
{
  PrefixedLines indent(os.rdbuf(), "  ");
  std::ostream  ios(&indent);
  ios << "VkClusterAccelerationStructureBuildTriangleClusterInfoNV{\n";
  ios << "clusterID " << x.clusterID << "\n";
  ios << "clusterFlags " << x.clusterFlags << "\n";
  ios << "triangleCount " << x.triangleCount << "\n";
  ios << "vertexCount " << x.vertexCount << "\n";
  ios << "positionTruncateBitCount " << x.positionTruncateBitCount << "\n";
  ios << "indexType " << x.indexType << "\n";
  ios << "opacityMicromapIndexType " << x.opacityMicromapIndexType << "\n";
  ios << "baseGeometryIndexAndGeometryFlags.geometryIndex "
      << x.baseGeometryIndexAndGeometryFlags.geometryIndex << "\n";
  ios << "baseGeometryIndexAndGeometryFlags.geometryFlags "
      << x.baseGeometryIndexAndGeometryFlags.geometryFlags << "\n";
  ios << "baseGeometryIndexAndGeometryFlags.reserved "
      << x.baseGeometryIndexAndGeometryFlags.reserved << "\n";
  ios << "indexBufferStride " << x.indexBufferStride << "\n";
  ios << "vertexBufferStride " << x.vertexBufferStride << "\n";
  ios << "geometryIndexAndFlagsBufferStride " << x.geometryIndexAndFlagsBufferStride << "\n";
  ios << "opacityMicromapIndexBufferStride " << x.opacityMicromapIndexBufferStride << "\n";
  ios << "indexBuffer " << x.indexBuffer << "\n";
  ios << "vertexBuffer " << x.vertexBuffer << "\n";
  ios << "geometryIndexAndFlagsBuffer " << x.geometryIndexAndFlagsBuffer << "\n";
  ios << "opacityMicromapArray " << x.opacityMicromapArray << "\n";
  ios << "opacityMicromapIndexBuffer " << x.opacityMicromapIndexBuffer << "\n";
  ios << "Indices ";
  if(x.indexType == VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV
     && x.indexBufferStride == sizeof(glm::u8vec3::x))
    rangeSummaryVk<glm::u8vec3>(ios, x.indexBuffer, x.triangleCount) << "\n";
  else if(x.indexType == VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_16BIT_NV
          && x.indexBufferStride == sizeof(glm::u16vec3::x))
    rangeSummaryVk<glm::u16vec3>(ios, x.indexBuffer, x.triangleCount) << "\n";
  else if(x.indexType == VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV
          && x.indexBufferStride == sizeof(glm::uvec3::x))
    rangeSummaryVk<glm::uvec3>(ios, x.indexBuffer, x.triangleCount) << "\n";
  else
    ios << "<range summary not supported>\n";
  ios << "Vertices ";
  if(x.vertexBufferStride == sizeof(glm::vec3))
    rangeSummaryVk<glm::vec3>(ios, x.vertexBuffer, x.vertexCount) << "\n";
  else
    ios << "<range summary not supported>\n";
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                const VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV& x)
{
  PrefixedLines indent(os.rdbuf(), "  ");
  std::ostream  ios(&indent);
  ios << "VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV{\n";
  ios << "clusterReferencesCount " << x.clusterReferencesCount << "\n";
  ios << "clusterReferencesStride " << x.clusterReferencesStride << "\n";
  //ios << "clusterReferences[Address] " << x.clusterReferences << "\n";
  rangeSummaryVk<uint64_t>(ios << "clusterReferences ", x.clusterReferences, x.clusterReferencesCount)
      << "\n";
  {
    vko::DeviceSpan<const VkDeviceAddress> span(
        vko::DeviceAddress<const VkDeviceAddress>(x.clusterReferences), x.clusterReferencesCount);
    auto hostArray = BufferDownloader::download(span);
    rangeSummary(ios << "clusterReferences ", hostArray, 20) << "\n";
#if !defined(NDEBUG)
    assert(std::ranges::find(hostArray, VkDeviceAddress(0)) == hostArray.end());  // no nullptrs
    assert(
        std::unordered_set<VkDeviceAddress>(hostArray.begin(), hostArray.end()).size()
        == hostArray.size());  // all unique
    for(auto& addr : hostArray)
      assert((addr & 127) == 0);  // properly aligned
#endif
  }
  os << "}";
  return os;
}

// Container for multiple bottom level acceleration structures, including a
// linearized array of input cluster adddresses and
class BlasArray
{
public:
  BlasArray(const vko::Device&   device,
            vko::vma::Allocator& allocator,
            uint32_t             blasCount,
            uint32_t             maxClustersPerMesh,
            uint32_t             maxTotalClusters);

  struct OldBuffers
  {
    std::optional<vko::DeviceBuffer<std::byte>>       m_blas;
    std::optional<vko::DeviceBuffer<std::byte>>       m_blasScratchBuffer;
    std::optional<vko::DeviceBuffer<VkDeviceAddress>> m_clasAddresses;
  };

  [[nodiscard]] OldBuffers resize(const vko::Device&   device,
                                  vko::vma::Allocator& allocator,
                                  uint32_t             maxTotalClusters);

  const vko::DeviceBuffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>& input() const
  {
    return *m_blasInfos;
  }

  const vko::DeviceBuffer<VkDeviceAddress>& inputPointers() const
  {
    return *m_clasAddresses;
  }

  // Per-mesh traversal creates a BLAS per mesh that is then shared by instances
  void cmdBuild(const vko::Device&              device,
                VkCommandBuffer                 cmd,
                vkobj::Buffer<VkDeviceAddress>& outputBlasAddresses);

  // Per-instance traversal creates a BLAS per instance that can be written directly to the TLAS input by the BLAS build
  void cmdBuild(const vko::Device& device,
                VkCommandBuffer    cmd,
                vkobj::Buffer<VkAccelerationStructureInstanceKHR>& outputTlasInfos);

  // Common build, called by the above overloads
  void cmdBuild(const vko::Device& device, VkCommandBuffer cmd, VkStridedDeviceAddressRegionKHR addresses);

  VkDeviceSize deviceMemory() const { return m_deviceMemory; }

  uint32_t maxTotalClusters() const { return m_maxTotalClusters; }
  uint32_t maxClustersPerMesh() const { return m_maxClustersPerMesh; }

private:
  // Worst case storage for per-instance BLAS cluster input
  std::optional<vko::DeviceBuffer<VkDeviceAddress>> m_clasAddresses;

  // Per-BLAS input descriptions
  std::optional<vko::DeviceBuffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>> m_blasInfos;

  // Black box acceleration structure and scratch storage for the build (use DeviceBuffer for address() method)
  std::optional<vko::DeviceBuffer<std::byte>> m_blas;
  std::optional<vko::DeviceBuffer<std::byte>> m_blasScratchBuffer;

  // alt: VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR
  VkBuildAccelerationStructureFlagsKHR m_buildFlags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

  VkDeviceSize m_deviceMemory = 0;

  uint32_t m_maxTotalClusters   = 0;
  uint32_t m_maxClustersPerMesh = 0;
};

// Top level acceleration structure. This contains instantiated BLASes.
// Depending on traversal, there may be a BLAS per instance or a BLAS per mesh
// that can be instantiated multiple times.
class Tlas
{
public:
  // alt: VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
  Tlas(const vko::Device&                                       device,
       vko::vma::Allocator&                                     allocator,
       const vkobj::Buffer<VkAccelerationStructureInstanceKHR>& tlasInfo,
       VkCommandBuffer                                          initCmd,
       VkBuildAccelerationStructureFlagsKHR buildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR);
  Tlas(const vko::Device*                   device,
       rt::BuiltAccelerationStructure       tlas,
       vko::DeviceBuffer<std::byte>         tlasScratchBuffer,
       VkBuildAccelerationStructureFlagsKHR buildFlags)
      : m_tlas(std::move(tlas))
      , m_tlasScratchBuffer(std::move(tlasScratchBuffer))
      , m_buildFlags(buildFlags)
      , m_device(device)
  {
  }

  void cmdUpdate(const vko::Device& device,
                 const vkobj::Buffer<VkAccelerationStructureInstanceKHR>& tlasInfo,
                 VkCommandBuffer cmd,
                 bool            rebuild);

  // Returns the top level acceleration structure to be used by raytracing
  // shaders
  VkAccelerationStructureKHR output() const { return m_tlas; }

  VkDeviceSize sizeBytes() const
  {
    return m_tlas.sizeBytes() + m_tlasScratchBuffer.sizeBytes();
  }

private:
  rt::BuiltAccelerationStructure m_tlas;
  vko::DeviceBuffer<std::byte> m_tlasScratchBuffer;  // DeviceBuffer for address() support
  VkBuildAccelerationStructureFlagsKHR m_buildFlags = {};
  const vko::Device* m_device = nullptr;  // Stored for build() and address() calls
};

// Holds temporary fixed sized buffers for the streaming thread to build cluster
// acceleration structures in batches, which are then copied and compacted into
// per-LOD-"group" allocations.
class ClasStaging
{
public:
  ClasStaging() = delete;
  ClasStaging(const vko::Instance&                 instance,
              const vko::Device&                   device,
              vko::vma::Allocator&                 allocator,
              VkPhysicalDevice                     physicalDevice,
              const Scene&                         scene,
              uint32_t                             maxGroupsPerBuild,
              uint32_t                             maxClustersPerBuild,
              uint32_t                             positionTruncateBits,
              VkBuildAccelerationStructureFlagsKHR buildFlags);

  // Building and compacting is split into separate calls so that we can execute
  // the build as one batch on the main render thread and vulkan queue. The
  // compaction step could be combined and happen immediately after if it
  // weren't for having a host side allocator. Moving to a GPU based memory pool
  void buildClas(vkobj::Staging&                  staging,
                 const vko::Device&               device,
                 streaming::FillClasInputProgram& fillClasInputProgram,
                 streaming::PackClasProgram&      packClasProgram,
                 shaders::StreamGroupModsList     uploadedMods,
                 std::span<const uint32_t>        loadClusterLoadGroupsHost,
                 std::span<const uint32_t>        loadGroupClusterOffsetsHost,
                 uint32_t                         totalClusters);
  void compactClas(vkobj::Staging&    staging,
                   const vko::Device& device,
                   PoolAllocator&     memoryPool,
                   vkobj::SemaphoreValue readySemaphoreState,  // signalled when buildClas(...) has finished
                   streaming::PackClasProgram&  packClasProgram,
                   shaders::StreamGroupModsList uploadedMods,
                   uint32_t                     totalClusters,
                   std::vector<PoolMemory>&     newClases);

  size_t maxClustersPerBuild() const { return m_maxClustersPerBuild; }

  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV clasProperties() const
  {
    return m_clasProperties;
  }

  size_t memoryUsage() const
  {
    // TODO: vkobj::ByteBuffer with size()?
    VkDeviceSize total = m_buildClasSizesInfo.accelerationStructureSize
                         + std::max(m_buildClasSizesInfo.buildScratchSize,
                                    m_moveClasSizesInfo.updateScratchSize)
                         + m_groupTotalClasSizes.sizeBytes();
    if(m_clasInfo)
      total += m_clasInfo->sizeBytes();
    if(m_clasAddresses)
      total += m_clasAddresses->sizeBytes();
    if(m_clasPackedAddresses)
      total += m_clasPackedAddresses->sizeBytes();
    if(m_clasSizes)
      total += m_clasSizes->sizeBytes();
    if(m_groupClasAllocNext)
      total += m_groupClasAllocNext->sizeBytes();
    if(m_groupClasBaseAddresses)
      total += m_groupClasBaseAddresses->sizeBytes();
    if(m_loadClusterLoadGroups)
      total += m_loadClusterLoadGroups->sizeBytes();
    if(m_loadGroupClusterOffsets)
      total += m_loadGroupClusterOffsets->sizeBytes();
    return total;
  }

private:
  uint32_t m_maxClustersPerBuild;

  VkBuildAccelerationStructureFlagsKHR m_buildFlags;
  VkClusterAccelerationStructureTriangleClusterInputNV m_triangleClusterInput;  // conservative maximums
  VkAccelerationStructureBuildSizesInfoKHR                 m_buildClasSizesInfo;
  VkAccelerationStructureBuildSizesInfoKHR                 m_moveClasSizesInfo;
  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV m_clasProperties;
  std::optional<vko::DeviceBuffer<VkClusterAccelerationStructureBuildTriangleClusterInfoNV>> m_clasInfo;
  std::optional<vko::DeviceBuffer<std::byte>>       m_clasData;
  std::optional<vko::DeviceBuffer<VkDeviceAddress>> m_clasAddresses;
  std::optional<vko::DeviceBuffer<VkDeviceAddress>> m_clasPackedAddresses;
  std::optional<vko::DeviceBuffer<uint32_t>>        m_clasSizes;
  std::optional<vko::DeviceBuffer<std::byte>>       m_clasScratch;
  std::optional<vko::DeviceBuffer<uint32_t>>        m_groupClasAllocNext;
  std::optional<vko::DeviceBuffer<VkDeviceAddress>> m_groupClasBaseAddresses;
  std::optional<vko::DeviceBuffer<uint32_t>>        m_loadClusterLoadGroups;
  std::optional<vko::DeviceBuffer<uint32_t>>        m_loadGroupClusterOffsets;
  vkobj::Buffer<uint32_t> m_groupTotalClasSizes;  // dedicated staging buffer
  vkobj::BufferMapping<uint32_t> m_groupTotalClasSizesMapping;  // conservatively sized
  std::span<uint32_t> m_groupTotalClasSizesHost;  // subspan of m_groupTotalClasSizesMapping
};
