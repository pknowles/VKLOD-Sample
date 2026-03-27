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

#include <debug_range_summary.hpp>
#include <decodeless/offset_span.hpp>
#include <decodeless/writer.hpp>
#include <filesystem>
#include <generate_mesh.hpp>
#include <memory.h>
#include <memory_resource>
#include <mesh_util.hpp>
#include <nvcluster/nvcluster.h>
#include <nvclusterlod/nvclusterlod_hierarchy.h>
#include <nvclusterlod/nvclusterlod_hierarchy_storage.hpp>

#include <nvclusterlod/nvclusterlod_mesh.h>
#include <nvclusterlod/nvclusterlod_mesh_storage.hpp>
#include <ranges>
#include <sample_allocation.hpp>
#include <sample_image.hpp>
#include <sample_vulkan_objects.hpp>
#include <shaders_scene.h>
#include <stdlib.h>
#include <traverse_device_host.h>
#include <vulkan/vulkan_core.h>

// The render cache is a memory mapped file of raw structs. This version is used
// to invalidate it after making a binary-incompatible change to the relevant
// structs below
#define SCENE_RENDERCACHE_VERSION 8

namespace fs = std::filesystem;

using file_writer = decodeless::file_writer;

template <class T>
using offset_span = decodeless::offset_span<T>;

using Instance = shaders::Instance;

// Arguments for generating LODs
struct SceneLodConfig
{
  uint32_t clusterSize              = 64;
  uint32_t clusterGroupSize         = 32;
  float    lodLevelDecimationFactor = 0.5f;
};

template <std::ranges::contiguous_range T>
std::span<std::ranges::range_value_t<T>> subspan(T&& span, nvcluster_Range range)
{
  return std::span(span).subspan(range.offset, range.count);
}

inline auto indices(nvcluster_Range range)
{
  return std::views::iota(range.offset, range.offset + range.count);
}

// Temporary whole-mesh structure before it's split into clusters
struct Mesh;

// Spatially local groups of geometry and the output from LOD/clustering. Most
// notable is the uint8_t indices, which can only reference a small number of
// triangles.
struct Cluster
{
  Cluster(const Mesh&                 baseMesh,
          std::span<const glm::uvec3> triangleIndices,
          std::span<const uint32_t>   vertexIndices,
          file_writer&                alloc);
  offset_span<glm::u8vec3> meshTriIndices;
  offset_span<glm::vec3>   meshPositions;
  offset_span<glm::vec3>   meshNormals;
  offset_span<glm::vec2>   meshTexCoords;
  shaders::Material        material;
};

// Spatial data structure for runtime cluster selection.
struct LodHierarchyView
{
  offset_span<nvclusterlod_HierarchyNode> nodes;
  offset_span<nvclusterlod_Sphere>        groupCumulativeBoundingSpheres;
  offset_span<float>                      groupCumulativeQuadricError;
};

// Per-mesh data for the scene
struct ClusteredMesh
{
  ClusteredMesh(nvclusterlod_Context  context,
                const Mesh&           mesh,
                const SceneLodConfig& lodConfig,
                file_writer&          alloc);
  ClusteredMesh(nvclusterlod_Context             context,
                const Mesh&                      baseMesh,
                nvclusterlod::LocalizedLodMesh&& lodMesh,
                file_writer&                     alloc);
  LodHierarchyView             hierarchy;
  offset_span<nvcluster_Range> clusterTriangleRanges;
  offset_span<nvcluster_Range> clusterVertexRanges;
  Cluster                      clusteredMesh;
  offset_span<nvcluster_Range> lodLevelGroups;  // needed for fixed LOD
  offset_span<uint32_t>        clusterGeneratingGroups;
  offset_span<nvcluster_Range> groupClusterRanges;
  offset_span<offset_span<uint32_t>> groupGeneratingGroups;  // TODO: replace with linearized array?
  offset_span<offset_span<uint32_t>> groupGeneratedGroups;  // TODO: rename/too similar
  AABB aabb;
};

// Scene geometry data for one group of clusters in vulkan buffers. This is the
// granularity of streaming.
class ClusterGroupGeometryVk
{
public:
  ClusterGroupGeometryVk(const vko::Device&   device,
                         vkobj::Staging&      staging,
                         VkBuffer             memoryBuffer,
                         PoolAllocator&       memoryPool,
                         const ClusteredMesh& mesh,
                         uint32_t             groupIndex);

  vkobj::DeviceAddress<shaders::ClusterGeometry> clusterGeometryAddressesAddress() const
  {
    return m_clusterGeometryAddressesAddress;
  }

  vkobj::DeviceAddress<uint32_t> clusterGeneratingGroupsAddress() const
  {
    return m_clusterGeneratingGroupsAddress;
  }

  // DANGER: this GPU data is not populated until after CLASes are built. That
  // can only happen after the geometry is uploaded. It is only stored here to
  // combine it in the same allocation.
  vkobj::DeviceAddress<VkDeviceAddress> clasAddressesAddress() const
  {
    return m_clasAddressesAddress;
  }

private:
  PoolMemory m_alloc;
  vkobj::DeviceAddress<shaders::ClusterGeometry> m_clusterGeometryAddressesAddress;
  vkobj::DeviceAddress<uint32_t>        m_clusterGeneratingGroupsAddress;
  vkobj::DeviceAddress<VkDeviceAddress> m_clasAddressesAddress;
};

// Limits of the scene, used by various systems for conservative allocation
struct SceneCounts
{
  uint32_t totalMeshes              = 0;
  uint32_t totalInstances           = 0;
  uint32_t totalGroups              = 0;
  uint32_t totalClusters            = 0;
  uint32_t totalTriangles           = 0;
  uint32_t totalVertices            = 0;
  uint32_t maxClustersPerGroup      = 0;
  uint32_t maxClustersPerMesh       = 0;
  uint32_t maxLod0ClustersPerMesh   = 0;
  uint32_t maxClusterTriangleCount  = 0;
  uint32_t maxClusterVertexCount    = 0;
  uint32_t maxInstancesPerMesh      = 0;
  uint32_t maxLODLevel              = 0;
  uint32_t maxTotalInstanceClusters = 0;
  uint32_t maxTotalInstanceNodes    = 0;
};

// In-memory image description and raw data
struct SceneImage
{
  SceneImage() = default;
  SceneImage(const Image& image, file_writer& alloc);
  VkFormat                     format;
  VkExtent2D                   extent;
  offset_span<const std::byte> data;
};

// Scene data in system memory. In practice everything points to data in a
// single memory mapped render cache file. This struct itself resides there.
struct Scene
{
  uint32_t                   version = 0;
  offset_span<ClusteredMesh> meshes;
  offset_span<Instance>      instances;
  offset_span<SceneImage>    images;
  offset_span<uint32_t> meshGroupOffsets;  // includes total count in the last element
  offset_span<uint32_t> meshInstanceCounts;  // Number of instances referencing each mesh
  SceneCounts counts;
  AABB        worldAABB                     = AABB::make_empty();
  float       maxWorldDiagonalInObjectSpace = 0.0f;
};

// A render cache file memory mapping and a pointer into it
struct SceneFile
{
  SceneFile(decodeless::file&& memoryMap, const fs::path& path = fs::path());
  const Scene*     data;
  fs::path         path;  // generated if empty
  decodeless::file memoryMap;
};

std::optional<SceneFile> makeSceneFromCache(const fs::path& gltfPath,
                                            const fs::path& cachePath);
SceneFile                makeSceneFromGltf(const fs::path&       gltfPath,
                                           const fs::path&       cachePath,
                                           const SceneLodConfig& lodConfig,
                                           TaskProgress&         progress);
SceneFile                makeSceneFromGenerated(GeneratedScene&&      generatedScene,
                                                const fs::path&       cachePath,
                                                const SceneLodConfig& lodConfig,
                                                TaskProgress&         progress);

// Per-mesh vulkan data for rendering
struct ClusteredMeshVk
{
  ClusteredMeshVk(vkobj::Staging&      staging,
                  const vko::Device&   device,
                  vko::vma::Allocator& allocator,
                  const ClusteredMesh& clusteredMesh);
  vkobj::Buffer<nvclusterlod_HierarchyNode> nodes;
  vkobj::Buffer<shaders::ClusterGroup>      groups;  // Streaming indirection
  vkobj::Buffer<float> groupQuadricErrors;  // For traversal without streaming
  vkobj::Buffer<nvclusterlod_Sphere> groupBoundingSphers;  // For traversal without streaming
  vkobj::Buffer<uint8_t> groupLodLevels;                   // visualization only
  static_assert(sizeof(nvclusterlod_Sphere) == sizeof(float) * 4);

  VkDeviceSize deviceMemoryUsage() const
  {
    return nodes.sizeBytes() + groups.sizeBytes() + groupQuadricErrors.sizeBytes()
           + groupBoundingSphers.sizeBytes() + groupLodLevels.sizeBytes();
  }
};

// Vulkan buffers of Scene data needed for rendering on the GPU with the
// exception of ClusterGroupGeometryVk, which is loaded by the streaming system
// and not owned by SceneVK.
struct SceneVK
{
  SceneVK(vkobj::Staging&      staging,
          const vko::Device&   device,
          vko::vma::Allocator& allocator,
          VkQueue              queue,
          const Scene&         scene);

  // Clears pointers to streamed memory. Used when re-creating the streaming object.
  // DANGER: dangling pointers if we forget
  // Perhaps SceneVK should be mutable and own ClusterGroupGeometryVk objects,
  // plumbed through from the streaming thread.
  void cmdResetStreaming(const vko::Device&       device,
                         const Scene&             scene,
                         vkobj::DedicatedStaging& staging);

  VkDeviceSize deviceMemoryUsage() const { return staticDeviceMemoryUsage; }

  // Everything needed to render the scene, with the exception of
  // ClusterGroupGeometryVk
  std::vector<ClusteredMeshVk>       clusteredMeshes;
  SceneCounts                        counts;
  vkobj::Buffer<Instance>            instances;
  vkobj::Buffer<shaders::Mesh>       meshPointers;
  vkobj::Buffer<uint8_t>             allGroupNeededFlags;
  std::vector<TextureVk>             textures;
  std::vector<VkDescriptorImageInfo> textureDescriptors;

  VkDeviceSize staticDeviceMemoryUsage = 0;

  // Counts of geometry streamed so far. Could be used for conservative
  // allocations but gets big fast. Better to place a global limit on resident
  // cluster groups.
  uint64_t totalResidentClusters         = 0;
  uint64_t totalResidentInstanceClusters = 0;
};

namespace shaders {

inline std::ostream& operator<<(std::ostream& os, const shaders::Material& x)
{
  using numerical_chars::operator<<;
  os << "Material{\n";
  os << "  albedo " << x.albedo << "\n";
  os << "  albedoTexture " << x.albedoTexture << "\n";
  os << "  metallicRoughnessTexture " << x.metallicRoughnessTexture << "\n";
  os << "  roughness " << x.roughness << "\n";
  os << "  metallic " << x.metallic << "\n";
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const shaders::ClusterGeometry& x)
{
  using numerical_chars::operator<<;
  PrefixedLines indent(os.rdbuf(), "  ");
  std::ostream  ios(&indent);
  ios << "ClusterGeometry{\n";
  rangeSummaryVk<glm::u8vec3>(ios << "triangleVertices ",
                              x.triangleVerticesAddress, x.triangleCount)
      << "\n";
  rangeSummaryVk<glm::vec3>(ios << "vertexPositions ", x.vertexPositionsAddress, x.vertexCount)
      << "\n";
  rangeSummaryVk<glm::vec3>(ios << "vertexNormals ", x.vertexNormalsAddress, x.vertexCount)
      << "\n";
  if(x.vertexTexcoordsAddress)
    rangeSummaryVk<glm::vec2>(ios << "vertexTexcoords ", x.vertexTexcoordsAddress, x.vertexCount)
        << "\n";
  else
    ios << "vertexTexcoords null\n";
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const shaders::ClusterGroup& x)
{
  PrefixedLines indent(os.rdbuf(), "  ");
  std::ostream  ios(&indent);
  ios << "ClusterGroup{\n";
  ios << "clusterGeometryAddressesAddress " << x.clusterGeometryAddressesAddress << "\n";
  ios << "clusterGeneratingGroupsAddress " << x.clusterGeneratingGroupsAddress << "\n";
  ios << "clasAddressesAddress " << x.clasAddressesAddress << "\n";
  ios << "clusterCount " << x.clusterCount << "\n";
  rangeSummaryVk<shaders::ClusterGeometry>(ios << "clusterGeometryAddresses ",
                                           x.clusterGeometryAddressesAddress, x.clusterCount)
      << "\n";
  rangeSummaryVk<uint32_t>(ios << "clusterGeneratingGroups ",
                           x.clusterGeneratingGroupsAddress, x.clusterCount)
      << "\n";
  rangeSummaryVk<uint64_t>(ios << "clasAddresses ", x.clasAddressesAddress, x.clusterCount)
      << "\n";
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const shaders::Mesh& x)
{
  PrefixedLines indent(os.rdbuf(), "  ");
  std::ostream  ios(&indent);
  ios << "Mesh{\n";
  ios << "nodesAddress " << x.nodesAddress << "\n";
  ios << "groupsAddress " << x.groupsAddress << "\n";
  ios << "groupQuadricErrorsAddress " << x.groupQuadricErrorsAddress << "\n";
  ios << "groupBoundingSpheresAddress " << x.groupBoundingSpheresAddress << "\n";
  ios << "groupNeededFlagsAddress " << x.groupNeededFlagsAddress << "\n";
  ios << "groupLodLevelsAddress " << x.groupLodLevelsAddress << "\n";
  ios << "material " << x.material << "\n";
  ios << "groupCount " << x.groupCount << "\n";
  ios << "residentClusterCount " << x.residentClusterCount << "\n";
  rangeSummaryVk<shaders::ClusterGroup>(ios << "groups ", x.groupsAddress, x.groupCount)
      << "\n";
  os << "}";
  return os;
}

}  // namespace shaders
