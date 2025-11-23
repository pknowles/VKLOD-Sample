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

#include <cstddef>
#include <lod_traverser.hpp>
#include <optional>
#include <sample_profiler.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <string_view>

// Helper function to optionally create a GPU profiler sample
template <class CommandBufferType>
auto makeOptionalSection(const vko::Device& device,
                         SampleProfiler*    profiler,
                         CommandBufferType& cmd,
                         std::string_view   name)
{
  using SampleType = std::optional<ScopedGpuTimer<CommandBufferType>>;
  if(profiler)
    return SampleType(std::in_place, device, *profiler, cmd, std::string(name));
  return SampleType{};
}

// Type conversion and column major (glm) to row major (vk)
inline VkTransformMatrixKHR makeVulkanMatrix(const glm::mat4& m)
{
  return {{{m[0][0], m[1][0], m[2][0], m[3][0]},  //
           {m[0][1], m[1][1], m[2][1], m[3][1]},  //
           {m[0][2], m[1][2], m[2][2], m[3][2]}}};
}

inline shaderc::CompileOptions makeTraverserCompileOptions(SampleGlslCompiler& glslCompiler,
                                                           bool perInstance)
{
  shaderc::CompileOptions options = glslCompiler.defaultOptions();
  options.AddMacroDefinition("TRAVERSE_PER_INSTANCE", perInstance ? "1" : "0");
  options.AddMacroDefinition("IS_RASTERIZATION", "0");
  return options;
}

inline vkobj::Buffer<VkAccelerationStructureInstanceKHR> createDeviceInstances(
    vkobj::Staging&      staging,
    const vko::Device&   device,
    vko::vma::Allocator& allocator,
    VkQueue              queue,
    const Scene&         scene)
{
  // This initializes the tlas input. Much of it is constant (even the transform
  // for this static demo), with the exception of the BLAS address that changes
  // every frame for LOD. With per-instance traversal there is one BLAS per
  // instance and we can write blas addresses directly into the tlas input
  // structs. Per-mesh traversal fills in the BLAS references using a compute
  // shader.
  std::vector<VkAccelerationStructureInstanceKHR> tlasInputHost;
  tlasInputHost.reserve(uint32_t(scene.instances.size()));
  uint32_t instanceIndex = 0;
  for(const Instance& instance : scene.instances)
  {
    tlasInputHost.push_back(VkAccelerationStructureInstanceKHR{
        .transform           = makeVulkanMatrix(instance.transform),
        .instanceCustomIndex = (instanceIndex++) & 0xFFFFFFu,
        .mask                = 0xff,
        .instanceShaderBindingTableRecordOffset = 0,
        .flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR,
        .accelerationStructureReference = 0,  // Written by blas build directly or write_instances.comp
    });
  }

  vkobj::Buffer<VkAccelerationStructureInstanceKHR> result =
      vko::upload(staging, device, allocator, tlasInputHost,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                      | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  memoryBarrier(device, staging.commandBuffer(), VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
  staging.submit();
  vko::check(device.vkQueueWaitIdle(queue));  // TODO: remove this?
  return result;
}

LodInstanceTraverser::LodInstanceTraverser(const vko::Device&   device,
                                           vko::vma::Allocator& allocator,
                                           vkobj::Staging&      staging,
                                           VkQueue              queue,
                                           SampleGlslCompiler&  glslCompiler,
                                           VkCommandPool        initPool,
                                           VkQueue              initQueue,
                                           [[maybe_unused]] uint32_t initQueueFamilyIndex,
                                           const Scene&   scene,
                                           const SceneVK& sceneVk,
                                           uint32_t       maxTotalClusterGroups)
    : m_traversalConstants(device,
                           1,
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           allocator)
    , m_uboDescriptorSet(device,
                         VK_SHADER_STAGE_COMPUTE_BIT,
                         {{shaders::BTraversalConstants, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                           VkDescriptorBufferInfo{m_traversalConstants, 0, VK_WHOLE_SIZE}}})
    , m_traverseInitPipeline(device,
                             glslCompiler,
                             "traverse_init.comp.glsl",
                             m_uboDescriptorSet.layout(),
                             makeTraverserCompileOptions(glslCompiler, true))
    , m_traversePipeline(device,
                         glslCompiler,
                         "traverse.comp.glsl",
                         m_uboDescriptorSet.layout(),
                         makeTraverserCompileOptions(glslCompiler, true))
    , m_traverseVerifyPipeline(device,
                               glslCompiler,
                               "traverse_verify.comp.glsl",
                               m_uboDescriptorSet.layout(),
                               makeTraverserCompileOptions(glslCompiler, true))
    , m_writeIndirectSizePipeline(device,
                                  glslCompiler,
                                  "traverse_write_indirect_size.comp.glsl",
                                  m_uboDescriptorSet.layout(),
                                  makeTraverserCompileOptions(glslCompiler, true))
    , m_writeSelectedClustersPipeline(device,
                                      glslCompiler,
                                      "traverse_write_selected_clusters.comp.glsl",
                                      m_uboDescriptorSet.layout(),
                                      makeTraverserCompileOptions(glslCompiler, true))
    , m_clusterQueue(device,
                     maxTotalClusterGroups * scene.counts.maxClustersPerGroup
                         * 2 /* typically we'll traverse twice as many clusters as are selected */,
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                         | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     allocator)
    , m_blas(device,
             allocator,
             scene.counts.totalInstances,
             scene.counts.maxLod0ClustersPerMesh,
             maxTotalClusterGroups * scene.counts.maxClustersPerGroup)
    , m_selectedClusters(device,
                         maxTotalClusterGroups * scene.counts.maxClustersPerGroup,
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         allocator)
    , m_writeSelectedClustersDispatchIndirect(device,
                                              1,
                                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                  | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                  | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                              allocator)
    , m_nodeQueue(device,
                  scene.counts.maxTotalInstanceNodes,
                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                      | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  allocator)
    , m_jobStatus(device,
                  1,
                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                      | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  allocator)
    , m_tlasInfos(createDeviceInstances(staging, device, allocator, queue, scene))
{
  {
    vkobj::ImmediateCommandBuffer initCmd(device, initPool, initQueue);
    device.vkCmdFillBuffer(initCmd, m_nodeQueue, 0, m_nodeQueue.sizeBytes(), 0);
    device.vkCmdFillBuffer(initCmd, m_clusterQueue, 0, m_clusterQueue.sizeBytes(), 0);
  }


  // Perform an initial traversal of the scene. The output is needed to build
  // the initial TLAS before an update command buffer can be recorded.
  {
    vkobj::ImmediateCommandBuffer cmd(device, initPool, initQueue);
    shaders::TraversalParams      traversalParams = initialTraversalParams();
    memoryBarrier(device, cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    // DANGER: Using makeSignalled() with null profiler, which
    traverse(device, allocator, traversalParams, sceneVk, m_blas.input(),
             m_blas.inputPointers(), nullptr, vko::SemaphoreValue::makeSignalled(), cmd);
  }

#if !defined(NDEBUG)
  // Old school printf debugging for indirect arguments
  BufferDownloader downloader(device, initQueueFamilyIndex, allocator);
  rangeSummaryVk(std::cerr << "Meshes: ", sceneVk.meshPointers) << "\n";
  rangeSummaryVk(std::cerr << "BLAS Input: ", m_blas.input()) << "\n";
#endif

  // Build the BLAS and create the intial TLAS
  {
    vkobj::ImmediateCommandBuffer cmd(device, initPool, initQueue);

    // The BLAS build can write directly into the TLAS input as there is a BLAS per instance
    m_blas.cmdBuild(device, cmd, m_tlasInfos);

    // The TLAS must be created after m_tlasInfos has been populated with
    // real data. This is because it does an initial
    // VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR when created and we record a
    // VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR to resubmit each frame.
    m_tlas.emplace(device, allocator, m_tlasInfos, cmd);
  }
}

void LodInstanceTraverser::traverseAndBuildBVH(const vko::Device&   device,
                                               vko::vma::Allocator& allocator,
                                               const shaders::TraversalParams& traversalParams,
                                               const SceneVK&  sceneVk,
                                               SampleProfiler& profiler,
                                               vkobj::SemaphoreValue submitSemaphore,
                                               VkCommandBuffer cmd)
{
  {
    ScopedGpuTimer sample(device, profiler, cmd, "Traverse Instance LOD");
    traverse(device, allocator, traversalParams, sceneVk, m_blas.input(),
             m_blas.inputPointers(), &profiler, submitSemaphore, cmd);
  }

  {
    // The BLAS build can write directly into the TLAS input as there is a BLAS per instance
    m_blas.cmdBuild(device, cmd, m_tlasInfos);
    m_tlas->cmdUpdate(device, m_tlasInfos, cmd, false);
  }
}

void LodInstanceTraverser::traverse(
    const vko::Device&              device,
    vko::vma::Allocator&            allocator,
    const shaders::TraversalParams& traversalParams,
    const SceneVK&                  sceneVk,
    const vkobj::Buffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>& blasInput,
    const vkobj::Buffer<VkDeviceAddress>& blasInputClusters,
    SampleProfiler*                       profiler,
    vkobj::SemaphoreValue                 submitSemaphore,
    VkCommandBuffer                       cmd)
{
  // Zero the job queue
  shaders::JobStatus initJobStatus{};
  device.vkCmdUpdateBuffer(cmd, m_jobStatus, 0, sizeof(shaders::JobStatus), &initJobStatus);

  auto statsBuffer = m_stats.getFreeBuffer(device, allocator);
  device.vkCmdFillBuffer(cmd, statsBuffer.deviceBuffer(), 0,
                         statsBuffer.deviceBuffer().sizeBytes(), 0);

  assert(sceneVk.clusteredMeshes.size() <= blasInput.size());
  shaders::TraversalConstants traversalConstants{
      .traversalParams      = traversalParams,
      .meshesAddress        = sceneVk.meshPointers,
      .instancesAddress     = sceneVk.instances,
      .nodeQueueAddress     = m_nodeQueue,
      .clusterQueueAddress  = m_clusterQueue,
      .jobStatusAddress     = m_jobStatus,
      .traverseStatsAddress = statsBuffer.deviceBuffer(),
      .blasInputAddress = vkobj::deviceReinterpretCast<shaders::ClusterBLASInfoNV>(
          vkobj::DeviceAddress(blasInput)),
      .blasInputClustersAddress = blasInputClusters,
      .drawClustersAddress      = vkobj::DeviceAddress<shaders::DrawCluster>(0),
      .drawMeshTasksIndirectAddress =
          vkobj::DeviceAddress<shaders::DrawMeshTasksIndirect>(0),
      .drawStatsAddress = vkobj::DeviceAddress<shaders::DrawStats>(0),
      .meshInstances    = vkobj::DeviceAddress<shaders::MeshInstances>(0),
      .sortingMeshInstances = vkobj::DeviceAddress<shaders::SortingMeshInstances>(0),
      .selectedClusters = m_selectedClusters,
      .writeSelectedClustersDispatchIndirect = m_writeSelectedClustersDispatchIndirect,
      .maxSelectedClusters = uint32_t(m_blas.maxTotalClusters()),
      .nodeQueueSize       = uint32_t(m_nodeQueue.size()),
      .clusterQueueSize    = uint32_t(m_clusterQueue.size()),
      .itemsSize = uint32_t(sceneVk.instances.size()),  // traverse per-instance
      .drawClustersSize = 0,
  };

  // Common to all shaders
  // Originally, this was designed to produce a re-submittable command buffer.
  // The traversal constants UBO could trivially be push constants instead.
  device.vkCmdUpdateBuffer(cmd, m_traversalConstants, 0,
                           sizeof(shaders::TraversalConstants), &traversalConstants);
  VkDescriptorSet uboSet = m_uboDescriptorSet;
  device.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 m_traverseInitPipeline.pipelineLayout, 0, 1,
                                 &uboSet, 0, nullptr);

  // Barrier: buffer updates -> traverse init
  memoryBarrier(device, cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: traverse init
  device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverseInitPipeline);
  {
    auto sec = makeOptionalSection(device, profiler, cmd, "Traverse Init");
    device.vkCmdDispatch(cmd,
                         div_ceil(uint32_t(sceneVk.instances.size()),
                                  uint32_t(TRAVERSAL_WORKGROUP_SIZE)),
                         1, 1);
  }

  // Barrier: traverse init -> traverse
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: traverse
  device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traversePipeline);
  {
    auto sec = makeOptionalSection(device, profiler, cmd, "Traverse Main");
    device.vkCmdDispatch(cmd, div_ceil(4096u, uint32_t(TRAVERSAL_WORKGROUP_SIZE)), 1, 1);
  }

  // Barrier: traverse -> traverse verify
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: traverse verify
  device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverseVerifyPipeline);
  {
    auto sec = makeOptionalSection(device, profiler, cmd, "Traverse Verify");
    device.vkCmdDispatch(cmd,
                         div_ceil(uint32_t(sceneVk.instances.size()),
                                  uint32_t(TRAVERSAL_WORKGROUP_SIZE)),
                         1, 1);
  }

  // Barrier: traverse verify -> write selected clusters
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: write selected clusters, plus computing the indirect launch size
  {
    auto sec = makeOptionalSection(device, profiler, cmd, "Write Selected Clusters");
    device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_writeIndirectSizePipeline);
    device.vkCmdDispatch(cmd, 1, 1, 1);
    memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
    device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_writeSelectedClustersPipeline);
    device.vkCmdDispatchIndirect(cmd, m_writeSelectedClustersDispatchIndirect, 0);
  }

  // Barrier: write selected clusters -> BLAS build (and stats buffer)
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
                    | VK_PIPELINE_STAGE_TRANSFER_BIT);

  statsBuffer.copyToHost(device, submitSemaphore, cmd);
  m_stats.queueBuffer(std::move(statsBuffer));
}

LodMeshTraverser::LodMeshTraverser(const vko::Device&   device,
                                   vko::vma::Allocator& allocator,
                                   vkobj::Staging&      staging,
                                   VkQueue              queue,
                                   SampleGlslCompiler&  glslCompiler,
                                   VkCommandPool        initPool,
                                   VkQueue              initQueue,
                                   [[maybe_unused]] uint32_t initQueueFamilyIndex,
                                   const Scene&   scene,
                                   const SceneVK& sceneVk,
                                   uint32_t       maxTotalClusterGroups)
    : m_traversalConstants(device,
                           1,
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           allocator)
    , m_uboDescriptorSet(device,
                         VK_SHADER_STAGE_COMPUTE_BIT,
                         {{shaders::BTraversalConstants, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                           VkDescriptorBufferInfo{m_traversalConstants, 0, VK_WHOLE_SIZE}}})
    , m_traverseSortInstances(device,
                              glslCompiler,
                              "traverse_sort_instances.comp.glsl",
                              VK_NULL_HANDLE,
                              makeTraverserCompileOptions(glslCompiler, false))
    , m_traverseInit(device,
                     glslCompiler,
                     "traverse_init.comp.glsl",
                     m_uboDescriptorSet.layout(),
                     makeTraverserCompileOptions(glslCompiler, false))
    , m_traverse(device,
                 glslCompiler,
                 "traverse.comp.glsl",
                 m_uboDescriptorSet.layout(),
                 makeTraverserCompileOptions(glslCompiler, false))
    , m_traverseVerify(device,
                       glslCompiler,
                       "traverse_verify.comp.glsl",
                       m_uboDescriptorSet.layout(),
                       makeTraverserCompileOptions(glslCompiler, false))
    , m_instanceWriter(device,
                       glslCompiler,
                       "write_instances.comp.glsl",
                       VK_NULL_HANDLE,
                       makeTraverserCompileOptions(glslCompiler, false))
    , m_nodeQueue(device,
                  scene.counts.totalMeshes * 100,  // Estimate: max nodes per mesh
                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                      | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  allocator)
    , m_clusterQueue(device,
                     maxTotalClusterGroups * scene.counts.maxClustersPerGroup,
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                         | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     allocator)
    , m_jobStatus(device,
                  1,
                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                      | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  allocator)
    , m_meshInstances(device,
                      scene.counts.totalMeshes,
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      allocator)
    , m_sortingMeshInstances(device,
                             scene.counts.totalMeshes,
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             allocator)
    , m_blasAddresses(device,
                      scene.counts.totalMeshes,
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      allocator)
    , m_blas(device,
             allocator,
             scene.counts.totalMeshes,
             scene.counts.maxLod0ClustersPerMesh,
             maxTotalClusterGroups * scene.counts.maxClustersPerGroup)
    , m_tlasInfos(createDeviceInstances(staging, device, allocator, queue, scene))
{
  m_device = &device;

  // Create shader compile options - these are needed for later recompilation
  shaderc::CompileOptions options = glslCompiler.defaultOptions();
  options.AddMacroDefinition("TRAVERSE_PER_INSTANCE", "0");
  options.AddMacroDefinition("IS_RASTERIZATION", "0");

  // TODO: move this computation to SceneCounts
  size_t totalNodes = 0;
  for(auto& mesh : scene.meshes)
  {
    totalNodes += mesh.hierarchy.nodes.size();
  }

  // Allocate only the node queue and job status buffer. The Cluster queue is
  // allocated just before use.
  m_nodeQueue = vkobj::Buffer<shaders::EncodedNodeJob>(
      device, totalNodes,
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
          | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  m_jobStatus = vkobj::Buffer<shaders::JobStatus>(
      device, 1,
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
          | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);

  {
    vkobj::ImmediateCommandBuffer initCmd(device, initPool, initQueue);
    device.vkCmdFillBuffer(initCmd, m_nodeQueue, 0, m_nodeQueue.sizeBytes(), 0);
    device.vkCmdFillBuffer(initCmd, m_clusterQueue, 0, m_clusterQueue.sizeBytes(), 0);
  }

  // Perform an initial traversal of the scene. The output is needed to build
  // the initial TLAS before an update command buffer can be recorded.
  {
    vkobj::ImmediateCommandBuffer cmd(device, initPool, initQueue);
    shaders::TraversalParams      traversalParams = initialTraversalParams();
    memoryBarrier(device, cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    // DANGER: Using makeSignalled() with null profiler. If profiler implementation changes to
    // read samples eagerly (e.g. during addSample), this could read stale GPU timestamp values.
    // Currently safe because profiler is nullptr so no samples are recorded.
    traverse(device, allocator, traversalParams, sceneVk, m_blas.input(),
             m_blas.inputPointers(), nullptr, vko::SemaphoreValue::makeSignalled(), cmd);
  }

#if !defined(NDEBUG)
  // Old school printf debugging for indirect arguments
  BufferDownloader downloader(device, initQueueFamilyIndex, allocator);
  rangeSummaryVk(std::cerr << "Meshes: ", sceneVk.meshPointers) << "\n";
  rangeSummaryVk(std::cerr << "BLAS Input: ", m_blas.input()) << "\n";
#endif

  {
    // Build the BLAS and create the intial TLAS
    vkobj::ImmediateCommandBuffer cmd(device, initPool, initQueue);
    m_blas.cmdBuild(device, cmd, m_blasAddresses);
    memoryBarrier(device, cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    shaders::WriteInstancesConstant constant{
        .instances         = sceneVk.instances,
        .meshBlasAddresses = m_blasAddresses,
        .tlasInfos = vkobj::deviceReinterpretCast<shaders::InstanceInfo>(
            vkobj::DeviceAddress(m_tlasInfos)),
        .instancesSize = uint32_t(m_tlasInfos.size()),
    };
    device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_instanceWriter);
    device.vkCmdPushConstants(cmd, m_instanceWriter.pipelineLayout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constant), &constant);
    device.vkCmdDispatch(cmd, div_ceil(uint32_t(m_tlasInfos.size()), uint32_t(TRAVERSAL_WORKGROUP_SIZE)),
                         1, 1);
    memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

    // The TLAS must be created after m_tlasInfos has been populated with
    // real data. This is because it does an initial
    // VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR when created and we record a
    // VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR to resubmit each frame.
    m_tlas.emplace(device, allocator, m_tlasInfos, cmd);
  }
}

void LodMeshTraverser::traverseAndBuildBVH(const vko::Device&   device,
                                           vko::vma::Allocator& allocator,
                                           const shaders::TraversalParams& traversalParams,
                                           const SceneVK&  sceneVk,
                                           SampleProfiler& profiler,
                                           vkobj::SemaphoreValue submitSemaphore,
                                           VkCommandBuffer cmd)
{
  {
    ScopedGpuTimer sample(device, profiler, cmd, "Traverse Scene LOD");
    traverse(device, allocator, traversalParams, sceneVk, m_blas.input(),
             m_blas.inputPointers(), &profiler, submitSemaphore, cmd);
  }

  {
    ScopedGpuTimer outerSec(device, profiler, cmd, "Build BVH");
    // BLAS build
    {
      ScopedGpuTimer sec(device, profiler, cmd, "BLAS");
      m_blas.cmdBuild(device, cmd, m_blasAddresses);
    }

    // BLAS build -> write instances
    memoryBarrier(device, cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // write instances
    shaders::WriteInstancesConstant constant{
        .instances         = sceneVk.instances,
        .meshBlasAddresses = m_blasAddresses,
        .tlasInfos = vkobj::deviceReinterpretCast<shaders::InstanceInfo>(
            vkobj::DeviceAddress(m_tlasInfos)),
        .instancesSize = uint32_t(m_tlasInfos.size()),
    };
    device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_instanceWriter);
    device.vkCmdPushConstants(cmd, m_instanceWriter.pipelineLayout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constant), &constant);
    {
      device.vkCmdDispatch(cmd, div_ceil(uint32_t(m_tlasInfos.size()), uint32_t(TRAVERSAL_WORKGROUP_SIZE)),
                           1, 1);
    }

    // write instances -> TLAS build
    memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

    // TLAS build
    {
      m_tlas->cmdUpdate(device, m_tlasInfos, cmd, false);
    }
  }
}

void LodMeshTraverser::traverse(const vko::Device&              device,
                                vko::vma::Allocator&            allocator,
                                const shaders::TraversalParams& traversalParams,
                                const SceneVK&                  sceneVk,
                                const vkobj::Buffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>& blasInput,
                                const vkobj::Buffer<VkDeviceAddress>& blasInputClusters,
                                SampleProfiler*       profiler,
                                vkobj::SemaphoreValue submitSemaphore,
                                VkCommandBuffer       cmd)
{
  // Zero the job queue
  shaders::JobStatus initJobStatus{};
  device.vkCmdUpdateBuffer(cmd, m_jobStatus, 0, sizeof(shaders::JobStatus), &initJobStatus);

  // Fill the per-mesh k-nearest instance buffer with ones in preparation for a short bubble sort
  device.vkCmdFillBuffer(cmd, m_sortingMeshInstances, 0,
                         m_sortingMeshInstances.sizeBytes(), 0xffffffff);

  auto statsBuffer = m_stats.getFreeBuffer(device, allocator);
  device.vkCmdFillBuffer(cmd, statsBuffer.deviceBuffer(), 0,
                         statsBuffer.deviceBuffer().sizeBytes(), 0);

  // Barrier: buffer updates -> sort instances
  memoryBarrier(device, cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Find the closest k instances, that will be used during traversal
  shaders::SortInstancesConstant sortInstancesConstant{
      .traversalParams      = traversalParams,
      .instances            = sceneVk.instances,
      .meshes               = sceneVk.meshPointers,
      .sortingMeshInstances = m_sortingMeshInstances,
      .instancesSize        = uint32_t(sceneVk.instances.size()),
  };
  device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverseSortInstances);
  device.vkCmdPushConstants(cmd, m_traverseSortInstances.pipelineLayout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            sizeof(sortInstancesConstant), &sortInstancesConstant);
  {
    auto sec = makeOptionalSection(device, profiler, cmd, "Traverse Sort");
    device.vkCmdDispatch(cmd,
                         div_ceil(uint32_t(sceneVk.instances.size()),
                                  uint32_t(TRAVERSAL_WORKGROUP_SIZE)),
                         1, 1);
  }

  // Write traversal parameters
  shaders::TraversalConstants traversalConstants{
      .traversalParams      = traversalParams,
      .meshesAddress        = sceneVk.meshPointers,
      .instancesAddress     = sceneVk.instances,
      .nodeQueueAddress     = m_nodeQueue,
      .clusterQueueAddress  = m_clusterQueue,
      .jobStatusAddress     = m_jobStatus,
      .traverseStatsAddress = statsBuffer.deviceBuffer(),
      .blasInputAddress = vkobj::deviceReinterpretCast<shaders::ClusterBLASInfoNV>(
          vkobj::DeviceAddress<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>(blasInput)),
      .blasInputClustersAddress = blasInputClusters,
      .drawClustersAddress      = vkobj::DeviceAddress<shaders::DrawCluster>(0),
      .drawMeshTasksIndirectAddress =
          vkobj::DeviceAddress<shaders::DrawMeshTasksIndirect>(0),
      .drawStatsAddress     = vkobj::DeviceAddress<shaders::DrawStats>(0),
      .meshInstances        = m_meshInstances,
      .sortingMeshInstances = m_sortingMeshInstances,
      .selectedClusters     = vkobj::DeviceAddress<shaders::SelectedCluster>(0),
      .writeSelectedClustersDispatchIndirect =
          vkobj::DeviceAddress<shaders::DispatchIndirect>(0),
      .maxSelectedClusters = uint32_t(m_blas.maxTotalClusters()),
      .nodeQueueSize       = uint32_t(m_nodeQueue.size()),
      .clusterQueueSize    = uint32_t(m_clusterQueue.size()),
      .itemsSize           = uint32_t(sceneVk.meshPointers.size()),
      .drawClustersSize    = 0,
  };

  // Common to all shaders
  // Originally, this was designed to produce a re-submittable command buffer.
  // The traversal constants UBO could trivially be push constants instead.
  device.vkCmdUpdateBuffer(cmd, m_traversalConstants, 0,
                           sizeof(traversalConstants), &traversalConstants);
  device.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 m_traverseInit.pipelineLayout, 0, 1,
                                 m_uboDescriptorSet.ptr(), 0, nullptr);

  // Barrier: sort instances -> traverse init
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: traverse init
  device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverseInit);
  {
    auto sec = makeOptionalSection(device, profiler, cmd, "Traverse Init");
    device.vkCmdDispatch(cmd,
                         div_ceil(uint32_t(sceneVk.meshPointers.size()),
                                  uint32_t(TRAVERSAL_WORKGROUP_SIZE)),
                         1, 1);
  }

  // Barrier: traverse init -> traverse
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: traverse
  device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverse);
  {
    auto sec = makeOptionalSection(device, profiler, cmd, "Traverse Main");
    device.vkCmdDispatch(cmd, div_ceil(4096u, uint32_t(TRAVERSAL_WORKGROUP_SIZE)), 1, 1);
  }

  // Barrier: traverse -> traverse verify
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: traverse verify
  device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverseVerify);
  {
    auto sec = makeOptionalSection(device, profiler, cmd, "Traverse Verify");
    device.vkCmdDispatch(cmd,
                         div_ceil(uint32_t(sceneVk.meshPointers.size()),
                                  uint32_t(TRAVERSAL_WORKGROUP_SIZE)),
                         1, 1);
  }

  // Barrier: traverse verify -> BLAS build (and stats buffer)
  memoryBarrier(device, cmd, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
                    | VK_PIPELINE_STAGE_TRANSFER_BIT);

  statsBuffer.copyToHost(device, submitSemaphore, cmd);
  m_stats.queueBuffer(std::move(statsBuffer));
}
