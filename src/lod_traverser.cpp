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
#include <nvvk/profiler_vk.hpp>
#include <optional>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>

// Utility to handle a null profiler for traverser initialization. Would use
// std::optional, but nvvk::ProfilerVK::Section is not move safe.
struct OptionalSection
{
public:
  [[nodiscard]] OptionalSection(nvvk::ProfilerVK* profiler, const char* name, VkCommandBuffer cmd)
      : m_id(profiler ? profiler->beginSection(name, cmd, false, false) : 0)
      , m_cmd(cmd)
      , m_profiler(profiler)
  {
  }
  ~OptionalSection()
  {
    if(m_profiler)
      m_profiler->endSection(m_id, m_cmd);
  }
  OptionalSection(const OptionalSection& other)            = delete;
  OptionalSection& operator=(const OptionalSection& other) = delete;

private:
  nvvk::ProfilerVK::SectionID m_id       = {};
  VkCommandBuffer             m_cmd      = VK_NULL_HANDLE;
  nvvk::ProfilerVK*           m_profiler = nullptr;
};

// Type conversion and column major (glm) to row major (vk)
inline VkTransformMatrixKHR makeVulkanMatrix(const glm::mat4& m)
{
  return {{{m[0][0], m[1][0], m[2][0], m[3][0]},  //
           {m[0][1], m[1][1], m[2][1], m[3][1]},  //
           {m[0][2], m[1][2], m[2][2], m[3][2]}}};
}

inline vkobj::Buffer<VkAccelerationStructureInstanceKHR> createDeviceInstances(ResourceAllocator* allocator,
                                                                               VkCommandPool      initPool,
                                                                               VkQueue            initQueue,
                                                                               const Scene&       scene)
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
        .transform                              = makeVulkanMatrix(instance.transform),
        .instanceCustomIndex                    = (instanceIndex++) & 0xFFFFFFu,
        .mask                                   = 0xff,
        .instanceShaderBindingTableRecordOffset = 0,
        .flags                                  = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR,
        .accelerationStructureReference         = 0,  // Written by blas build directly or write_instances.comp
    });
  }

  vkobj::ImmediateCommandBuffer                     initCmd(allocator->getDevice(), initPool, initQueue);
  vkobj::Buffer<VkAccelerationStructureInstanceKHR> result(allocator, tlasInputHost,
                                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                               | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, initCmd);
  memoryBarrier(initCmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
  return result;
}

LodInstanceTraverser::LodInstanceTraverser(ResourceAllocator*        allocator,
                                           SampleGlslCompiler&       glslCompiler,
                                           VkCommandPool             initPool,
                                           VkQueue                   initQueue,
                                           [[maybe_unused]] uint32_t initQueueFamilyIndex,
                                           const Scene&              scene,
                                           const SceneVK&            sceneVk,
                                           uint32_t                  maxTotalClusterGroups)
    : m_traversalConstants(allocator, 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    , m_uboDescriptorSet(allocator->getDevice(),
                         VK_SHADER_STAGE_COMPUTE_BIT,
                         {{shaders::BTraversalConstants, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                           VkDescriptorBufferInfo{m_traversalConstants, 0, VK_WHOLE_SIZE}}})
    , m_clusterQueue(vkobj::Buffer<shaders::EncodedClusterJob>(
          allocator,
          maxTotalClusterGroups * scene.counts.maxClustersPerGroup * 2 /* typically we'll traverse twice as many clusters as are selected */,
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
    , m_blas(allocator,
             scene.counts.totalInstances,
             scene.counts.maxLod0ClustersPerMesh,
             maxTotalClusterGroups * scene.counts.maxClustersPerGroup)
    , m_selectedClusters(allocator,
                         maxTotalClusterGroups * scene.counts.maxClustersPerGroup,
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    , m_writeSelectedClustersDispatchIndirect(allocator,
                                              1,
                                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    , m_tlasInfos(createDeviceInstances(allocator, initPool, initQueue, scene))
{
  // Allocate only the node queue and job status buffer. The Cluster queue is
  // allocated just before use.
  m_nodeQueue = vkobj::Buffer<shaders::EncodedNodeJob>(allocator, scene.counts.maxTotalInstanceNodes,
                                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_jobStatus = vkobj::Buffer<shaders::JobStatus>(allocator, 1, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  {
    vkobj::ImmediateCommandBuffer initCmd(allocator->getDevice(), initPool, initQueue);
    vkCmdFillBuffer(initCmd, m_nodeQueue, 0, m_nodeQueue.size_bytes(), 0);
    vkCmdFillBuffer(initCmd, m_clusterQueue, 0, m_clusterQueue.size_bytes(), 0);
  }

  VkDevice device = allocator->getDevice();

  shaderc::CompileOptions options = glslCompiler.defaultOptions();
  options.AddMacroDefinition("TRAVERSE_PER_INSTANCE", "1");
  options.AddMacroDefinition("IS_RASTERIZATION", "0");

  // Compute shader to initialize the work queues
  m_traverseInitPipeline =
      vkobj::SimpleComputePipeline(device, glslCompiler, "traverse_init.comp.glsl", m_uboDescriptorSet.layout(), &options);

  // Main traversal compute shader
  m_traversePipeline =
      vkobj::SimpleComputePipeline(device, glslCompiler, "traverse.comp.glsl", m_uboDescriptorSet.layout(), &options);

  // Post-traversal verification. The bottom level acceleration structure
  // build cannot be given zero clusters. If Traversal did not produce any
  // clusters, default to the first (which should be lowest detail anyway).
  // TODO: change traversal so this is not needed
  m_traverseVerifyPipeline =
      vkobj::SimpleComputePipeline(device, glslCompiler, "traverse_verify.comp.glsl", m_uboDescriptorSet.layout(), &options);

  // Single-thread launch to write selectedClusterAlloc to the DispatchIndirect
  // buffer. This could almost be done in traverse_verify.comp.glsl if it
  // weren't for it possibly writting more selected clusters to handle empty
  // traversal output (e.g. in case of out of memory).
  m_writeIndirectSizePipeline = vkobj::SimpleComputePipeline(device, glslCompiler, "traverse_write_indirect_size.comp.glsl",
                                                             m_uboDescriptorSet.layout(), &options);

  // Compute shader to write the selected clusters to their individual BLAS input lists
  m_writeSelectedClustersPipeline = vkobj::SimpleComputePipeline(device, glslCompiler, "traverse_write_selected_clusters.comp.glsl",
                                                                 m_uboDescriptorSet.layout(), &options);

  // Perform an initial traversal of the scene. The output is needed to build
  // the initial TLAS before an update command buffer can be recorded.
  {
    vkobj::ImmediateCommandBuffer cmd(allocator->getDevice(), initPool, initQueue);
    shaders::TraversalParams      traversalParams = initialTraversalParams();
    memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    traverse(allocator, traversalParams, sceneVk, m_blas.input(), m_blas.inputPointers(), nullptr, {}, cmd);
  }

#if !defined(NDEBUG)
  // Old school printf debugging for indirect arguments
  BufferDownloader downloader(allocator->getDevice(), initQueueFamilyIndex, allocator->getStaging());
  rangeSummaryVk(std::cerr << "Meshes: ", sceneVk.meshPointers) << "\n";
  rangeSummaryVk(std::cerr << "BLAS Input: ", m_blas.input()) << "\n";
#endif

  // Build the BLAS and create the intial TLAS
  {
    vkobj::ImmediateCommandBuffer cmd(allocator->getDevice(), initPool, initQueue);

    // The BLAS build can write directly into the TLAS input as there is a BLAS per instance
    m_blas.cmdBuild(cmd, m_tlasInfos);

    // The TLAS must be created after m_tlasInfos has been populated with
    // real data. This is because it does an initial
    // VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR when created and we record a
    // VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR to resubmit each frame.
    m_tlas.emplace(allocator, m_tlasInfos, cmd);
  }
}

void LodInstanceTraverser::traverseAndBuildBVH(ResourceAllocator*              allocator,
                                               const shaders::TraversalParams& traversalParams,
                                               const SceneVK&                  sceneVk,
                                               nvvk::ProfilerVK&               profiler,
                                               vkobj::SemaphoreValue&&         submitSemaphore,
                                               VkCommandBuffer                 cmd)
{
  {
    nvvk::ProfilerVK::Section sec = profiler.timeRecurring("Traverse Scene LOD", cmd);
    traverse(allocator, traversalParams, sceneVk, m_blas.input(), m_blas.inputPointers(), &profiler,
             vkobj::SemaphoreValue{submitSemaphore}, cmd);
  }

  {
    nvvk::ProfilerVK::Section sec = profiler.timeRecurring("Build BVH", cmd);
    // The BLAS build can write directly into the TLAS input as there is a BLAS per instance
    m_blas.cmdBuild(cmd, m_tlasInfos);
    m_tlas->cmdUpdate(m_tlasInfos, cmd, false);
  }
}

void LodInstanceTraverser::traverse(ResourceAllocator*              allocator,
                                    const shaders::TraversalParams& traversalParams,
                                    const SceneVK&                  sceneVk,
                                    const vkobj::Buffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>& blasInput,
                                    const vkobj::Buffer<VkDeviceAddress>& blasInputClusters,
                                    nvvk::ProfilerVK*                     profiler,
                                    vkobj::SemaphoreValue&&               submitSemaphore,
                                    VkCommandBuffer                       cmd)
{
  // Zero the job queue
  shaders::JobStatus initJobStatus{};
  vkCmdUpdateBuffer(cmd, m_jobStatus, 0, sizeof(shaders::JobStatus), &initJobStatus);

  auto statsBuffer = m_stats.getFreeBuffer(allocator);
  vkCmdFillBuffer(cmd, statsBuffer.deviceBuffer(), 0, statsBuffer.deviceBuffer().size_bytes(), 0);

  assert(sceneVk.clusteredMeshes.size() <= blasInput.size());
  shaders::TraversalConstants traversalConstants{
      .traversalParams                       = traversalParams,
      .meshesAddress                         = sceneVk.meshPointers.address(),
      .instancesAddress                      = sceneVk.instances.address(),
      .nodeQueueAddress                      = m_nodeQueue.address(),
      .clusterQueueAddress                   = m_clusterQueue.address(),
      .jobStatusAddress                      = m_jobStatus.address(),
      .traverseStatsAddress                  = statsBuffer.deviceBuffer().address(),
      .blasInputAddress                      = deviceReinterpretCast<shaders::ClusterBLASInfoNV>(blasInput.address()),
      .blasInputClustersAddress              = blasInputClusters.address(),
      .drawClustersAddress                   = vkobj::DeviceAddress<shaders::DrawCluster>(0),
      .drawMeshTasksIndirectAddress          = vkobj::DeviceAddress<shaders::DrawMeshTasksIndirect>(0),
      .drawStatsAddress                      = vkobj::DeviceAddress<shaders::DrawStats>(0),
      .meshInstances                         = vkobj::DeviceAddress<shaders::MeshInstances>(0),
      .sortingMeshInstances                  = vkobj::DeviceAddress<shaders::SortingMeshInstances>(0),
      .selectedClusters                      = m_selectedClusters.address(),
      .writeSelectedClustersDispatchIndirect = m_writeSelectedClustersDispatchIndirect.address(),
      .maxSelectedClusters                   = uint32_t(m_blas.maxTotalClusters()),
      .nodeQueueSize                         = uint32_t(m_nodeQueue.size()),
      .clusterQueueSize                      = uint32_t(m_clusterQueue.size()),
      .itemsSize                             = uint32_t(sceneVk.instances.size()),  // traverse per-instance
      .drawClustersSize                      = 0,
  };

  // Common to all shaders
  // Originally, this was designed to produce a re-submittable command buffer.
  // The traversal constants UBO could trivially be push constants instead.
  vkCmdUpdateBuffer(cmd, m_traversalConstants, 0, sizeof(shaders::TraversalConstants), &traversalConstants);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverseInitPipeline.pipelineLayout, 0, 1,
                          &m_uboDescriptorSet.get(), 0, nullptr);

  // Barrier: buffer updates -> traverse init
  memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: traverse init
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverseInitPipeline);
  {
    OptionalSection sec(profiler, "Traverse Init", cmd);
    vkCmdDispatch(cmd, div_ceil(uint32_t(sceneVk.instances.size()), uint32_t(TRAVERSAL_WORKGROUP_SIZE)), 1, 1);
  }

  // Barrier: traverse init -> traverse
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: traverse
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traversePipeline);
  {
    OptionalSection sec(profiler, "Traverse Main", cmd);
    vkCmdDispatch(cmd, div_ceil(4096u, uint32_t(TRAVERSAL_WORKGROUP_SIZE)), 1, 1);
  }

  // Barrier: traverse -> traverse verify
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: traverse verify
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverseVerifyPipeline);
  {
    OptionalSection sec(profiler, "Traverse Verify", cmd);
    vkCmdDispatch(cmd, div_ceil(uint32_t(sceneVk.instances.size()), uint32_t(TRAVERSAL_WORKGROUP_SIZE)), 1, 1);
  }

  // Barrier: traverse verify -> write selected clusters
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: write selected clusters, plus computing the indirect launch size
  {
    OptionalSection sec(profiler, "Write Selected Clusters", cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_writeIndirectSizePipeline);
    vkCmdDispatch(cmd, 1, 1, 1);
    memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_writeSelectedClustersPipeline);
    vkCmdDispatchIndirect(cmd, m_writeSelectedClustersDispatchIndirect, 0);
  }

  // Barrier: write selected clusters -> BLAS build
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

  statsBuffer.copyToHost(submitSemaphore, cmd);
  m_stats.queueBuffer(std::move(statsBuffer));
}

LodMeshTraverser::LodMeshTraverser(ResourceAllocator*        allocator,
                                   SampleGlslCompiler&       glslCompiler,
                                   VkCommandPool             initPool,
                                   VkQueue                   initQueue,
                                   [[maybe_unused]] uint32_t initQueueFamilyIndex,
                                   const Scene&              scene,
                                   const SceneVK&            sceneVk,
                                   uint32_t                  maxTotalClusterGroups)
    : m_traversalConstants(allocator, 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    , m_uboDescriptorSet(allocator->getDevice(),
                         VK_SHADER_STAGE_COMPUTE_BIT,
                         {{shaders::BTraversalConstants, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                           VkDescriptorBufferInfo{m_traversalConstants, 0, VK_WHOLE_SIZE}}})
    , m_clusterQueue(vkobj::Buffer<shaders::EncodedClusterJob>(allocator,
                                                               maxTotalClusterGroups * scene.counts.maxClustersPerGroup,
                                                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
    , m_meshInstances(allocator,
                      scene.counts.totalMeshes,
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    , m_sortingMeshInstances(allocator,
                             scene.counts.totalMeshes,
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    , m_blasAddresses(allocator,
                      scene.counts.totalMeshes,
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    , m_blas(allocator,
             scene.counts.totalMeshes,
             scene.counts.maxLod0ClustersPerMesh,
             maxTotalClusterGroups * scene.counts.maxClustersPerGroup)
    , m_tlasInfos(createDeviceInstances(allocator, initPool, initQueue, scene))
{
  shaderc::CompileOptions options = glslCompiler.defaultOptions();
  options.AddMacroDefinition("TRAVERSE_PER_INSTANCE", "0");
  options.AddMacroDefinition("IS_RASTERIZATION", "0");

  // Compute shader to find the k nearest instances per mesh
  m_traverseSortInstances = vkobj::SimpleComputePipeline<shaders::SortInstancesConstant>(
      allocator->getDevice(), glslCompiler, "traverse_sort_instances.comp.glsl", VK_NULL_HANDLE, &options);

  // Compute shader to initialize the work queues
  m_traverseInit = vkobj::SimpleComputePipeline(allocator->getDevice(), glslCompiler, "traverse_init.comp.glsl",
                                                m_uboDescriptorSet.layout(), &options);

  // Main traversal compute shader
  m_traverse = vkobj::SimpleComputePipeline(allocator->getDevice(), glslCompiler, "traverse.comp.glsl",
                                            m_uboDescriptorSet.layout(), &options);

  // Post-traversal verification. The bottom level acceleration structure
  // build cannot be given zero clusters. If Traversal did not produce any
  // clusters, default to the first (which should be lowest detail anyway).
  // TODO: change traversal so this is not needed
  m_traverseVerify = vkobj::SimpleComputePipeline(allocator->getDevice(), glslCompiler, "traverse_verify.comp.glsl",
                                                  m_uboDescriptorSet.layout(), &options);

  m_instanceWriter = vkobj::SimpleComputePipeline<shaders::WriteInstancesConstant>(allocator->getDevice(), glslCompiler,
                                                                                   "write_instances.comp.glsl", &options);

  // TODO: move this computation to SceneCounts
  size_t totalNodes = 0;
  for(auto& mesh : scene.meshes)
  {
    totalNodes += mesh.hierarchy.nodes.size();
  }

  // Allocate only the node queue and job status buffer. The Cluster queue is
  // allocated just before use.
  m_nodeQueue = vkobj::Buffer<shaders::EncodedNodeJob>(allocator, totalNodes,
                                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_jobStatus = vkobj::Buffer<shaders::JobStatus>(allocator, 1, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  {
    vkobj::ImmediateCommandBuffer initCmd(allocator->getDevice(), initPool, initQueue);
    vkCmdFillBuffer(initCmd, m_nodeQueue, 0, m_nodeQueue.size_bytes(), 0);
    vkCmdFillBuffer(initCmd, m_clusterQueue, 0, m_clusterQueue.size_bytes(), 0);
  }

  // Perform an initial traversal of the scene. The output is needed to build
  // the initial TLAS before an update command buffer can be recorded.
  {
    vkobj::ImmediateCommandBuffer cmd(allocator->getDevice(), initPool, initQueue);
    shaders::TraversalParams      traversalParams = initialTraversalParams();
    memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    traverse(allocator, traversalParams, sceneVk, m_blas.input(), m_blas.inputPointers(), nullptr, {}, cmd);
  }

#if !defined(NDEBUG)
  // Old school printf debugging for indirect arguments
  BufferDownloader downloader(allocator->getDevice(), initQueueFamilyIndex, allocator->getStaging());
  rangeSummaryVk(std::cerr << "Meshes: ", sceneVk.meshPointers) << "\n";
  rangeSummaryVk(std::cerr << "BLAS Input: ", m_blas.input()) << "\n";
#endif

  {
    // Build the BLAS and create the intial TLAS
    vkobj::ImmediateCommandBuffer cmd(allocator->getDevice(), initPool, initQueue);
    m_blas.cmdBuild(cmd, m_blasAddresses);
    memoryBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    shaders::WriteInstancesConstant constant{
        .instances         = sceneVk.instances.address(),
        .meshBlasAddresses = m_blasAddresses.address(),
        .tlasInfos         = deviceReinterpretCast<shaders::InstanceInfo>(m_tlasInfos.address()),
        .instancesSize     = uint32_t(m_tlasInfos.size()),
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_instanceWriter);
    vkCmdPushConstants(cmd, m_instanceWriter.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constant), &constant);
    vkCmdDispatch(cmd, div_ceil(uint32_t(m_tlasInfos.size()), uint32_t(TRAVERSAL_WORKGROUP_SIZE)), 1, 1);
    memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

    // The TLAS must be created after m_tlasInfos has been populated with
    // real data. This is because it does an initial
    // VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR when created and we record a
    // VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR to resubmit each frame.
    m_tlas.emplace(allocator, m_tlasInfos, cmd);
  }
}

void LodMeshTraverser::traverseAndBuildBVH(ResourceAllocator*              allocator,
                                           const shaders::TraversalParams& traversalParams,
                                           const SceneVK&                  sceneVk,
                                           nvvk::ProfilerVK&               profiler,
                                           vkobj::SemaphoreValue&&         submitSemaphore,
                                           VkCommandBuffer                 cmd)
{
  {
    nvvk::ProfilerVK::Section sec = profiler.timeRecurring("Traverse Scene LOD", cmd);
    traverse(allocator, traversalParams, sceneVk, m_blas.input(), m_blas.inputPointers(), &profiler,
             vkobj::SemaphoreValue{submitSemaphore}, cmd);
  }

  {
    nvvk::ProfilerVK::Section outerSec = profiler.timeRecurring("Build BVH", cmd);
    // BLAS build
    {
      nvvk::ProfilerVK::Section sec = profiler.timeRecurring("BLAS", cmd);
      m_blas.cmdBuild(cmd, m_blasAddresses);
    }

    // BLAS build -> write instances
    memoryBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // write instances
    shaders::WriteInstancesConstant constant{
        .instances         = sceneVk.instances.address(),
        .meshBlasAddresses = m_blasAddresses.address(),
        .tlasInfos         = deviceReinterpretCast<shaders::InstanceInfo>(m_tlasInfos.address()),
        .instancesSize     = uint32_t(m_tlasInfos.size()),
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_instanceWriter);
    vkCmdPushConstants(cmd, m_instanceWriter.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constant), &constant);
    {
      nvvk::ProfilerVK::Section sec = profiler.timeRecurring("Write Instances", cmd);
      vkCmdDispatch(cmd, div_ceil(uint32_t(m_tlasInfos.size()), uint32_t(TRAVERSAL_WORKGROUP_SIZE)), 1, 1);
    }

    // write instances -> TLAS build
    memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

    // TLAS build
    {
      nvvk::ProfilerVK::Section sec = profiler.timeRecurring("TLAS", cmd);
      m_tlas->cmdUpdate(m_tlasInfos, cmd, false);
    }
  }
}

void LodMeshTraverser::traverse(ResourceAllocator*              allocator,
                                const shaders::TraversalParams& traversalParams,
                                const SceneVK&                  sceneVk,
                                const vkobj::Buffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>& blasInput,
                                const vkobj::Buffer<VkDeviceAddress>& blasInputClusters,
                                nvvk::ProfilerVK*                     profiler,
                                vkobj::SemaphoreValue&&               submitSemaphore,
                                VkCommandBuffer                       cmd)
{
  // Zero the job queue
  shaders::JobStatus initJobStatus{};
  vkCmdUpdateBuffer(cmd, m_jobStatus, 0, sizeof(shaders::JobStatus), &initJobStatus);

  // Fill the per-mesh k-nearest instance buffer with ones in preparation for a short bubble sort
  vkCmdFillBuffer(cmd, m_sortingMeshInstances, 0, m_sortingMeshInstances.size_bytes(), 0xffffffff);

  auto statsBuffer = m_stats.getFreeBuffer(allocator);
  vkCmdFillBuffer(cmd, statsBuffer.deviceBuffer(), 0, statsBuffer.deviceBuffer().size_bytes(), 0);

  // Barrier: buffer updates -> sort instances
  memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Find the closest k instances, that will be used during traversal
  shaders::SortInstancesConstant sortInstancesConstant{
      .traversalParams      = traversalParams,
      .instances            = sceneVk.instances.address(),
      .meshes               = sceneVk.meshPointers.address(),
      .sortingMeshInstances = m_sortingMeshInstances.address(),
      .instancesSize        = uint32_t(sceneVk.instances.size()),
  };
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverseSortInstances);
  vkCmdPushConstants(cmd, m_traverseSortInstances.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(sortInstancesConstant), &sortInstancesConstant);
  {
    OptionalSection sec(profiler, "Traverse Sort", cmd);
    vkCmdDispatch(cmd, div_ceil(uint32_t(sceneVk.instances.size()), uint32_t(TRAVERSAL_WORKGROUP_SIZE)), 1, 1);
  }

  // Write traversal parameters
  shaders::TraversalConstants traversalConstants{
      .traversalParams                       = traversalParams,
      .meshesAddress                         = sceneVk.meshPointers.address(),
      .instancesAddress                      = sceneVk.instances.address(),
      .nodeQueueAddress                      = m_nodeQueue.address(),
      .clusterQueueAddress                   = m_clusterQueue.address(),
      .jobStatusAddress                      = m_jobStatus.address(),
      .traverseStatsAddress                  = statsBuffer.deviceBuffer().address(),
      .blasInputAddress                      = deviceReinterpretCast<shaders::ClusterBLASInfoNV>(blasInput.address()),
      .blasInputClustersAddress              = blasInputClusters.address(),
      .drawClustersAddress                   = vkobj::DeviceAddress<shaders::DrawCluster>(0),
      .drawMeshTasksIndirectAddress          = vkobj::DeviceAddress<shaders::DrawMeshTasksIndirect>(0),
      .drawStatsAddress                      = vkobj::DeviceAddress<shaders::DrawStats>(0),
      .meshInstances                         = m_meshInstances.address(),
      .sortingMeshInstances                  = m_sortingMeshInstances.address(),
      .selectedClusters                      = vkobj::DeviceAddress<shaders::SelectedCluster>(0),
      .writeSelectedClustersDispatchIndirect = vkobj::DeviceAddress<shaders::DispatchIndirect>(0),
      .maxSelectedClusters                   = uint32_t(m_blas.maxTotalClusters()),
      .nodeQueueSize                         = uint32_t(m_nodeQueue.size()),
      .clusterQueueSize                      = uint32_t(m_clusterQueue.size()),
      .itemsSize                             = uint32_t(sceneVk.meshPointers.size()),
      .drawClustersSize                      = 0,
  };

  // Common to all shaders
  // Originally, this was designed to produce a re-submittable command buffer.
  // The traversal constants UBO could trivially be push constants instead.
  vkCmdUpdateBuffer(cmd, m_traversalConstants, 0, sizeof(traversalConstants), &traversalConstants);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverseInit.pipelineLayout, 0, 1,
                          &m_uboDescriptorSet.get(), 0, nullptr);

  // Barrier: sort instances -> traverse init
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: traverse init
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverseInit);
  {
    OptionalSection sec(profiler, "Traverse Init", cmd);
    vkCmdDispatch(cmd, div_ceil(uint32_t(sceneVk.meshPointers.size()), uint32_t(TRAVERSAL_WORKGROUP_SIZE)), 1, 1);
  }

  // Barrier: traverse init -> traverse
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: traverse
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverse);
  {
    OptionalSection sec(profiler, "Traverse Main", cmd);
    vkCmdDispatch(cmd, div_ceil(4096u, uint32_t(TRAVERSAL_WORKGROUP_SIZE)), 1, 1);
  }

  // Barrier: traverse -> traverse verify
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Run: traverse verify
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traverseVerify);
  {
    OptionalSection sec(profiler, "Traverse Verify", cmd);
    vkCmdDispatch(cmd, div_ceil(uint32_t(sceneVk.meshPointers.size()), uint32_t(TRAVERSAL_WORKGROUP_SIZE)), 1, 1);
  }

  // Barrier: traverse verify -> BLAS build
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

  statsBuffer.copyToHost(submitSemaphore, cmd);
  m_stats.queueBuffer(std::move(statsBuffer));
}
