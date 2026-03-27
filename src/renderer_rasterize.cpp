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

#include <dh_bindings.h>
#include <renderer_rasterize.hpp>
#include <sample_profiler.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>


RasterizeRenderer::RasterizeRenderer(const RenderInitParams& params)
    : m_streaming(std::make_unique<streaming::StreamingSceneVk>(
          params.context.instance.get(),
          params.context.device.get(),
          params.context.allocator.get(),
          params.context.staging,
          params.context.physicalDevice,
          params.queue,
          params.glslCompiler,
          params.streamingBufferSize,
          params.streamingMaxResidentGroups,
          params.initPool,
          params.initQueue,
          params.scene,
          params.sceneVk,
          params.streamingGreedyUnload,
          requiresCLAS(),
          params.transferQueue,
          params.profiler))
{
  auto& device    = params.context.device.get();
  auto& allocator = params.context.allocator.get();

  // key drawing data
  {
    m_drawingData.constants.emplace(device, 1, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);

    m_drawingData.drawClusters.emplace(device, size_t(1) << m_config.maxDrawableClusterBits,
                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                           | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);

    m_drawingData.drawIndirect.emplace(device, 1,
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                           | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                           | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);

    m_drawingData.drawStats.emplace(device, 1,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                        | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);

    m_drawingData.drawStatsHostVisible.emplace(
        device, MAX_CYCLES,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        allocator);
    m_drawingData.drawStatsMapping.emplace(m_drawingData.drawStatsHostVisible->map());
  }

  // key traversal data
  {
    m_traversalData.constants.emplace(device, 1, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);

    // Allocate only the node queue and job status buffer. The Cluster queue is
    // allocated just before use.
    m_traversalData.nodeQueue.emplace(device, params.sceneVk.counts.maxTotalInstanceNodes,
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                          | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
    m_traversalData.jobStatus.emplace(device, 1,
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                          | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);

    vkobj::ImmediateCommandBuffer initCmd(device, params.initPool, params.queue);
    device.vkCmdFillBuffer(initCmd, *m_traversalData.nodeQueue, 0,
                           m_traversalData.nodeQueue->sizeBytes(), 0);
  }

  initDrawingPipeline(device, params.glslCompiler, params.common, params.scene,
                      params.sceneVk, params.framebuffer);
  initTraversalPipeline(device, params.glslCompiler, params.framebuffer);
}

static uint32_t getBlockCount(uint32_t targetThreadCount, uint32_t blockSize)
{
  return (targetThreadCount + blockSize - 1) / blockSize;
}

void RasterizeRenderer::render(const RenderParams& params,
                               const SceneVK&      sceneVk,
                               vko::StagingStream<vko::vma::RecyclingStagingPool<vko::Device>>& staging)
{
  auto&           device        = params.context.device.get();
  auto&           cmd           = staging.commandBuffer();
  VkMemoryBarrier memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  //////////////////////////////////////////////////////////////////////////

  // handle resizes of scratch buffers due to ui settings / streaming

  resizeDrawingData(params);
  resizeTraversalData(params, sceneVk, cmd);

  //////////////////////////////////////////////////////////////////////////

  // setup basics

  shaders::RasterizeConstants rasterSceneInfo = {
      .instancesAddress    = sceneVk.instances,
      .meshesAddress       = sceneVk.meshPointers,
      .drawClustersAddress = (*m_drawingData.drawClusters),
      .drawStatsAddress    = (*m_drawingData.drawStats),
      .config              = m_config.shaders,
      .frame               = 0,
      .errorOverDistanceThreshold = params.common.m_traversalParams.errorOverDistanceThreshold,
  };

  device.vkCmdUpdateBuffer(cmd, (*m_drawingData.constants), 0,
                           sizeof(rasterSceneInfo), &rasterSceneInfo);

  // Zero draw indirect
  device.vkCmdFillBuffer(cmd, (*m_drawingData.drawIndirect), 0,
                         sizeof(shaders::DrawMeshTasksIndirect), 0);
  // Zero draw stats
  device.vkCmdFillBuffer(cmd, (*m_drawingData.drawStats), 0, sizeof(shaders::DrawStats), 0);

  // Execute the traversal shader
  shaders::TraversalConstants traversalConstants = {
      .traversalParams      = params.common.m_traversalParams,
      .meshesAddress        = sceneVk.meshPointers,
      .instancesAddress     = sceneVk.instances,
      .nodeQueueAddress     = (*m_traversalData.nodeQueue),
      .clusterQueueAddress  = (*m_traversalData.clusterQueue),
      .jobStatusAddress     = (*m_traversalData.jobStatus),
      .traverseStatsAddress = vkobj::DeviceAddress<shaders::TraverseStats>(0),
      .blasInputAddress = vkobj::DeviceAddress<shaders::ClusterBLASInfoNV>(0),  // not relevant to raster
      .blasInputClustersAddress = vkobj::DeviceAddress<uint64_t>(0),  // not relevant to raster
      .drawClustersAddress          = (*m_drawingData.drawClusters),
      .drawMeshTasksIndirectAddress = (*m_drawingData.drawIndirect),
      .drawStatsAddress             = (*m_drawingData.drawStats),
      .meshInstances = vkobj::DeviceAddress<shaders::MeshInstances>(0),
      .sortingMeshInstances = vkobj::DeviceAddress<shaders::SortingMeshInstances>(0),
      .selectedClusters = vkobj::DeviceAddress<shaders::SelectedCluster>(0),
      .writeSelectedClustersDispatchIndirect =
          vkobj::DeviceAddress<shaders::DispatchIndirect>(0),
      .maxSelectedClusters = uint32_t(0),
      .nodeQueueSize       = uint32_t(m_traversalData.nodeQueue->size()),
      .clusterQueueSize    = uint32_t(m_traversalData.clusterQueue->size()),
      .itemsSize           = uint32_t(sceneVk.instances.size()),
      .drawClustersSize    = uint32_t(m_drawingData.drawClusters->size()),
  };

  device.vkCmdUpdateBuffer(cmd, (*m_traversalData.constants), 0,
                           sizeof(traversalConstants), &traversalConstants);

  // Zero the job queue
  device.vkCmdFillBuffer(cmd, (*m_traversalData.jobStatus), 0,
                         sizeof(shaders::JobStatus), 0);


  //////////////////////////////////////////////////////////////////////////

  // first do traversal

  {
    ScopedGpuTimer sec(device, params.profiler, cmd, "Traverse Scene LOD");

    device.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   *m_traversal.pipelineLayout, 0, 1,
                                   m_traversal.descriptorSet->ptr(), 0, nullptr);

    memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                &memoryBarrier, 0, nullptr, 0, nullptr);

    // sets up node and job queue so that every instance enqueues its lod root node.
    // outputs:
    //  m_traversalData.jobStatus
    //  m_traversalData.nodeQueue

    device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             *m_traversal.traverseInitPipeline);
    device.vkCmdDispatch(cmd, getBlockCount(uint32_t(sceneVk.instances.size()), TRAVERSAL_WORKGROUP_SIZE),
                         1, 1);

    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                &memoryBarrier, 0, nullptr, 0, nullptr);

    // primary traversal over the instance's lod nodes.
    // reads:
    //   scene data: meshes, instances, cluster groups ... data
    // modifies:
    //  m_traversalData.jobStatus
    //  m_traversalData.nodeQueue
    //  m_traversalData.clusterQueue
    //
    //  state related to streaming:
    //    sceneVk.allGroupNeededFlags contains a big array over streaming info for all meshes
    //
    // outputs:
    //  m_traversalData.drawClusters
    //  m_traversalData.drawIndirect

    device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, *m_traversal.traversePipeline);
    device.vkCmdDispatch(cmd, getBlockCount(4096u, TRAVERSAL_WORKGROUP_SIZE), 1, 1);

    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                &memoryBarrier, 0, nullptr, 0, nullptr);

    // sets up the indirect draw call
    // also clamps the number of generated draw calls to stay within limit
    // modifies:
    //  m_traversalData.drawIndirect
    //  m_traversalData.drawStats

    device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             *m_traversal.traverseVerifyPipeline);
    device.vkCmdDispatch(cmd, 1, 1, 1);

    // graphics must wait
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV
                                    | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
  }

  //////////////////////////////////////////////////////////////////////////

  // then draw

  {
    ScopedGpuTimer sec(device, params.profiler, cmd, "Rasterize");

    VkClearValue colorClear{.color = {{0.0F, 0.0F, 0.0F, 1.0F}}};
    VkClearValue depthClear{.depthStencil = {1.0F, 0}};

    VkRenderingAttachmentInfo colorAttachment = {
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = params.framebuffer.renderHdrView(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = colorClear,
    };

    VkRenderingAttachmentInfo depthStencilAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = params.framebuffer.depthView(),
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = depthClear,
    };

    // Dynamic rendering information: color and depth attachments
    VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT | VK_RENDERING_CONTENTS_INLINE_BIT_KHR,
        .renderArea           = {{0, 0}, params.framebuffer.size()},
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
        .pDepthAttachment     = &depthStencilAttachment,
    };

    device.vkCmdBeginRendering(cmd, &renderingInfo);

    VkExtent2D imageSize = params.framebuffer.size();

    VkViewport vp;
    vp.minDepth = 0;
    vp.maxDepth = 1.0f;
    vp.x        = 0;
    vp.y        = 0;
    vp.width    = float(imageSize.width);
    vp.height   = float(imageSize.height);

    VkRect2D scissor;
    scissor.extent = imageSize;
    scissor.offset = {0, 0};

    device.vkCmdSetViewport(cmd, 0, 1, &vp);
    device.vkCmdSetScissor(cmd, 0, 1, &scissor);

    device.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, *m_drawing.pipeline);
    device.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                   *m_drawing.pipelineLayout, 0, 1,
                                   m_drawing.descriptorSet->ptr(), 0, nullptr);


    // reads:
    //  m_traversalData.drawClusters
    //  m_traversalData.drawIndirect
    //  (as well as instances, meshes... data )

    device.vkCmdDrawMeshTasksIndirectNV(cmd, (*m_drawingData.drawIndirect), 0, 1,
                                        uint32_t(sizeof(shaders::DrawMeshTasksIndirect)));

    device.vkCmdEndRendering(cmd);
  }

  {
    ScopedGpuTimer sec(device, params.profiler, cmd, "Read & HiZ");

    // barrier for copying stats to host
    VkBufferMemoryBarrier bufferBarrier = {};
    bufferBarrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.buffer        = (*m_drawingData.drawStats);
    bufferBarrier.size          = m_drawingData.drawStats->sizeBytes();
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    // barrier for hiz generation
    VkImageMemoryBarrier imageBarrier = {};
    imageBarrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.image                = params.framebuffer.depthImage();
    imageBarrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    imageBarrier.subresourceRange.baseMipLevel   = 0;
    imageBarrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

    imageBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    device.vkCmdPipelineBarrier(
        cmd, VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &bufferBarrier, 1, &imageBarrier);

    // copy back statistics to host visible buffer
    {
      VkBufferCopy copy;
      copy.dstOffset = sizeof(shaders::DrawStats) * (m_frame % MAX_CYCLES);
      copy.size      = sizeof(shaders::DrawStats);
      copy.srcOffset = 0;

      device.vkCmdCopyBuffer(cmd, *m_drawingData.drawStats,
                             *m_drawingData.drawStatsHostVisible, 1, &copy);
    }

    // update hiz
    if(!params.common.m_frameIndex || !params.common.m_config->lockLodCamera)
    {
      params.framebuffer.hiz().cmdUpdateHiz(cmd, params.framebuffer.hizUpdate(),
                                            uint32_t(0));
    }

    // barrier to transition depth image back to rendering state

    imageBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    imageBarrier.oldLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                                    | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                                0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
  }

  m_frame++;
}

void RasterizeRenderer::resizeTraversalData(const RenderParams& params,
                                            const SceneVK&      sceneVk,
                                            VkCommandBuffer     cmd)
{
  if(!m_traversalData.clusterQueue
     || sceneVk.totalResidentInstanceClusters < m_traversalData.clusterQueue->size() / 3
     || sceneVk.totalResidentInstanceClusters > m_traversalData.clusterQueue->size())
  {
    size_t newClusterQueueSize = (sceneVk.totalResidentInstanceClusters * 3) / 2;
    printf("Reallocating traversal cluster queue: %zu\n", newClusterQueueSize);

    // nextSubmitSemaphore() will be signalled after rendering the current frame
    // Only push to garbage if the optional is already initialized
    if(m_traversalData.clusterQueue)
    {
      params.garbage.push(
          Garbage{moveAny(std::move((*m_traversalData.clusterQueue))),
                  params.context.staging.commandBuffer().nextSubmitSemaphore()});
    }

    m_traversalData.clusterQueue.emplace(
        params.context.device.get(), newClusterQueueSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, params.context.allocator.get());
    params.context.device.get().vkCmdFillBuffer(cmd, *m_traversalData.clusterQueue, 0,
                                                m_traversalData.clusterQueue->sizeBytes(), 0);
  }
}

void RasterizeRenderer::resizeDrawingData(const RenderParams& params)
{
  size_t newSize = size_t(1) << m_config.maxDrawableClusterBits;

  if(newSize != m_drawingData.drawClusters->size())
  {
    // nextSubmitSemaphore() will be signalled after rendering the current frame
    params.garbage.push(
        Garbage{moveAny(std::move((*m_drawingData.drawClusters))),
                params.context.staging.commandBuffer().nextSubmitSemaphore()});

    m_drawingData.drawClusters.emplace(params.context.device.get(), newSize,
                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                           | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                       params.context.allocator.get());
  }
}

void RasterizeRenderer::updatedFrambuffer(const RenderParams& params)
{
  vko::WriteDescriptorSetBuilder builder;
  builder.push_back<VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER>(
      *m_traversal.descriptorSet, m_traversal.bindings->bindings[shaders::BTraversalHiZTex],
      0, params.framebuffer.hizFar());
  params.context.device.get().vkUpdateDescriptorSets(
      params.context.device.get(), static_cast<uint32_t>(builder.writes().size()),
      builder.writes().data(), 0, nullptr);
}

void RasterizeRenderer::initDrawingPipeline(const vko::Device&    device,
                                            SampleGlslCompiler&   glslCompiler,
                                            const RendererCommon& common,
                                            const Scene&          scene,
                                            const SceneVK&        sceneVk,
                                            [[maybe_unused]] Framebuffer& framebuffer)
{
  // rasterization shaders

  shaderc::CompileOptions options = glslCompiler.defaultOptions();
  options.AddMacroDefinition("CLUSTER_VERTEX_COUNT",
                             std::to_string(scene.counts.maxClusterVertexCount));
  options.AddMacroDefinition("CLUSTER_TRIANGLE_COUNT",
                             std::to_string(scene.counts.maxClusterTriangleCount));

  vkobj::ShaderModule meshModule =
      reloadUntilCompiling(device, glslCompiler, "rasterize.mesh.glsl",
                           shaderc_shader_kind::shaderc_glsl_mesh_shader, &options);

  vkobj::ShaderModule fragmentModule =
      reloadUntilCompiling(device, glslCompiler, "rasterize.frag.glsl",
                           shaderc_shader_kind::shaderc_glsl_fragment_shader, &options);

  // descriptor set / binding

  {
    VkShaderStageFlags shaderStageFlags =
        VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_MESH_BIT_NV;

    m_drawing.bindings.emplace(vko::makeBindings({
        {{shaders::BRasterFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, shaderStageFlags, nullptr}, 0},
        {{shaders::BRasterConstants, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, shaderStageFlags, nullptr}, 0},
        {{shaders::BRasterSkyParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, shaderStageFlags, nullptr}, 0},
        {{shaders::BRasterTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          std::max(1u, uint32_t(sceneVk.textureDescriptors.size())), shaderStageFlags, nullptr},
         0},
    }));
    m_drawing.descriptorSetLayout.emplace(
        vko::makeDescriptorSetLayout(device, *m_drawing.bindings, 0));
    VkDescriptorSetLayout layoutsArray[] = {*m_drawing.descriptorSetLayout};
    m_drawing.pipelineLayout.emplace(
        device, VkPipelineLayoutCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                           .pNext          = nullptr,
                                           .flags          = 0,
                                           .setLayoutCount = 1,
                                           .pSetLayouts    = layoutsArray,
                                           .pushConstantRangeCount = 0,
                                           .pPushConstantRanges    = nullptr});
    m_drawing.descriptorPool.emplace(device, m_drawing.bindings->bindings,
                                     VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);
    m_drawing.descriptorSet.emplace(device, nullptr, *m_drawing.descriptorPool,
                                    *m_drawing.descriptorSetLayout);

    VkDescriptorBufferInfo frameInfoDesc{common.m_bFrameInfo, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo constantsDesc{(*m_drawingData.constants), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo skyParamsDesc{common.m_bSkyParams, 0, VK_WHOLE_SIZE};

    vko::WriteDescriptorSetBuilder builder;
    builder.push_back<VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER>(
        *m_drawing.descriptorSet,
        m_drawing.bindings->bindings[shaders::BRasterFrameInfo], 0, frameInfoDesc);
    builder.push_back<VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER>(
        *m_drawing.descriptorSet,
        m_drawing.bindings->bindings[shaders::BRasterConstants], 0, constantsDesc);
    builder.push_back<VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER>(
        *m_drawing.descriptorSet,
        m_drawing.bindings->bindings[shaders::BRasterSkyParams], 0, skyParamsDesc);
    if(!sceneVk.textureDescriptors.empty())
      builder.push_back<VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER>(
          *m_drawing.descriptorSet, m_drawing.bindings->bindings[shaders::BRasterTextures],
          0, sceneVk.textureDescriptors);
    device.vkUpdateDescriptorSets(device, static_cast<uint32_t>(builder.writes().size()),
                                  builder.writes().data(), 0, nullptr);
  }

  // pipeline

  {
    VkPipelineVertexInputStateCreateInfo vertexInputState{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineRasterizationStateCreateInfo rasterizationState{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizationState.cullMode  = VK_CULL_MODE_BACK_BIT;
    rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationState.lineWidth = 1.0f;

    VkPipelineDepthStencilStateCreateInfo depthStencilState{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencilState.depthTestEnable       = VK_TRUE;
    depthStencilState.depthWriteEnable      = VK_TRUE;
    depthStencilState.depthCompareOp        = VK_COMPARE_OP_LESS;
    depthStencilState.depthBoundsTestEnable = VK_FALSE;
    depthStencilState.stencilTestEnable     = VK_FALSE;
    depthStencilState.minDepthBounds        = 0.0f;
    depthStencilState.maxDepthBounds        = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampleState{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
        | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlendState{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlendState.attachmentCount = 1;
    colorBlendState.pAttachments    = &colorBlendAttachment;

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates    = dynamicStates;

    VkPipelineViewportStateCreateInfo viewportState{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkFormat colorFormat = Framebuffer::c_colorFormat;
    VkFormat depthFormat = Framebuffer::s_depthFormat;

    VkPipelineRenderingCreateInfo pipelineRendering{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    pipelineRendering.colorAttachmentCount    = 1;
    pipelineRendering.pColorAttachmentFormats = &colorFormat;
    pipelineRendering.depthAttachmentFormat   = depthFormat;

    VkPipelineShaderStageCreateInfo stages[] = {
        {.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext  = nullptr,
         .flags  = 0,
         .stage  = VK_SHADER_STAGE_MESH_BIT_NV,
         .module = meshModule,
         .pName  = "main",
         .pSpecializationInfo = nullptr},
        {.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext  = nullptr,
         .flags  = 0,
         .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fragmentModule,
         .pName  = "main",
         .pSpecializationInfo = nullptr},
    };

    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = &pipelineRendering,
        .flags               = 0,
        .stageCount          = 2,
        .pStages             = stages,
        .pVertexInputState   = &vertexInputState,
        .pInputAssemblyState = &inputAssemblyState,
        .pTessellationState  = nullptr,
        .pViewportState      = &viewportState,
        .pRasterizationState = &rasterizationState,
        .pMultisampleState   = &multisampleState,
        .pDepthStencilState  = &depthStencilState,
        .pColorBlendState    = &colorBlendState,
        .pDynamicState       = &dynamicState,
        .layout              = *m_drawing.pipelineLayout,
        .renderPass          = VK_NULL_HANDLE,
        .subpass             = 0,
        .basePipelineHandle  = VK_NULL_HANDLE,
        .basePipelineIndex   = 0,
    };
    m_drawing.pipeline.emplace(device, pipelineInfo);
  }
}

void RasterizeRenderer::initTraversalPipeline(const vko::Device&  device,
                                              SampleGlslCompiler& glslCompiler,
                                              Framebuffer&        framebuffer)
{
  // traversal shaders

  shaderc::CompileOptions options = glslCompiler.defaultOptions();
  options.AddMacroDefinition("IS_RASTERIZATION", "1");
  options.AddMacroDefinition("TRAVERSE_PER_INSTANCE", "1");

  vkobj::ShaderModule traverseModule =
      reloadUntilCompiling(device, glslCompiler, "traverse.comp.glsl",
                           shaderc_shader_kind::shaderc_glsl_compute_shader, &options);
  vkobj::ShaderModule traverseInitModule =
      reloadUntilCompiling(device, glslCompiler, "traverse_init.comp.glsl",
                           shaderc_shader_kind::shaderc_glsl_compute_shader, &options);
  vkobj::ShaderModule traverseVerifyModule =
      reloadUntilCompiling(device, glslCompiler, "traverse_verify.comp.glsl",
                           shaderc_shader_kind::shaderc_glsl_compute_shader, &options);

  // binding
  {
    m_traversal.bindings.emplace(vko::makeBindings({
        {{shaders::BTraversalConstants, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
          VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
         0},
        {{shaders::BTraversalHiZTex, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
         0},
    }));
    m_traversal.descriptorSetLayout.emplace(
        vko::makeDescriptorSetLayout(device, *m_traversal.bindings, 0));
    VkDescriptorSetLayout traversalLayoutsArray[] = {*m_traversal.descriptorSetLayout};
    m_traversal.pipelineLayout.emplace(
        device, VkPipelineLayoutCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                           .pNext          = nullptr,
                                           .flags          = 0,
                                           .setLayoutCount = 1,
                                           .pSetLayouts = traversalLayoutsArray,
                                           .pushConstantRangeCount = 0,
                                           .pPushConstantRanges    = nullptr});
    m_traversal.descriptorPool.emplace(device, m_traversal.bindings->bindings,
                                       VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);
    m_traversal.descriptorSet.emplace(device, nullptr, *m_traversal.descriptorPool,
                                      *m_traversal.descriptorSetLayout);

    VkDescriptorBufferInfo constantsDesc{(*m_traversalData.constants), 0, VK_WHOLE_SIZE};

    vko::WriteDescriptorSetBuilder builder;
    builder.push_back<VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER>(
        *m_traversal.descriptorSet,
        m_traversal.bindings->bindings[shaders::BTraversalConstants], 0, constantsDesc);
    builder.push_back<VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER>(
        *m_traversal.descriptorSet, m_traversal.bindings->bindings[shaders::BTraversalHiZTex],
        0, framebuffer.hizFar());
    device.vkUpdateDescriptorSets(device, static_cast<uint32_t>(builder.writes().size()),
                                  builder.writes().data(), 0, nullptr);
  }

  // pipeline
  {
    VkComputePipelineCreateInfo computeCreateInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};

    computeCreateInfo.stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };

    computeCreateInfo.layout = *m_traversal.pipelineLayout;

    computeCreateInfo.stage.module = traverseModule;
    m_traversal.traversePipeline.emplace(device, computeCreateInfo);

    computeCreateInfo.stage.module = traverseInitModule;
    m_traversal.traverseInitPipeline.emplace(device, computeCreateInfo);

    computeCreateInfo.stage.module = traverseVerifyModule;
    m_traversal.traverseVerifyPipeline.emplace(device, computeCreateInfo);
  }
}

void RasterizeRenderer::uiOverlay()
{
  ImGui::Text("Triangles: %s",
              formatThousands(
                  m_drawingData.drawStatsMapping->span()[m_frame % MAX_CYCLES].triangleCount)
                  .c_str());
}

void RasterizeRenderer::uiSection(bool&, bool&)
{

  ImGui::Text("Rasterization");
  int tempBits = int(m_config.maxDrawableClusterBits);
  if(ImGui::SliderInt("Draw Clusters (bits)", &tempBits, 0, 28))
    m_config.maxDrawableClusterBits = uint32_t(tempBits);

  shaders::DrawStats drawStats =
      m_frame > MAX_CYCLES ?
          m_drawingData.drawStatsMapping->span()[(m_frame + 1) % MAX_CYCLES] :
          shaders::DrawStats{};

  ImGui::Text("Max Draw Clusters %d", uint32_t(m_drawingData.drawClusters->size()));
  ImGui::Text("Requested Draw Clusters %d (%f pct)", drawStats.requestedClusterCount,
              double(drawStats.requestedClusterCount) * 100.0
                  / double(m_drawingData.drawClusters->size()));
  ImGui::Text("Rastered Triangles %d", drawStats.triangleCount);

#if 0
  static bool asFloat = false;
  ImGui::Checkbox("as float", &asFloat);
  for(uint32_t i = 0; i < 128; i++)
  {
    if(asFloat)
      ImGui::Text("Debug %2d - %f", i, *((float*)&drawStats.debug[i]));
    else
      ImGui::Text("Debug %2d - %10d", i, drawStats.debug[i]);
  }
#endif
}
