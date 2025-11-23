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
#include <renderer_raytrace.hpp>
#include <sample_profiler.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>

GBufferRT::GBufferRT(const vko::Device&        device,
                     vko::vma::Allocator&      allocator,
                     VkCommandBuffer           cmd,
                     glm::uvec2                size,
                     std::span<const VkFormat> colorFormats)
    : m_size(size)
    , m_colorFormats(colorFormats.begin(), colorFormats.end())
{
  m_colorImages.reserve(colorFormats.size());
  for(VkFormat format : colorFormats)
  {
    m_colorImages.emplace_back(device,
                               VkImageCreateInfo{
                                   .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                   .pNext = nullptr,
                                   .flags = 0,
                                   .imageType   = VK_IMAGE_TYPE_2D,
                                   .format      = format,
                                   .extent      = {size.x, size.y, 1},
                                   .mipLevels   = 1,
                                   .arrayLayers = 1,
                                   .samples     = VK_SAMPLE_COUNT_1_BIT,
                                   .tiling      = VK_IMAGE_TILING_OPTIMAL,
                                   .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                            | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                   .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                   .queueFamilyIndexCount = 0,
                                   .pQueueFamilyIndices   = nullptr,
                                   .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                               },
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);
  }

  // Transition all images from UNDEFINED to GENERAL
  vko::ImageAccess undefined{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_IMAGE_LAYOUT_UNDEFINED};
  vko::ImageAccess general{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                           VK_IMAGE_LAYOUT_GENERAL};
  for(const auto& image : m_colorImages)
  {
    vko::cmdImageBarrier(device, cmd, image.image, undefined, general);
  }
}

// #DLSS_RR
// halton low discrepancy sequence, from https://www.shadertoy.com/view/wdXSW8
inline glm::vec2 halton(int index)
{
  const glm::vec2 coprimes = glm::vec2(2.0F, 3.0F);
  glm::vec2       s        = glm::vec2(index, index);
  glm::vec4       a        = glm::vec4(1, 1, 0, 0);
  while(s.x > 0. && s.y > 0.)
  {
    a.x = a.x / coprimes.x;
    a.y = a.y / coprimes.y;
    a.z += a.x * fmod(s.x, coprimes.x);
    a.w += a.y * fmod(s.y, coprimes.y);
    s.x = floorf(s.x / coprimes.x);
    s.y = floorf(s.y / coprimes.y);
  }
  return glm::vec2(a.z, a.w);
}

PathtracingPipeline::PathtracingPipeline(SampleGlslCompiler&  glslCompiler,
                                         const vko::Instance& instance,
                                         VkPhysicalDevice     physicalDevice,
                                         const vko::Device&   device,
                                         vko::vma::Allocator& allocator,
                                         VkCommandPool        commandPool,
                                         VkQueue              queue,
                                         uint32_t             textureCount)
    : m_device(&device)
    , m_bindings(vko::makeBindings({
          {{BRtTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL, nullptr}, 0},
          {{BRtOutBaseColor_Metalness, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr}, 0},
          {{BRtOutSpecAlbedo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr}, 0},
          {{BRtOutSpecHitDist, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr}, 0},
          {{BRtOutNormalRoughness, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr}, 0},
          {{BRtOutMotionVectors, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr}, 0},
          {{BRtOutViewZ, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr}, 0},
          {{BRtOutColor, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr}, 0},
          {{BRtFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr}, 0},
          {{BRtSkyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr}, 0},
          {{BRtTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            std::max(textureCount, 1u), VK_SHADER_STAGE_ALL, nullptr},
           0},
      }))
    , m_descriptorSetLayout(vko::makeDescriptorSetLayout(device, m_bindings, 0))
{
  vkobj::ShaderModule shaderRaygen =
      reloadUntilCompiling(device, glslCompiler, "pathtrace.rgen.glsl",
                           shaderc_shader_kind::shaderc_glsl_raygen_shader);
  vkobj::ShaderModule shaderMiss =
      reloadUntilCompiling(device, glslCompiler, "pathtrace.rmiss.glsl",
                           shaderc_shader_kind::shaderc_glsl_miss_shader);
  vkobj::ShaderModule shaderClosestHit =
      reloadUntilCompiling(device, glslCompiler, "pathtrace.rchit.glsl",
                           shaderc_shader_kind::shaderc_glsl_closesthit_shader);
  std::vector<VkPipelineShaderStageCreateInfo> shaderStages{
      {
          .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .pNext  = nullptr,
          .flags  = 0,
          .stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
          .module = shaderRaygen,
          .pName  = "main",
          .pSpecializationInfo = nullptr,
      },
      {
          .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .pNext  = nullptr,
          .flags  = 0,
          .stage  = VK_SHADER_STAGE_MISS_BIT_KHR,
          .module = shaderMiss,
          .pName  = "main",
          .pSpecializationInfo = nullptr,
      },
      {
          .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .pNext  = nullptr,
          .flags  = 0,
          .stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
          .module = shaderClosestHit,
          .pName  = "main",
          .pSpecializationInfo = nullptr,
      },
  };
  for([[maybe_unused]] auto& stage : shaderStages)
    assert(stage.module != VK_NULL_HANDLE);

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shadingGroups{
      {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .pNext = nullptr,
       .type  = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
       .generalShader                   = 0 /* "raygen" shaderStages[0] */,
       .closestHitShader                = VK_SHADER_UNUSED_KHR,
       .anyHitShader                    = VK_SHADER_UNUSED_KHR,
       .intersectionShader              = VK_SHADER_UNUSED_KHR,
       .pShaderGroupCaptureReplayHandle = nullptr},
      {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .pNext = nullptr,
       .type  = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
       .generalShader                   = 1 /* "miss" shaderStages[1] */,
       .closestHitShader                = VK_SHADER_UNUSED_KHR,
       .anyHitShader                    = VK_SHADER_UNUSED_KHR,
       .intersectionShader              = VK_SHADER_UNUSED_KHR,
       .pShaderGroupCaptureReplayHandle = nullptr},
      {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .pNext = nullptr,
       .type  = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
       .generalShader                   = VK_SHADER_UNUSED_KHR,
       .closestHitShader                = 2 /* "closest hit" shaderStages[2] */,
       .anyHitShader                    = VK_SHADER_UNUSED_KHR,
       .intersectionShader              = VK_SHADER_UNUSED_KHR,
       .pShaderGroupCaptureReplayHandle = nullptr}};

  // Create pipeline layout
  VkPushConstantRange        pushConstantRange{VK_SHADER_STAGE_ALL, 0,
                                        sizeof(shaders::PathtraceConstant)};
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext                  = nullptr,
      .flags                  = 0,
      .setLayoutCount         = 1,
      .pSetLayouts            = m_descriptorSetLayout.ptr(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstantRange,
  };
  m_pipelineLayout.emplace(device, pipelineLayoutInfo);

  // Enable cluster acceleration structures in the pipeline
  VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV pipelineClusterAccelerationStructureCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
      .pNext                             = nullptr,
      .allowClusterAccelerationStructure = true};

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo{
      .sType      = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .pNext      = &pipelineClusterAccelerationStructureCreateInfo,
      .flags      = 0,
      .stageCount = static_cast<uint32_t>(shaderStages.size()),
      .pStages    = shaderStages.data(),
      .groupCount = static_cast<uint32_t>(shadingGroups.size()),
      .pGroups    = shadingGroups.data(),
      .maxPipelineRayRecursionDepth = PATHTRACE_MAX_VK_RECURSION_DEPTH,
      .pLibraryInfo                 = nullptr,
      .pLibraryInterface            = nullptr,
      .pDynamicState                = nullptr,
      .layout                       = *m_pipelineLayout,
      .basePipelineHandle           = VK_NULL_HANDLE,
      .basePipelineIndex            = 0,
  };
  m_pipeline.emplace(device, pipelineCreateInfo);

  // Requesting ray tracing properties
  // Get ray tracing pipeline properties
  auto rtPipelineProperties =
      vko::simple::rayTracingPipelineProperties(instance, physicalDevice);

  // Create Shader Binding Table (SBT)
  vko::simple::HitGroupHandles handles(device, rtPipelineProperties, *m_pipeline,
                                       pipelineCreateInfo.groupCount);

  // Build the staging tables with handles for raygen, miss, hit groups
  std::vector<std::span<const std::byte>> raygenHandles;
  std::vector<std::span<const std::byte>> missHandles;
  std::vector<std::span<const std::byte>> hitHandles;
  std::vector<std::span<const std::byte>> callableHandles;  // empty for now

  // Group indices based on the shader groups created above
  raygenHandles.push_back(handles[0]);  // raygen group
  missHandles.push_back(handles[1]);    // miss group
  hitHandles.push_back(handles[2]);     // hit group

  vko::simple::ShaderBindingTablesStaging staging(allocator, device, rtPipelineProperties,
                                                  raygenHandles, missHandles,
                                                  hitHandles, callableHandles);

  // Create the final SBT by copying staging to device-local memory
  m_sbt = std::make_unique<vko::simple::ShaderBindingTables<vko::vma::Allocator>>(
      device, commandPool, queue, std::move(staging), allocator);
}

void PathtracingPipeline::writeDescriptorSetInitial(VkAccelerationStructureKHR tlas,
                                                    VkBuffer frameInfo,
                                                    VkBuffer skyParams,
                                                    std::span<const VkDescriptorImageInfo> textures,
                                                    VkDescriptorSet descriptorSet) const
{
  const auto&            bindings = m_bindings.bindings;
  VkDescriptorBufferInfo frameInfoDesc{frameInfo, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo skyParamsDesc{skyParams, 0, VK_WHOLE_SIZE};

  vko::WriteDescriptorSetBuilder builder;
  builder.push_back<VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR>(
      descriptorSet, bindings[BRtTlas], 0, tlas);
  builder.push_back<VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER>(descriptorSet, bindings[BRtFrameInfo],
                                                       0, frameInfoDesc);
  builder.push_back<VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER>(descriptorSet, bindings[BRtSkyParam],
                                                       0, skyParamsDesc);
  if(!textures.empty())
    builder.push_back<VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER>(
        descriptorSet, bindings[BRtTextures], 0, textures);

  m_device->vkUpdateDescriptorSets(*m_device,
                                   static_cast<uint32_t>(builder.writes().size()),
                                   builder.writes().data(), 0, nullptr);
}

void PathtracingPipeline::writeDescriptorSetFramebuffer(const GBufferRT& gBuffer,
                                                        std::optional<VkDescriptorImageInfo> passthroughColor,
                                                        VkDescriptorSet descriptorSet) const
{
  const auto&                    bindings = m_bindings.bindings;
  vko::WriteDescriptorSetBuilder builder;
  builder.push_back<VK_DESCRIPTOR_TYPE_STORAGE_IMAGE>(
      descriptorSet, bindings[BRtOutBaseColor_Metalness], 0,
      gBuffer.getDescriptorImageInfo(eGBufBaseColor_Metalness));
  builder.push_back<VK_DESCRIPTOR_TYPE_STORAGE_IMAGE>(
      descriptorSet, bindings[BRtOutSpecAlbedo], 0,
      gBuffer.getDescriptorImageInfo(eGBufSpecAlbedo));
  builder.push_back<VK_DESCRIPTOR_TYPE_STORAGE_IMAGE>(
      descriptorSet, bindings[BRtOutSpecHitDist], 0,
      gBuffer.getDescriptorImageInfo(eGBufSpecHitDist));
  builder.push_back<VK_DESCRIPTOR_TYPE_STORAGE_IMAGE>(
      descriptorSet, bindings[BRtOutNormalRoughness], 0,
      gBuffer.getDescriptorImageInfo(eGBufNormalRoughness));
  builder.push_back<VK_DESCRIPTOR_TYPE_STORAGE_IMAGE>(
      descriptorSet, bindings[BRtOutMotionVectors], 0,
      gBuffer.getDescriptorImageInfo(eGBufMotionVectors));
  builder.push_back<VK_DESCRIPTOR_TYPE_STORAGE_IMAGE>(
      descriptorSet, bindings[BRtOutViewZ], 0, gBuffer.getDescriptorImageInfo(eGBufViewZ));
  builder.push_back<VK_DESCRIPTOR_TYPE_STORAGE_IMAGE>(
      descriptorSet, bindings[BRtOutColor], 0,
      passthroughColor ? *passthroughColor : gBuffer.getDescriptorImageInfo(eGBufColor));

  m_device->vkUpdateDescriptorSets(*m_device,
                                   static_cast<uint32_t>(builder.writes().size()),
                                   builder.writes().data(), 0, nullptr);
}

void PathtracingPipeline::trace(VkCommandBuffer              cmd,
                                std::vector<VkDescriptorSet> descriptorSets,
                                const shaders::PathtraceConstant& pushConstant,
                                glm::uvec2                        vpSize) const
{
  m_device->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, *m_pipeline);
  m_device->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                    *m_pipelineLayout, DSRt,
                                    static_cast<uint32_t>(descriptorSets.size()),
                                    descriptorSets.data(), 0, nullptr);
  m_device->vkCmdPushConstants(cmd, *m_pipelineLayout, VK_SHADER_STAGE_ALL, 0,
                               sizeof(pushConstant), &pushConstant);
  m_device->vkCmdTraceRaysKHR(cmd, &m_sbt->raygenTableOffset,
                              &m_sbt->missTableOffset, &m_sbt->hitTableOffset,
                              &m_sbt->callableTableOffset, vpSize.x, vpSize.y, 1);
  memoryBarrier(*m_device, cmd, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

RaytraceRenderer::RaytraceRenderer(const RenderInitParams&         params,
                                   vko::shared_obj<RaytraceConfig> config)
    : m_streaming(std::make_unique<streaming::StreamingSceneVk>(
          params.context.instance,
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
          params.profiler
        ))
    , m_config(std::move(config))
    , m_rtPipeline(params.glslCompiler,
                   params.context.instance,
                   params.context.physicalDevice,
                   params.context.device.get(),
                   params.context.allocator.get(),
                   params.context.commandPool,
                   params.context.queue.get(),
                   uint32_t(params.sceneVk.textures.size()))
    , m_descriptorPool(params.context.device.get(),
                       m_rtPipeline.bindings().bindings,
                       VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT)
    , m_rtDescriptorSet(params.context.device.get(),
                        nullptr,
                        m_descriptorPool,
                        m_rtPipeline.descriptorSetLayout())
    , m_sceneCounts(params.scene.counts)
    , m_sceneAabb(params.scene.worldAABB)
    , m_ngxParameter(vko::ngx::CapabilityParameter::null())
{
  // Initialize DLSS-RR NGX parameters if available
  if(params.dlssAvailable)
  {
    m_ngxParameter = vko::ngx::CapabilityParameter();
    try
    {
      auto preset = NVSDK_NGX_DLSS_Hint_Render_Preset_Default;
      m_ngxParameter->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
                          preset);
      m_ngxParameter->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality,
                          preset);
      m_ngxParameter->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
                          preset);
      m_ngxParameter->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance,
                          preset);
      m_ngxParameter->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance,
                          preset);
      vko::ngx::assertRayReconstructionSupported(*m_ngxParameter);
    }
    catch(const std::exception& e)
    {
      std::cout << "DLSS-RR parameter initialization failed (will continue without DLSS): "
                << e.what() << std::endl;
      m_ngxParameter.reset();  // Clear the parameter on failure
    }
  }
  VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
  if(m_config->perInstanceTraversal)
  {
    m_lodInstanceTraverser =
        LodInstanceTraverser{params.context.device.get(),
                             params.context.allocator.get(),
                             params.context.staging,
                             params.context.queue.get(),
                             params.glslCompiler,
                             params.context.commandPool,
                             params.context.queue.get(),
                             params.context.queueFamilyIndex,
                             params.scene,
                             params.sceneVk,
                             uint32_t(float(params.streamingMaxResidentGroups)
                                      * m_config->memoryReserveScale)};
    tlas = m_lodInstanceTraverser->tlas();
  }
  else
  {
    m_lodMeshTraverser = LodMeshTraverser{params.context.device.get(),
                                          params.context.allocator.get(),
                                          params.context.staging,
                                          params.context.queue.get(),
                                          params.glslCompiler,
                                          params.context.commandPool,
                                          params.context.queue.get(),
                                          params.context.queueFamilyIndex,
                                          params.scene,
                                          params.sceneVk,
                                          params.streamingMaxResidentGroups};
    tlas               = m_lodMeshTraverser->tlas();
  }
  m_rtPipeline.writeDescriptorSetInitial(tlas, params.common.m_bFrameInfo,
                                         params.common.m_bSkyParams,
                                         params.sceneVk.textureDescriptors,
                                         m_rtDescriptorSet);
}

void RaytraceRenderer::updatedFrambuffer(const RenderParams&)
{
  // Mark the gBuffer as stale. It's more convenient to handle updates in
  // render()
  m_gBufferStale = true;
}

void RaytraceRenderer::render(const RenderParams& params,
                              const SceneVK&      sceneVk,
                              vko::StagingStream<vko::vma::RecyclingStagingPool<vko::Device>>& staging)
{
  auto& cmd = staging.commandBuffer();
  // Check if the framebuffer is stale. This can happen on first launch or if
  // the window is resized.
  if(m_gBufferStale)
  {
    VkExtent2D gbufferSize = params.framebuffer.size();

    // Recreate DLSS-RR with the new framebuffer size
    if(m_dlssRR)
    {
      params.garbage.push(Garbage{moveAny(std::move(*m_dlssRR)),
                                  staging.commandBuffer().nextSubmitSemaphore()});
      m_dlssRR.reset();
    }
    if(m_ngxParameter && m_config->dlssQuality != DlssQuality::Disabled
       && gbufferSize.width >= 32 && gbufferSize.height >= 32)
    {
      // Map quality index to NVSDK_NGX_PerfQuality_Value
      NVSDK_NGX_PerfQuality_Value quality;
      switch(m_config->dlssQuality)
      {
        case DlssQuality::MaxQuality:
          quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
          break;
        case DlssQuality::Balanced:
          quality = NVSDK_NGX_PerfQuality_Value_Balanced;
          break;
        case DlssQuality::Performance:
          quality = NVSDK_NGX_PerfQuality_Value_MaxPerf;
          break;
        case DlssQuality::UltraPerformance:
          quality = NVSDK_NGX_PerfQuality_Value_UltraPerformance;
          break;
        default:
          quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
          break;
      }
      vko::ngx::OptimalSettings dlssOptimal(*m_ngxParameter,
                                            params.framebuffer.size().width,
                                            params.framebuffer.size().height, quality);
      gbufferSize = {dlssOptimal.renderOptimalWidth, dlssOptimal.renderOptimalHeight};
      const uint32_t creationNodeMask   = 0x1;
      const uint32_t visibilityNodeMask = 0x1;
      m_dlssRR.emplace(params.context.device.get(), cmd, creationNodeMask,
                       visibilityNodeMask, *m_ngxParameter,
                       NVSDK_NGX_DLSSD_Create_Params{
                           .InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified,
                           .InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed,
                           .InUseHWDepth   = NVSDK_NGX_DLSS_Depth_Type_Linear,
                           .InWidth        = gbufferSize.width,
                           .InHeight       = gbufferSize.height,
                           .InTargetWidth  = params.framebuffer.size().width,
                           .InTargetHeight = params.framebuffer.size().height,
                           .InPerfQualityValue = quality,
                           .InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes,
                           .InEnableOutputSubrects = false,
                       });
    }

    // Recreate the G-buffer with the new DLSS-RR input size. Currently all
    // images are created even when DLSS is disabled for simplicity
    std::vector<VkFormat> colorBuffers{
        VK_FORMAT_R8G8B8A8_UNORM,       // [eGBufBaseColor_Metalness]
        VK_FORMAT_R8G8B8A8_UNORM,       // [eGBufSpecAlbedo]
        VK_FORMAT_R16_SFLOAT,           // [eGBufSpecHitDist]
        VK_FORMAT_R16G16B16A16_SFLOAT,  // [eGBufNormalRoughness]
        VK_FORMAT_R16G16_SFLOAT,        // [eGBufMotionVectors]
        VK_FORMAT_R16_SFLOAT,           // [eGBufViewZ]
        VK_FORMAT_R16G16B16A16_SFLOAT,  // [eGBufColor]
    };
    params.garbage.push(Garbage{moveAny(std::move(m_gBuffer)),
                                staging.commandBuffer().nextSubmitSemaphore()});
    m_gBuffer.emplace(params.context.device, params.context.allocator.get(), cmd,
                      glm::uvec2{gbufferSize.width, gbufferSize.height}, colorBuffers);

    // Write the new G-buffer image descriptors

    // Currently this code wants to partialy update the descriptor set, but
    // that's messy to do asynchronously. Instead, we recreate the whole
    // descriptor set.
    // TODO: split into two desciptorsets - one for acceleration structures and
    // one for gbuffer output
    params.garbage.push(Garbage{moveAny(std::move(m_rtDescriptorSet)),
                                staging.commandBuffer().nextSubmitSemaphore()});  // set first
    params.garbage.push(Garbage{moveAny(std::move(m_descriptorPool)),
                                staging.commandBuffer().nextSubmitSemaphore()});  // pool last
    m_descriptorPool = vko::SingleDescriptorSetPool(
        params.context.device.get(), m_rtPipeline.bindings().bindings,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);
    m_rtDescriptorSet = vko::DescriptorSet(params.context.device.get(), nullptr, m_descriptorPool,
                                           m_rtPipeline.descriptorSetLayout());
    auto tlas = m_lodInstanceTraverser ? m_lodInstanceTraverser->tlas() :
                                         m_lodMeshTraverser->tlas();
    m_rtPipeline.writeDescriptorSetInitial(tlas, params.common.m_bFrameInfo,
                                           params.common.m_bSkyParams,
                                           sceneVk.textureDescriptors, m_rtDescriptorSet);
    m_rtPipeline.writeDescriptorSetFramebuffer(
        *m_gBuffer,
        (m_config->dlssQuality != DlssQuality::Disabled) ?
            std::nullopt :
            std::optional{params.framebuffer.renderHdrImageInfo()},
        m_rtDescriptorSet);
    m_gBufferStale = false;
  }

  // Scene traversal
  {
    assert(bool(m_lodInstanceTraverser) ^ bool(m_lodMeshTraverser));  // must be one but not both
    if(m_lodInstanceTraverser)
    {
      m_lodInstanceTraverser->traverseAndBuildBVH(
          params.context.device.get(), params.context.allocator.get(),
          params.common.m_traversalParams, sceneVk, params.profiler,
          staging.commandBuffer().nextSubmitSemaphore(), cmd);
      m_lastTraverseStats = m_lodInstanceTraverser->stats(params.context.device.get());
    }
    if(m_lodMeshTraverser)
    {
      m_lodMeshTraverser->traverseAndBuildBVH(
          params.context.device.get(), params.context.allocator.get(),
          params.common.m_traversalParams, sceneVk, params.profiler,
          staging.commandBuffer().nextSubmitSemaphore(), cmd);
      m_lastTraverseStats = m_lodMeshTraverser->stats(params.context.device.get());
    }
  }

  float errorOverDistanceThreshold =
      nvclusterlodErrorOverDistance(params.common.m_config->lodTargetPixelError,
                                    params.camera.verticalFov,
                                    float(params.framebuffer.size().height));

  shaders::PathtraceConstant pushConstant{
      .instancesAddress           = sceneVk.instances,
      .meshesAddress              = sceneVk.meshPointers,
      .config                     = m_config->shaders,
      .frame                      = params.common.m_frameAccumIndex++,
      .errorOverDistanceThreshold = errorOverDistanceThreshold,
      .jitter      = halton(int(params.common.m_frameAccumIndex)),
      .dlssEnabled = (m_config->dlssQuality != DlssQuality::Disabled) ? 1 : 0,
  };

  // Ray trace
  {
    ScopedGpuTimer timer(params.context.device.get(), params.profiler, cmd, "Ray Trace");
    m_rtPipeline.trace(cmd, {m_rtDescriptorSet}, pushConstant,
                       {m_gBuffer->getSize().x, m_gBuffer->getSize().y});
  }

  // #DLSS_RR: Apply DLSS RR denoising if enabled
  if(m_dlssRR)
  {
    ScopedGpuTimer timer(params.context.device.get(), params.profiler, cmd, "DLSS RR Denoise");

    NVSDK_NGX_Resource_VK colorOut = NVSDK_NGX_Create_ImageView_Resource_VK(
        params.framebuffer.renderHdrView(), params.framebuffer.renderHdrImage(),
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, params.framebuffer.c_colorFormat,
        params.framebuffer.size().width, params.framebuffer.size().height, true);

    auto inputImage = [&](uint32_t gbufferIndex) {
      return NVSDK_NGX_Create_ImageView_Resource_VK(
          m_gBuffer->getColorImageView(gbufferIndex), m_gBuffer->getColorImage(gbufferIndex),
          {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, m_gBuffer->getColorFormat(gbufferIndex),
          m_gBuffer->getSize().x, m_gBuffer->getSize().y, false);
    };

    assert(m_gBuffer);
    NVSDK_NGX_Resource_VK albedoMetalness = inputImage(eGBufBaseColor_Metalness);
    NVSDK_NGX_Resource_VK specAlbedo      = inputImage(eGBufSpecAlbedo);
    NVSDK_NGX_Resource_VK specHitDist     = inputImage(eGBufSpecHitDist);
    NVSDK_NGX_Resource_VK normalRoughness = inputImage(eGBufNormalRoughness);
    NVSDK_NGX_Resource_VK motionVectors   = inputImage(eGBufMotionVectors);
    NVSDK_NGX_Resource_VK viewZ           = inputImage(eGBufViewZ);
    NVSDK_NGX_Resource_VK color           = inputImage(eGBufColor);

    // m_lastFrameInfo has been updated to the current frame by now. (This
    // doesn't read well, I know)
    glm::mat4 viewMat = params.common.m_lastFrameInfo.view;
    glm::mat4 projMat = params.common.m_lastFrameInfo.proj;

    NVSDK_NGX_VK_DLSSD_Eval_Params dlssdEvalParams{};
    dlssdEvalParams.pInOutput              = &colorOut;
    dlssdEvalParams.pInColor               = &color;
    dlssdEvalParams.pInDiffuseAlbedo       = &albedoMetalness;
    dlssdEvalParams.pInSpecularAlbedo      = &specAlbedo;
    dlssdEvalParams.pInSpecularHitDistance = &specHitDist;
    dlssdEvalParams.pInNormals             = &normalRoughness;
    dlssdEvalParams.pInDepth               = &viewZ;
    dlssdEvalParams.pInMotionVectors       = &motionVectors;
    dlssdEvalParams.pInRoughness           = &normalRoughness;
    dlssdEvalParams.InJitterOffsetX        = 0.5f - pushConstant.jitter.x;
    dlssdEvalParams.InJitterOffsetY        = 0.5f - pushConstant.jitter.y;
    dlssdEvalParams.InMVScaleX             = 1.0f;
    dlssdEvalParams.InMVScaleY             = 1.0f;
    dlssdEvalParams.InRenderSubrectDimensions.Width  = m_gBuffer->getSize().x;
    dlssdEvalParams.InRenderSubrectDimensions.Height = m_gBuffer->getSize().y;
    dlssdEvalParams.pInWorldToViewMatrix             = glm::value_ptr(viewMat);
    dlssdEvalParams.pInViewToClipMatrix              = glm::value_ptr(projMat);
    dlssdEvalParams.InReset = params.common.m_frameAccumIndex <= 1;
    m_dlssRR->evaluate(cmd, *m_ngxParameter, dlssdEvalParams);
    memoryBarrier(params.context.device.get(), cmd, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
  }

  // Blit debug visualization if enabled (works with or without DLSS)
  if(m_config->showGBufferDebug && m_config->dlssQuality != DlssQuality::Disabled)
  {
    blitGBufferDebugVisualization(params.context.device, cmd,
                                  params.framebuffer.renderHdrImage(),
                                  params.framebuffer.size());
  }
}

void RaytraceRenderer::uiOverlay()
{
  if(m_lastTraverseStats)
  {
    ImGui::Text("Triangles: %s",
                formatThousands(m_lastTraverseStats->triangleCount).c_str());

    ImVec4 errorColor = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
    if(m_lastTraverseStats->errorNodeQueueOverflow)
      ImGui::TextColored(errorColor, "Error: Node queue overflow");
    if(m_lastTraverseStats->errorClusterQueueOverflow)
      ImGui::TextColored(errorColor, "Error: Cluster queue overflow");
    if(m_lastTraverseStats->errorTraversalIncomplete)
      ImGui::TextColored(errorColor, "Error: Traversal incomplete");
    if(m_lastTraverseStats->errorEmptyTraversalResults)
      ImGui::TextColored(errorColor, "Error: Empty traversal results (%u)",
                         m_lastTraverseStats->errorEmptyTraversalResults);
    if(m_lastTraverseStats->errorBlasInputOverflow)
      ImGui::TextColored(errorColor, "Error: BLAS input overflow");
  }
}

void RaytraceRenderer::uiInline(bool& recreateRenderer, bool& resetFrameAccumulation)
{
  const char* visualizeItems[] = VISUALIZE_ENUM_NAMES;
  if(ImGui::BeginCombo("LOD Visualization", visualizeItems[m_config->shaders.lodVisualization],
                       ImGuiComboFlags_HeightLargest))
  {
    for(int32_t i = 0; i < int32_t(std::size(visualizeItems)); i++)
    {
      bool isSelected = (m_config->shaders.lodVisualization == i);
      if(ImGui::Selectable(visualizeItems[i], isSelected))
      {
        m_config->shaders.lodVisualization = i;
        resetFrameAccumulation             = true;
      }
      if(isSelected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  bool perMeshLod = !m_config->perInstanceTraversal;
  if(ImGui::Checkbox("Per-Mesh LOD", &perMeshLod))
  {
    recreateRenderer = true;
  }
  m_config->perInstanceTraversal = !perMeshLod;
}

void RaytraceRenderer::uiSection(bool& recreateRenderer, bool& resetFrameAccumulation)
{
  ImGui::Text("Ray Tracing");

  if(m_lodInstanceTraverser)
  {
    ImGui::Text("Memory: Traverse %s  BLAS %s  TLAS %s",
                formatBytes(m_lodInstanceTraverser->traversalMemory()).c_str(),
                formatBytes(m_lodInstanceTraverser->blasDeviceMemory()).c_str(),
                formatBytes(m_lodInstanceTraverser->tlasDeviceMemory()).c_str());
    if(ImGui::IsItemHovered())
      ImGui::SetTooltip("DLSS-RR and GBuffer memory is not tracked");
    ImGui::Text("Instances traversed per mesh: %.1f (avg)",
                float(m_sceneCounts.totalInstances) / float(m_sceneCounts.totalMeshes));
    ImGui::Text("Max instances traversed per mesh: %u", m_sceneCounts.maxInstancesPerMesh);
  }
  if(m_lodMeshTraverser)
  {
    ImGui::Text("Memory: Traverse %s  BLAS %s  TLAS %s",
                formatBytes(m_lodMeshTraverser->traversalMemory()).c_str(),
                formatBytes(m_lodMeshTraverser->blasDeviceMemory()).c_str(),
                formatBytes(m_lodMeshTraverser->tlasDeviceMemory()).c_str());
    if(ImGui::IsItemHovered())
      ImGui::SetTooltip("DLSS-RR and GBuffer memory is not tracked");
    ImGui::Text("Instances traversed per mesh: %.1f/%u (avg)",
                float(m_lastTraverseStats ? m_lastTraverseStats->instancesTraversed : 0u)
                    / float(m_sceneCounts.totalMeshes),
                TRAVERSAL_NEAREST_INSTANCE_COUNT);
    if(ImGui::IsItemHovered())
      ImGui::SetTooltip("The average number of instances considered per instance to produce a conservatively high detailed mesh");
    ImGui::Text("Max instances traversed per mesh: %u",
                m_lastTraverseStats ? m_lastTraverseStats->maxInstancesPerMesh : 0u);
    if(ImGui::IsItemHovered())
      ImGui::SetTooltip("The maximum number of instances considered per instance to produce a conservatively high detailed mesh");
  }

  // DLSS Quality dropdown
  const char* dlssQualityItems[] = {"Disabled", "Max Quality", "Balanced",
                                    "Performance", "Ultra Performance"};
  int         quality            = static_cast<int>(m_config->dlssQuality);

  // Disable DLSS controls if DLSS is not available
  ImGui::BeginDisabled(!m_ngxParameter);

  if(ImGui::Combo("DLSS Quality", &quality, dlssQualityItems, IM_ARRAYSIZE(dlssQualityItems)))
  {
    m_config->dlssQuality = static_cast<DlssQuality>(quality);
    m_gBufferStale        = true;
  }

  ImGui::EndDisabled();

  if(!m_ngxParameter && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
  {
    ImGui::SetTooltip("DLSS unsupported or disabled");
  }

  ImGui::Checkbox("Show GBuffer Debug", &m_config->showGBufferDebug);

  if(ImGui::Checkbox("Per-Instance Traversal", &m_config->perInstanceTraversal))
  {
    recreateRenderer = true;
  }

  ImGui::BeginDisabled(!m_config->perInstanceTraversal);
  if(ImGui::SliderFloat("Memory Reserve Scale", &m_config->memoryReserveScale, 1.0f, 50.0f))
  {
    recreateRenderer = true;
  }
  ImGui::EndDisabled();

  if(ImGui::Checkbox("Pathtrace", reinterpret_cast<bool*>(&m_config->shaders.pathtrace)))
  {
    resetFrameAccumulation = true;
  }
  if(ImGui::SliderInt("Subpixel Samples", &m_config->shaders.sampleCountPixel, 1, 32))
  {
    resetFrameAccumulation = true;
  }

  ImGui::BeginDisabled(!m_config->shaders.pathtrace);
  if(ImGui::SliderInt("Pathtrace Depth", &m_config->shaders.maxDepth, 1,
                      PATHTRACE_MAX_RGEN_RECURSION_DEPTH))
  {
    resetFrameAccumulation = true;
  }
  ImGui::EndDisabled();

  ImGui::BeginDisabled(m_config->shaders.pathtrace);
  if(ImGui::SliderInt("Ambient Occlusion Samples", &m_config->shaders.sampleCountAO, 1, 32))
  {
    resetFrameAccumulation = true;
  }
  if(ImGui::SliderFloat("Ambient Occlusion Radius", &m_config->shaders.aoRadius, 0.0f, 1000.0f))
  {
    resetFrameAccumulation = true;
  }
  ImGui::EndDisabled();

  if(ImGui::SliderFloat("Fog Height Offset", &m_config->shaders.fogHeightOffset,
                        m_sceneAabb.min.y, m_sceneAabb.max.y))
  {
    resetFrameAccumulation = true;
  }
  if(ImGui::SliderFloat("Fog Density", &m_config->shaders.fogDensity, 0.0f,
                        10.0f, "%.6f", ImGuiSliderFlags_Logarithmic))
  {
    resetFrameAccumulation = true;
  }
}

VkDeviceSize RaytraceRenderer::deviceMemoryUsage() const
{
  if(m_lodInstanceTraverser)
  {
    return m_lodInstanceTraverser->traversalMemory()
           + m_lodInstanceTraverser->blasDeviceMemory()
           + m_lodInstanceTraverser->tlasDeviceMemory();
  }
  if(m_lodMeshTraverser)
  {
    return m_lodMeshTraverser->traversalMemory() + m_lodMeshTraverser->blasDeviceMemory()
           + m_lodMeshTraverser->tlasDeviceMemory();
  }
  return 0;
}

void RaytraceRenderer::blitGBufferDebugVisualization(const vko::Device& device,
                                                     VkCommandBuffer    cmd,
                                                     VkImage    outputImage,
                                                     VkExtent2D outputSize)
{
  // Blit all non-color gbuffer images to corners of the output image
  // TODO: Visualize alpha channels
  // Layout: 3 on top (left to right), 3 on bottom (left to right)
  const uint32_t numImages  = 6;  // eGBufBaseColor_Metalness through eGBufViewZ
  const uint32_t debugWidth = outputSize.width / 6;  // Divide screen width by 6
  const uint32_t debugHeight = outputSize.height / 6;  // Divide screen height by 6

  // Transition output image to transfer dst optimal
  VkImageMemoryBarrier toTransferDst{
      .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .pNext               = nullptr,
      .srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout           = VK_IMAGE_LAYOUT_GENERAL,
      .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image               = outputImage,
      .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                              nullptr, 1, &toTransferDst);

  for(uint32_t i = 0; i < numImages; i++)
  {
    VkImage srcImage = m_gBuffer->getColorImage(i);

    // Transition source image to transfer src optimal
    VkImageMemoryBarrier toTransferSrc{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext               = nullptr,
        .srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = srcImage,
        .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                0, nullptr, 1, &toTransferSrc);

    // Calculate position: 3 on top row, 3 on bottom row
    uint32_t xPos, yPos;
    if(i < 3)
    {
      // Top row: left, center, right
      xPos = i * debugWidth;
      yPos = 0;
    }
    else
    {
      // Bottom row: left, center, right
      xPos = (i - 3) * debugWidth;
      yPos = outputSize.height - debugHeight;
    }

    // Blit the gbuffer image to the corner
    VkImageBlit blitRegion{};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[0]             = {0, 0, 0};
    blitRegion.srcOffsets[1]             = {int32_t(m_gBuffer->getSize().x),
                                            int32_t(m_gBuffer->getSize().y), 1};
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[0]             = {int32_t(xPos), int32_t(yPos), 0};
    blitRegion.dstOffsets[1] = {int32_t(xPos + debugWidth), int32_t(yPos + debugHeight), 1};

    device.vkCmdBlitImage(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          outputImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                          &blitRegion, VK_FILTER_LINEAR);

    // Transition source image back to general
    VkImageMemoryBarrier backToGeneral{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext               = nullptr,
        .srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = srcImage,
        .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0,
                                0, nullptr, 0, nullptr, 1, &backToGeneral);
  }

  // Transition output image back to general
  VkImageMemoryBarrier outputBackToGeneral{
      .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .pNext               = nullptr,
      .srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask       = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image               = outputImage,
      .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                              nullptr, 0, nullptr, 1, &outputBackToGeneral);
}
