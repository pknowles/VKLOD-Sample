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

#include <imgui/imgui_helper.h>
#include <nvvk/images_vk.hpp>
#include <nvvk/profiler_vk.hpp>
#include <renderer_raytrace.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <shaders/dh_bindings.h>

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

PathtracingPipeline::PathtracingPipeline(SampleGlslCompiler&                glslCompiler,
                                         ResourceAllocator*                 allocator,
                                         uint32_t                           queueGCT,
                                         std::vector<VkDescriptorSetLayout> descriptorSetLayouts)
{
  nvvk::DebugUtil dutil(allocator->getDevice());

  vkobj::ShaderModule shaderRaygen = reloadUntilCompiling(allocator->getDevice(), glslCompiler, "pathtrace.rgen.glsl",
                                                          shaderc_shader_kind::shaderc_glsl_raygen_shader);
  vkobj::ShaderModule shaderMiss   = reloadUntilCompiling(allocator->getDevice(), glslCompiler, "pathtrace.rmiss.glsl",
                                                          shaderc_shader_kind::shaderc_glsl_miss_shader);
  vkobj::ShaderModule shaderClosestHit = reloadUntilCompiling(allocator->getDevice(), glslCompiler, "pathtrace.rchit.glsl",
                                                              shaderc_shader_kind::shaderc_glsl_closesthit_shader);
  std::vector<VkPipelineShaderStageCreateInfo> shaderStages{
      {
          .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .pNext               = nullptr,
          .flags               = 0,
          .stage               = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
          .module              = shaderRaygen,
          .pName               = "main",
          .pSpecializationInfo = nullptr,
      },
      {
          .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .pNext               = nullptr,
          .flags               = 0,
          .stage               = VK_SHADER_STAGE_MISS_BIT_KHR,
          .module              = shaderMiss,
          .pName               = "main",
          .pSpecializationInfo = nullptr,
      },
      {
          .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .pNext               = nullptr,
          .flags               = 0,
          .stage               = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
          .module              = shaderClosestHit,
          .pName               = "main",
          .pSpecializationInfo = nullptr,
      },
  };
  for([[maybe_unused]] auto& stage : shaderStages)
    assert(stage.module != VK_NULL_HANDLE);
  dutil.setObjectName(shaderStages[0].module, "Raygen");
  dutil.setObjectName(shaderStages[1].module, "Miss");
  dutil.setObjectName(shaderStages[2].module, "Closest Hit");

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shadingGroups{
      {.sType                           = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .pNext                           = nullptr,
       .type                            = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
       .generalShader                   = 0 /* "raygen" shaderStages[0] */,
       .closestHitShader                = VK_SHADER_UNUSED_KHR,
       .anyHitShader                    = VK_SHADER_UNUSED_KHR,
       .intersectionShader              = VK_SHADER_UNUSED_KHR,
       .pShaderGroupCaptureReplayHandle = nullptr},
      {.sType                           = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .pNext                           = nullptr,
       .type                            = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
       .generalShader                   = 1 /* "miss" shaderStages[1] */,
       .closestHitShader                = VK_SHADER_UNUSED_KHR,
       .anyHitShader                    = VK_SHADER_UNUSED_KHR,
       .intersectionShader              = VK_SHADER_UNUSED_KHR,
       .pShaderGroupCaptureReplayHandle = nullptr},
      {.sType                           = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .pNext                           = nullptr,
       .type                            = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
       .generalShader                   = VK_SHADER_UNUSED_KHR,
       .closestHitShader                = 2 /* "closest hit" shaderStages[2] */,
       .anyHitShader                    = VK_SHADER_UNUSED_KHR,
       .intersectionShader              = VK_SHADER_UNUSED_KHR,
       .pShaderGroupCaptureReplayHandle = nullptr}};

  // Push constants - small struct to upload in the command buffer each frame
  VkPushConstantRange pushConstantRange{VK_SHADER_STAGE_ALL, 0, sizeof(shaders::PathtraceConstant)};

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext                  = nullptr,
      .flags                  = 0,
      .setLayoutCount         = static_cast<uint32_t>(descriptorSetLayouts.size()),
      .pSetLayouts            = descriptorSetLayouts.data(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstantRange,
  };
  m_pipelineLayout = vkobj::PipelineLayout(allocator->getDevice(), pipelineLayoutCreateInfo);
  dutil.DBG_NAME(m_pipelineLayout);

  // Enable cluster acceleration structures in the pipeline
  VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV pipelineClusterAccelerationStructureCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
      .pNext = nullptr,
      .allowClusterAccelerationStructure = true};

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo{
      .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .pNext                        = &pipelineClusterAccelerationStructureCreateInfo,
      .flags                        = 0,
      .stageCount                   = static_cast<uint32_t>(shaderStages.size()),
      .pStages                      = shaderStages.data(),
      .groupCount                   = static_cast<uint32_t>(shadingGroups.size()),
      .pGroups                      = shadingGroups.data(),
      .maxPipelineRayRecursionDepth = PATHTRACE_MAX_VK_RECURSION_DEPTH,
      .pLibraryInfo                 = nullptr,
      .pLibraryInterface            = nullptr,
      .pDynamicState                = nullptr,
      .layout                       = m_pipelineLayout,
      .basePipelineHandle           = VK_NULL_HANDLE,
      .basePipelineIndex            = 0,
  };
  VkPipeline pipeline;
  NVVK_CHECK(vkCreateRayTracingPipelinesKHR(allocator->getDevice(), {}, {}, 1, &pipelineCreateInfo, nullptr, &pipeline));
  m_pipeline = vkobj::Pipeline(allocator->getDevice(), std::move(pipeline));
  dutil.DBG_NAME(m_pipeline);

  // Requesting ray tracing properties
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProperties{
      .sType                              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
      .pNext                              = nullptr,
      .shaderGroupHandleSize              = 0,
      .maxRayRecursionDepth               = 0,
      .maxShaderGroupStride               = 0,
      .shaderGroupBaseAlignment           = 0,
      .shaderGroupHandleCaptureReplaySize = 0,
      .maxRayDispatchInvocationCount      = 0,
      .shaderGroupHandleAlignment         = 0,
      .maxRayHitAttributeSize             = 0,
  };
  VkPhysicalDeviceProperties2 prop2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &rtPipelineProperties, .properties = {}};
  vkGetPhysicalDeviceProperties2(allocator->getPhysicalDevice(), &prop2);

  // Create utilities to create BLAS/TLAS and the Shader Binding Table (SBT)
  m_sbt = std::make_unique<SBT>();
  m_sbt->setup(allocator->getDevice(), queueGCT, allocator, rtPipelineProperties);
  m_sbt->create(m_pipeline, pipelineCreateInfo);
}

std::unique_ptr<nvvk::DescriptorSetContainer> PathtracingPipeline::makeDescriptorSet(VkDevice device, uint32_t textureCount)
{
  // This descriptor set, holds the top level acceleration structure and the output image
  auto result = std::make_unique<nvvk::DescriptorSetContainer>(device);
  result->addBinding(BRtTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  result->addBinding(BRtOutBaseColor_Metalness, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);  // eGBufBaseColor_Metalness
  result->addBinding(BRtOutSpecAlbedo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);   // eGBufSpecAlbedo
  result->addBinding(BRtOutSpecHitDist, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);  // eGBufSpecHitDist
  result->addBinding(BRtOutNormalRoughness, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);  // eGBufNormalRoughness
  result->addBinding(BRtOutMotionVectors, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);  // eGBufMotionVectors
  result->addBinding(BRtOutViewZ, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);          // eGBufViewZ
  result->addBinding(BRtOutColor, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);          // eGBufColor
  result->addBinding(BRtFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  result->addBinding(BRtSkyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  result->addBinding(BRtTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, std::max(textureCount, 1u), VK_SHADER_STAGE_ALL);
  result->initLayout();
  result->initPool(1);
  return result;
}

void PathtracingPipeline::writeDescriptorSetInitial(VkDevice                               device,
                                                    VkAccelerationStructureKHR             tlas,
                                                    VkBuffer                               frameInfo,
                                                    VkBuffer                               skyParams,
                                                    std::span<const VkDescriptorImageInfo> textures,
                                                    nvvk::DescriptorSetContainer&          descriptorSet)
{
  VkWriteDescriptorSetAccelerationStructureKHR tlasDesc{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                                                        .pNext                      = nullptr,
                                                        .accelerationStructureCount = 1,
                                                        .pAccelerationStructures    = &tlas};
  VkDescriptorBufferInfo            frameInfoDesc{frameInfo, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo            skyParamsDesc{skyParams, 0, VK_WHOLE_SIZE};
  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(descriptorSet.makeWrite(DSRt, BRtTlas, &tlasDesc));
  writes.emplace_back(descriptorSet.makeWrite(DSRt, BRtFrameInfo, &frameInfoDesc));
  writes.emplace_back(descriptorSet.makeWrite(DSRt, BRtSkyParam, &skyParamsDesc));
  if(textures.size())
    writes.emplace_back(descriptorSet.makeWriteArray(DSRt, BRtTextures, textures.data()));
  vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void PathtracingPipeline::writeDescriptorSetFramebuffer(VkDevice                             device,
                                                        const nvvkhl::GBuffer&               gBuffer,
                                                        std::optional<VkDescriptorImageInfo> passthroughColor,
                                                        nvvk::DescriptorSetContainer&        descriptorSet)
{
  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(descriptorSet.makeWrite(DSRt, BRtOutBaseColor_Metalness,
                                              &gBuffer.getDescriptorImageInfo(eGBufBaseColor_Metalness)));
  writes.emplace_back(descriptorSet.makeWrite(DSRt, BRtOutSpecAlbedo, &gBuffer.getDescriptorImageInfo(eGBufSpecAlbedo)));
  writes.emplace_back(descriptorSet.makeWrite(DSRt, BRtOutSpecHitDist, &gBuffer.getDescriptorImageInfo(eGBufSpecHitDist)));
  writes.emplace_back(descriptorSet.makeWrite(DSRt, BRtOutNormalRoughness, &gBuffer.getDescriptorImageInfo(eGBufNormalRoughness)));
  writes.emplace_back(descriptorSet.makeWrite(DSRt, BRtOutMotionVectors, &gBuffer.getDescriptorImageInfo(eGBufMotionVectors)));
  writes.emplace_back(descriptorSet.makeWrite(DSRt, BRtOutViewZ, &gBuffer.getDescriptorImageInfo(eGBufViewZ)));
  writes.emplace_back(descriptorSet.makeWrite(
      DSRt, BRtOutColor, passthroughColor ? &*passthroughColor : &gBuffer.getDescriptorImageInfo(eGBufColor)));
  vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void PathtracingPipeline::trace(VkCommandBuffer                   cmd,
                                std::vector<VkDescriptorSet>      descriptorSets,
                                const shaders::PathtraceConstant& pushConstant,
                                glm::uvec2                        vpSize) const
{
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, DSRt,
                          static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pushConstant), &pushConstant);
  const auto& regions = m_sbt->getRegions();
  vkCmdTraceRaysKHR(cmd, regions.data(), &regions[1], &regions[2], &regions[3], vpSize.x, vpSize.y, 1);
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}


RaytraceRenderer::RaytraceRenderer(const RenderInitParams& params, RaytraceConfig& config)
    : m_config(config)
    , m_rtBinding(PathtracingPipeline::makeDescriptorSet(params.context.device, uint32_t(params.sceneVk.textures.size())))
    , m_rtPipeline(params.glslCompiler, params.context.allocator, params.context.queueFamilyIndex, {m_rtBinding->getLayout()})
    , m_sceneCounts(params.scene.counts)
    , m_sceneAabb(params.scene.worldAABB)
    , m_ngxParameter(ngx::makeCapabilityParameter())
    , m_tonemap(params.context.device, params.context.allocator)
{
  auto preset = NVSDK_NGX_DLSS_Hint_Render_Preset_Default;
  m_ngxParameter->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality, preset);
  m_ngxParameter->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality, preset);
  m_ngxParameter->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced, preset);
  m_ngxParameter->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance, preset);
  m_ngxParameter->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance, preset);
  ngx::assertRayReconstructionSupported(*m_ngxParameter);

  VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
  if(m_config.perInstanceTraversal)
  {
    m_lodInstanceTraverser = LodInstanceTraverser{params.context.allocator,
                                                  params.glslCompiler,
                                                  params.context.commandPool,
                                                  params.context.queue,
                                                  params.context.queueFamilyIndex,
                                                  params.scene,
                                                  params.sceneVk,
                                                  uint32_t(float(m_config.maxTotalClusterGroups) * m_config.memoryReserveScale)};
    tlas                   = m_lodInstanceTraverser->tlas();
  }
  else
  {
    m_lodMeshTraverser = LodMeshTraverser{params.context.allocator,
                                          params.glslCompiler,
                                          params.context.commandPool,
                                          params.context.queue,
                                          params.context.queueFamilyIndex,
                                          params.scene,
                                          params.sceneVk,
                                          m_config.maxTotalClusterGroups};
    tlas               = m_lodMeshTraverser->tlas();
  }
  PathtracingPipeline::writeDescriptorSetInitial(params.context.device, tlas, params.common.m_bFrameInfo,
                                                 params.common.m_bSkyParams, params.sceneVk.textureDescriptors, *m_rtBinding);
}

void RaytraceRenderer::updatedFrambuffer(const RenderParams&)
{
  // Mark the gBuffer as stale. It's more convenient to handle updates in
  // render()
  m_gBufferStale = true;
}

void RaytraceRenderer::render(const RenderParams& params, const SceneVK& sceneVk, VkCommandBuffer cmd)
{
  // Check if the framebuffer is stale. This can happen on first launch or if
  // the window is resized.
  if(m_gBufferStale)
  {
    VkExtent2D gbufferSize = params.framebuffer.size();

    // Recreate DLSS-RR with the new framebuffer size
    if(m_dlssRR)
    {
      params.garbage.push(Garbage{moveAny(std::move(*m_dlssRR)), params.queueStates.primary.nextSubmitValue()});
      m_dlssRR.reset();
    }
    if(m_config.dlssQuality != DlssQuality::Disabled && gbufferSize.width >= 32 && gbufferSize.height >= 32)
    {
      // Map quality index to NVSDK_NGX_PerfQuality_Value
      NVSDK_NGX_PerfQuality_Value quality;
      switch(m_config.dlssQuality)
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
      ngx::OptimalSettings dlssOptimal(*m_ngxParameter, params.framebuffer.size().width, params.framebuffer.size().height, quality);
      gbufferSize                       = {dlssOptimal.renderOptimalWidth, dlssOptimal.renderOptimalHeight};
      const uint32_t creationNodeMask   = 0x1;
      const uint32_t visibilityNodeMask = 0x1;
      m_dlssRR.emplace(params.context.device, cmd, creationNodeMask, visibilityNodeMask, *m_ngxParameter,
                       NVSDK_NGX_DLSSD_Create_Params{
                           .InDenoiseMode      = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified,
                           .InRoughnessMode    = NVSDK_NGX_DLSS_Roughness_Mode_Packed,
                           .InUseHWDepth       = NVSDK_NGX_DLSS_Depth_Type_Linear,
                           .InWidth            = gbufferSize.width,
                           .InHeight           = gbufferSize.height,
                           .InTargetWidth      = params.framebuffer.size().width,
                           .InTargetHeight     = params.framebuffer.size().height,
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
    params.garbage.push(Garbage{moveAny(std::move(m_gBuffer)), params.queueStates.primary.nextSubmitValue()});
    m_gBuffer = std::make_unique<nvvkhl::GBuffer>(params.context.device, params.context.allocator, gbufferSize,
                                                  colorBuffers, VK_FORMAT_UNDEFINED);

    // Write the new G-buffer image descriptors
    PathtracingPipeline::writeDescriptorSetFramebuffer(params.context.device, *m_gBuffer.get(),
                                                       (m_config.dlssQuality != DlssQuality::Disabled) ?
                                                           std::nullopt :
                                                           std::optional{params.framebuffer.renderHdrImageInfo()},
                                                       *m_rtBinding);
    m_gBufferStale = false;
  }

  // Scene traversal
  {
    assert(bool(m_lodInstanceTraverser) ^ bool(m_lodMeshTraverser));  // must be one but not both
    if(m_lodInstanceTraverser)
    {
      m_lodInstanceTraverser->traverseAndBuildBVH(params.context.allocator, params.common.m_traversalParams, sceneVk,
                                                  params.profiler, params.queueStates.primary.nextSubmitValue(), cmd);
      m_lastTraverseStats = m_lodInstanceTraverser->stats(params.context.device);
    }
    if(m_lodMeshTraverser)
    {
      m_lodMeshTraverser->traverseAndBuildBVH(params.context.allocator, params.common.m_traversalParams, sceneVk,
                                              params.profiler, params.queueStates.primary.nextSubmitValue(), cmd);
      m_lastTraverseStats = m_lodMeshTraverser->stats(params.context.device);
    }
  }

  float errorOverDistanceThreshold =
      nvclusterlodErrorOverDistance(params.common.m_config.lodTargetPixelError, glm::radians(CameraManip.getFov()),
                                    float(params.framebuffer.size().height));

  shaders::PathtraceConstant pushConstant{
      .instancesAddress           = sceneVk.instances.address(),
      .meshesAddress              = sceneVk.meshPointers.address(),
      .config                     = m_config.shaders,
      .frame                      = params.common.m_frameAccumIndex++,
      .errorOverDistanceThreshold = errorOverDistanceThreshold,
      .jitter                     = halton(int(params.common.m_frameAccumIndex)),
      .dlssEnabled                = (m_config.dlssQuality != DlssQuality::Disabled) ? 1 : 0,
  };

  // Ray trace
  {
    nvvk::ProfilerVK::Section timer(params.profiler, "Ray Trace", cmd);
    m_rtPipeline.trace(cmd, {m_rtBinding->getSet()}, pushConstant, {m_gBuffer->getSize().width, m_gBuffer->getSize().height});
  }

  // #DLSS_RR: Apply DLSS RR denoising if enabled
  if(m_dlssRR)
  {
    nvvk::ProfilerVK::Section timer(params.profiler, "DLSS RR Denoise", cmd);

    NVSDK_NGX_Resource_VK colorOut =
        NVSDK_NGX_Create_ImageView_Resource_VK(params.framebuffer.renderHdrView(), params.framebuffer.renderHdrImage(),
                                               {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, params.framebuffer.c_colorFormat,
                                               params.framebuffer.size().width, params.framebuffer.size().height, true);

    auto inputImage = [&](uint32_t gbufferIndex) {
      return NVSDK_NGX_Create_ImageView_Resource_VK(m_gBuffer->getColorImageView(gbufferIndex),
                                                    m_gBuffer->getColorImage(gbufferIndex),
                                                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, m_gBuffer->getColorFormat(gbufferIndex),
                                                    m_gBuffer->getSize().width, m_gBuffer->getSize().height, false);
    };

    assert(m_gBuffer);
    NVSDK_NGX_Resource_VK albedoMetalness = inputImage(eGBufBaseColor_Metalness);
    NVSDK_NGX_Resource_VK specAlbedo      = inputImage(eGBufSpecAlbedo);
    NVSDK_NGX_Resource_VK specHitDist     = inputImage(eGBufSpecHitDist);
    NVSDK_NGX_Resource_VK normalRoughness = inputImage(eGBufNormalRoughness);
    NVSDK_NGX_Resource_VK motionVectors   = inputImage(eGBufMotionVectors);
    NVSDK_NGX_Resource_VK viewZ           = inputImage(eGBufViewZ);
    NVSDK_NGX_Resource_VK color           = inputImage(eGBufColor);

    NVSDK_NGX_VK_DLSSD_Eval_Params dlssdEvalParams{};
    dlssdEvalParams.pInOutput                        = &colorOut;
    dlssdEvalParams.pInColor                         = &color;
    dlssdEvalParams.pInDiffuseAlbedo                 = &albedoMetalness;
    dlssdEvalParams.pInSpecularAlbedo                = &specAlbedo;
    dlssdEvalParams.pInSpecularHitDistance           = &specHitDist;
    dlssdEvalParams.pInNormals                       = &normalRoughness;
    dlssdEvalParams.pInDepth                         = &viewZ;
    dlssdEvalParams.pInMotionVectors                 = &motionVectors;
    dlssdEvalParams.pInRoughness                     = &normalRoughness;
    dlssdEvalParams.InJitterOffsetX                  = 0.5f - pushConstant.jitter.x;
    dlssdEvalParams.InJitterOffsetY                  = 0.5f - pushConstant.jitter.y;
    dlssdEvalParams.InMVScaleX                       = 1.0f;
    dlssdEvalParams.InMVScaleY                       = 1.0f;
    dlssdEvalParams.InRenderSubrectDimensions.Width  = m_gBuffer->getSize().width;
    dlssdEvalParams.InRenderSubrectDimensions.Height = m_gBuffer->getSize().height;
    dlssdEvalParams.pInWorldToViewMatrix             = glm::value_ptr(params.common.m_lastFrameInfo.view);
    dlssdEvalParams.pInViewToClipMatrix              = glm::value_ptr(params.common.m_lastFrameInfo.proj);
    dlssdEvalParams.InReset                          = params.common.m_frameAccumIndex <= 1;
    m_dlssRR->evaluate(cmd, *m_ngxParameter, dlssdEvalParams);
    memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
  }

  // Blit debug visualization if enabled (works with or without DLSS)
  if(m_config.showGBufferDebug && m_config.dlssQuality != DlssQuality::Disabled)
  {
    blitGBufferDebugVisualization(cmd, params.framebuffer.renderHdrImage(), params.framebuffer.size());
  }
}

void RaytraceRenderer::uiOverlay()
{
  if(m_lastTraverseStats)
  {
    ImGui::Text("Triangles: %s", formatThousands(m_lastTraverseStats->triangleCount).c_str());
    ImGuiH::tooltip("Total ray traced triangles");

    ImVec4 errorColor = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
    if(m_lastTraverseStats->errorNodeQueueOverflow)
      ImGui::TextColored(errorColor, "Error: Node queue overflow");
    if(m_lastTraverseStats->errorClusterQueueOverflow)
      ImGui::TextColored(errorColor, "Error: Cluster queue overflow");
    if(m_lastTraverseStats->errorTraversalIncomplete)
      ImGui::TextColored(errorColor, "Error: Traversal incomplete");
    if(m_lastTraverseStats->errorEmptyTraversalResults)
      ImGui::TextColored(errorColor, "Error: Empty traversal results (%u)", m_lastTraverseStats->errorEmptyTraversalResults);
    if(m_lastTraverseStats->errorBlasInputOverflow)
      ImGui::TextColored(errorColor, "Error: BLAS input overflow");
  }
}

void RaytraceRenderer::uiInline(bool& recreateRenderer, bool& resetFrameAccumulation)
{
  const char* visualizeItems[] = VISUALIZE_ENUM_NAMES;
  if(ImGui::BeginCombo("LOD Visualization", visualizeItems[m_config.shaders.lodVisualization]))
  {
    for(int32_t i = 0; i < int32_t(std::size(visualizeItems)); i++)
    {
      bool isSelected = (m_config.shaders.lodVisualization == i);
      if(ImGui::Selectable(visualizeItems[i], isSelected))
      {
        m_config.shaders.lodVisualization = i;
        resetFrameAccumulation            = true;
      }
      if(isSelected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  bool perMeshLod = !m_config.perInstanceTraversal;
  if(ImGui::Checkbox("Per-Mesh LOD", &perMeshLod))
  {
    recreateRenderer = true;
  }
  m_config.perInstanceTraversal = !perMeshLod;
}

void RaytraceRenderer::uiSection(bool& recreateRenderer, bool& resetFrameAccumulation)
{
  ImGui::Text("Ray Tracing");

  if(m_lodInstanceTraverser)
  {
    ImGui::Text("Memory: Traverse %s  BLAS %s  TLAS %s", formatBytes(m_lodInstanceTraverser->traversalMemory()).c_str(),
                formatBytes(m_lodInstanceTraverser->blasDeviceMemory()).c_str(),
                formatBytes(m_lodInstanceTraverser->tlasDeviceMemory()).c_str());
    ImGuiH::tooltip("DLSS-RR and GBuffer memory is not tracked", true);
    ImGui::Text("Instances traversed per mesh: %.1f (avg)", float(m_sceneCounts.totalInstances) / float(m_sceneCounts.totalMeshes));
    ImGui::Text("Max instances traversed per mesh: %u", m_sceneCounts.maxInstancesPerMesh);
  }
  if(m_lodMeshTraverser)
  {
    ImGui::Text("Memory: Traverse %s  BLAS %s  TLAS %s", formatBytes(m_lodMeshTraverser->traversalMemory()).c_str(),
                formatBytes(m_lodMeshTraverser->blasDeviceMemory()).c_str(),
                formatBytes(m_lodMeshTraverser->tlasDeviceMemory()).c_str());
    ImGuiH::tooltip("DLSS-RR and GBuffer memory is not tracked", true);
    ImGui::Text("Instances traversed per mesh: %.1f/%u (avg)",
                float(m_lastTraverseStats ? m_lastTraverseStats->instancesTraversed : 0u) / float(m_sceneCounts.totalMeshes),
                TRAVERSAL_NEAREST_INSTANCE_COUNT);
    ImGuiH::tooltip("The average number of instances considered per instance to produce a conservatively high detailed mesh", true);
    ImGui::Text("Max instances traversed per mesh: %u", m_lastTraverseStats ? m_lastTraverseStats->maxInstancesPerMesh : 0u);
    ImGuiH::tooltip("The maximum number of instances considered per instance to produce a conservatively high detailed mesh", true);
  }

  using namespace ImGuiH;
  PropertyEditor::begin();
  resetFrameAccumulation = false;
  recreateRenderer       = false;

  // DLSS Quality dropdown
  const char* dlssQualityItems[] = {"Disabled", "Max Quality", "Balanced", "Performance", "Ultra Performance"};
  if(PropertyEditor::entry("DLSS Quality", [&] {
       int  quality         = static_cast<int>(m_config.dlssQuality);
       bool changed         = ImGui::Combo("DLSS Quality", &quality, dlssQualityItems, IM_ARRAYSIZE(dlssQualityItems));
       m_config.dlssQuality = static_cast<DlssQuality>(quality);
       return changed;
     }))
  {
    m_gBufferStale = true;
  }
  PropertyEditor::entry("Show GBuffer Debug",
                        [&] { return ImGui::Checkbox("Show GBuffer Debug", &m_config.showGBufferDebug); });
  recreateRenderer       = recreateRenderer | PropertyEditor::entry("Per-Instance Traversal", [&] {
                       return ImGui::Checkbox("Per-Instance Traversal", &m_config.perInstanceTraversal);
                     });
  ImGui::BeginDisabled(!m_config.perInstanceTraversal);
  recreateRenderer = recreateRenderer | PropertyEditor::entry("Memory Reserve Scale", [&] {
                       return ImGui::SliderFloat("Memory Reserve Scale", &m_config.memoryReserveScale, 1.0f, 50.0f);
                     });
  ImGui::EndDisabled();
  resetFrameAccumulation = resetFrameAccumulation | PropertyEditor::entry("Pathtrace", [&] {
                             return ImGui::Checkbox("Pathtrace", reinterpret_cast<bool*>(&m_config.shaders.pathtrace));
                           });
  resetFrameAccumulation = resetFrameAccumulation | PropertyEditor::entry("Subpixel Samples", [&] {
                             return ImGui::SliderInt("Subpixel Samples", &m_config.shaders.sampleCountPixel, 1, 32);
                           });
  ImGui::BeginDisabled(!m_config.shaders.pathtrace);
  resetFrameAccumulation =
      resetFrameAccumulation | PropertyEditor::entry("Pathtrace Depth", [&] {
        return ImGui::SliderInt("Pathtrace Depth", &m_config.shaders.maxDepth, 1, PATHTRACE_MAX_RGEN_RECURSION_DEPTH);
      });
  ImGui::EndDisabled();
  ImGui::BeginDisabled(m_config.shaders.pathtrace);
  resetFrameAccumulation = resetFrameAccumulation | PropertyEditor::entry("Ambient Occlusion Samples", [&] {
                             return ImGui::SliderInt("Ambient Occlusion Samples", &m_config.shaders.sampleCountAO, 1, 32);
                           });
  resetFrameAccumulation = resetFrameAccumulation | PropertyEditor::entry("Ambient Occlusion Radius", [&] {
                             return ImGui::SliderFloat("Ambient Occlusion Radius", &m_config.shaders.aoRadius, 0.0f, 1000.0f);
                           });
  ImGui::EndDisabled();
  resetFrameAccumulation = resetFrameAccumulation | PropertyEditor::entry("Fog Height Offset", [&] {
                             return ImGui::SliderFloat("Fog Height Offset", &m_config.shaders.fogHeightOffset,
                                                       m_sceneAabb.min.y, m_sceneAabb.max.y);
                           });
  resetFrameAccumulation = resetFrameAccumulation | PropertyEditor::entry("Fog Density", [&] {
                             return ImGui::SliderFloat("Fog Density", &m_config.shaders.fogDensity, 0.0f, 10.0f, "%.6f",
                                                       ImGuiSliderFlags_Logarithmic);
                           });
  PropertyEditor::end();
}

VkDeviceSize RaytraceRenderer::deviceMemoryUsage() const
{
  if(m_lodInstanceTraverser)
  {
    return m_lodInstanceTraverser->traversalMemory() + m_lodInstanceTraverser->blasDeviceMemory()
           + m_lodInstanceTraverser->tlasDeviceMemory();
  }
  if(m_lodMeshTraverser)
  {
    return m_lodMeshTraverser->traversalMemory() + m_lodMeshTraverser->blasDeviceMemory() + m_lodMeshTraverser->tlasDeviceMemory();
  }
  return 0;
}

void RaytraceRenderer::blitGBufferDebugVisualization(VkCommandBuffer cmd, VkImage outputImage, VkExtent2D outputSize)
{
  // Blit all non-color gbuffer images to corners of the output image
  // TODO: Visualize alpha channels
  // Layout: 3 on top (left to right), 3 on bottom (left to right)
  const uint32_t numImages   = 6;                      // eGBufBaseColor_Metalness through eGBufViewZ
  const uint32_t debugWidth  = outputSize.width / 6;   // Divide screen width by 6
  const uint32_t debugHeight = outputSize.height / 6;  // Divide screen height by 6

  // Transition output image to transfer dst optimal
  nvvk::cmdBarrierImageLayout(cmd, outputImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

  for(uint32_t i = 0; i < numImages; i++)
  {
    VkImage srcImage = m_gBuffer->getColorImage(i);

    // Transition source image to transfer src optimal
    nvvk::cmdBarrierImageLayout(cmd, srcImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

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
    blitRegion.srcOffsets[1] = {int32_t(m_gBuffer->getSize().width), int32_t(m_gBuffer->getSize().height), 1};
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[0]             = {int32_t(xPos), int32_t(yPos), 0};
    blitRegion.dstOffsets[1]             = {int32_t(xPos + debugWidth), int32_t(yPos + debugHeight), 1};

    vkCmdBlitImage(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, outputImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_LINEAR);

    // Transition source image back to general
    nvvk::cmdBarrierImageLayout(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  }

  // Transition output image back to general
  nvvk::cmdBarrierImageLayout(cmd, outputImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
}
