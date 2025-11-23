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

#include <glm/common.hpp>
#include <nvvkhl/tonemap_postprocess.hpp>
#include <renderer_common.hpp>
#include <sample_vulkan_context.hpp>
#include <sample_vulkan_objects.hpp>
#include <vko/shortcuts.hpp>

TimelineQueueContainer::TimelineQueueContainer(const SampleVulkanContext& vkContext)
    : primary(vkContext.device,
              vkContext.primary.location.family,
              vkContext.primary.location.index)
    , compute(vkContext.device,
              vkContext.compute.location.family,
              vkContext.compute.location.index)
    , transfer(vkContext.device,
               vkContext.transfer.location.family,
               vkContext.transfer.location.index)
    , asyncLoad(vkContext.device,
                vkContext.asyncLoad.location.family,
                vkContext.asyncLoad.location.index)
{
}

RendererCommon::RendererCommon(const vko::Device&              device,
                               vko::vma::Allocator&            allocator,
                               vko::shared_obj<RendererConfig> config)
    : m_config(std::move(config))
    , m_traversalParams(initialTraversalParams())
    , m_skyParams(nvvkhl_shaders::initSimpleSkyParameters())
    , m_bFrameInfo(device,
                   1,
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   allocator)
    , m_bSkyParams(device,
                   1,
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   allocator)
{
  // Tone down the sky intensity relative to the sun
  //m_skyParams.horizonColor *= 0.6;
  //m_skyParams.groundColor *= 0.6;
  //m_skyParams.skyColor *= 0.6;
  m_skyParams.groundColor = m_skyParams.horizonColor;

  // Brighter yellow sun
  m_skyParams.sunIntensity  = 4.0f;
  m_skyParams.sunColor      = glm::vec3(1.0f, 0.9f, 0.7f);
  m_skyParams.lightRadiance = m_skyParams.sunColor * m_skyParams.sunIntensity;
}

void RendererCommon::cmdUpdateParams(const vko::Device& device,
                                     Framebuffer&       framebuffer,
                                     Camera&            camera,
                                     float maxWorldDiagonalInObjectSpace,
                                     VkCommandBuffer cmd)
{
  auto  imageSize       = framebuffer.size();
  float viewAspectRatio = float(imageSize.width) / float(imageSize.height);

  // Update the uniform buffer containing frame info
  shaders::FrameParams frameInfo{};
  frameInfo.view = camera.view();
  frameInfo.proj = glm::perspectiveRH_ZO(camera.verticalFov, viewAspectRatio,
                                         camera.clipPlanes.x, camera.clipPlanes.y);
  frameInfo.proj[1][1] *= -1;
  frameInfo.projInv  = glm::inverse(frameInfo.proj);
  frameInfo.viewInv  = glm::inverse(frameInfo.view);
  frameInfo.viewProj = frameInfo.proj * frameInfo.view;
  frameInfo.viewLast = m_lastFrameInfo.view;
  frameInfo.camPos   = glm::vec3(camera.viewInv()[3]);
  device.vkCmdUpdateBuffer(cmd, m_bFrameInfo, 0, sizeof(frameInfo), &frameInfo);

  // Update the sky
  device.vkCmdUpdateBuffer(cmd, m_bSkyParams, 0, sizeof(m_skyParams), &m_skyParams);

  vko::cmdMemoryBarrier(device, cmd, {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT},
                        {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT});

  // first frame use same
  if(!m_frameIndex)
    m_lastFrameInfo = frameInfo;

  // Update traversal parameters
  if(!m_frameIndex || !m_config->lockLodCamera)
  {
    m_traversalParams.viewTransform = frameInfo.view;
    m_traversalParams.viewPosition  = glm::vec3(frameInfo.viewInv[3]);
    m_traversalParams.hizViewProj   = m_lastFrameInfo.viewProj;
  }

  float errorOverDistanceThreshold =
      nvclusterlodErrorOverDistance(m_config->lodTargetPixelError,
                                    camera.verticalFov, float(imageSize.height));

  m_traversalParams.errorOverDistanceThreshold = errorOverDistanceThreshold;
  m_traversalParams.distanceToUNorm32 =
      float(double(std::numeric_limits<uint32_t>::max()) / double(maxWorldDiagonalInObjectSpace));
  m_traversalParams.useOcclusion   = m_config->useOcclusion ? 1 : 0;
  m_traversalParams.hizSizeFactors = framebuffer.hizFarFactors();
  m_traversalParams.hizSizeMax     = framebuffer.hizFarMax();
  m_traversalParams.hizViewport =
      glm::vec2(framebuffer.size().width, framebuffer.size().height);

  m_lastFrameInfo = frameInfo;
  m_frameIndex++;
}

bool RendererCommon::uiLod(const Scene&, const Framebuffer&)
{
  bool changed = false;

  ImGui::Text("Level of Detail (View)");
  changed |= ImGui::SliderFloat("Max. Pixel Error",
                                &m_config->lodTargetPixelError, 0.5f, 1000.0f);
  ImGui::Checkbox("Lock Camera", &m_config->lockLodCamera);
  ImGui::Checkbox("Use Occlusion", &m_config->useOcclusion);

  return changed;
}

bool RendererCommon::uiSky()
{
  bool changed = false;
  ImGui::Text("Sun Orientation");
  changed |= ImGui::SliderFloat("Sun Brightness", &m_skyParams.sunIntensity, 0.0f, 100.0f);

  // Simple azimuth/elevation sliders for sun direction
  glm::vec3 dir       = m_skyParams.directionToLight;
  float     azimuth   = atan2f(dir.x, dir.z);
  float     elevation = asinf(dir.y);
  if(ImGui::SliderAngle("Azimuth", &azimuth, -180.0f, 180.0f))
  {
    changed = true;
  }
  if(ImGui::SliderAngle("Elevation", &elevation, -90.0f, 90.0f))
  {
    changed = true;
  }
  if(changed)
  {
    m_skyParams.directionToLight =
        glm::vec3(sinf(azimuth) * cosf(elevation), sinf(elevation),
                  cosf(azimuth) * cosf(elevation));
    m_skyParams.lightRadiance = m_skyParams.sunColor * m_skyParams.sunIntensity;
  }
  return changed;
}

TonemapPipeline::TonemapPipeline(const vko::Device& device, SampleGlslCompiler& glslCompiler)
    : m_tonemapper(std::make_unique<nvvkhl::TonemapperPostProcess>())
{
  // Find and compile tonemap compute shader
  auto shaderPath = glslCompiler.find("nvvkhl/shaders/tonemapper.comp.glsl");
  if(shaderPath.empty())
  {
    throw std::runtime_error("Failed to find tonemapper.comp.glsl");
  }

  std::ifstream file(shaderPath, std::ios::binary);
  if(!file)
    throw std::runtime_error("Failed to open shader source file: " + shaderPath.string());
  std::string source((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());

  vko::shaderc::SpirvBinary binary(glslCompiler.m_compiler, source,
                                   shaderc_glsl_compute_shader, shaderPath.string(),
                                   "main", glslCompiler.m_options);

  m_tonemapper->createComputePipeline(device, binary.span());
}

TonemapPipeline::~TonemapPipeline() {}

static VkImageCreateInfo make2DImageCreateInfo(VkFormat format, VkExtent2D extent, VkImageUsageFlags usage)
{
  return VkImageCreateInfo{
      .sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext                 = nullptr,
      .flags                 = 0,
      .imageType             = VK_IMAGE_TYPE_2D,
      .format                = format,
      .extent                = {extent.width, extent.height, 1},
      .mipLevels             = 1,
      .arrayLayers           = 1,
      .samples               = VK_SAMPLE_COUNT_1_BIT,
      .tiling                = VK_IMAGE_TILING_OPTIMAL,
      .usage                 = usage,
      .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices   = nullptr,
      .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
  };
}

Framebuffer::Framebuffer(const vko::Device&   device,
                         vko::vma::Allocator& allocator,
                         VkCommandPool        commandPool,
                         VkQueue              queue,
                         SampleGlslCompiler&  glslCompiler,
                         glm::uvec2           vpSize)
    : m_size(vpSize)
    , m_colorHDR(device,
                 make2DImageCreateInfo(c_colorFormat,
                                       {vpSize.x, vpSize.y},
                                       VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                           | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT),
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 allocator)
    , m_colorLDR(device,
                 make2DImageCreateInfo(c_colorLDRFormat,
                                       {vpSize.x, vpSize.y},
                                       VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                           | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                           | VK_IMAGE_USAGE_STORAGE_BIT),
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 allocator)
    , m_depth(device,
              make2DImageCreateInfo(s_depthFormat,
                                    {vpSize.x, vpSize.y},
                                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                        | VK_IMAGE_USAGE_SAMPLED_BIT),
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
              allocator)
    , m_sampler(device,
                VkSamplerCreateInfo{
                    .sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    .pNext            = nullptr,
                    .flags            = 0,
                    .magFilter        = VK_FILTER_LINEAR,
                    .minFilter        = VK_FILTER_LINEAR,
                    .mipmapMode       = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                    .addressModeU     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    .addressModeV     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    .addressModeW     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    .mipLodBias       = 0.0f,
                    .anisotropyEnable = VK_FALSE,
                    .maxAnisotropy    = 1.0f,
                    .compareEnable    = VK_FALSE,
                    .compareOp        = VK_COMPARE_OP_NEVER,
                    .minLod           = 0.0f,
                    .maxLod           = 0.0f,
                    .borderColor      = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
                    .unnormalizedCoordinates = VK_FALSE,
                })
    , m_imguiTexture(m_sampler, m_colorLDR.view, VK_IMAGE_LAYOUT_GENERAL)
    , m_allocator(&allocator)
    , m_commandPool(commandPool)
    , m_queue(queue)
    , m_hiz(device, glslCompiler)
    , m_tonemap(device, glslCompiler)
{
  // Transition images from UNDEFINED to expected layouts
  {
    vko::simple::ImmediateCommandBuffer cmd(device, commandPool, queue);
    vko::ImageAccess undefined{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_IMAGE_LAYOUT_UNDEFINED};
    vko::ImageAccess general{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                             VK_IMAGE_LAYOUT_GENERAL};
    vko::ImageAccess depthAttachment{VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    vko::cmdImageBarrier(device, cmd, m_colorHDR.image, undefined, general);
    vko::cmdImageBarrier(device, cmd, m_colorLDR.image, undefined, general);
    vko::cmdImageBarrier(device, cmd, m_depth.image, undefined, depthAttachment,
                         VK_IMAGE_ASPECT_DEPTH_BIT);
  }

  initHiz(device, vpSize);
  m_tonemap->updateComputeDescriptorSets(renderHdrImageInfo(), displayLdrImageInfo());
}

Framebuffer::~Framebuffer()
{
  m_hiz.deinitUpdateViews(m_hizUpdate);
}

void Framebuffer::cmdTonemap(VkCommandBuffer cmd)
{
  m_tonemap->runCompute(cmd, {size().width, size().height});
  // Note: memoryBarrier requires device, but we don't have cmd available in this context
  // This barrier happens in the calling code's command buffer instead
}

void Framebuffer::renderUI()
{
  m_tonemap->onUI();
}

glm::vec4 Framebuffer::hizFarFactors() const
{
  glm::vec4 vec;
  m_hizUpdate.farInfo.getShaderFactors(glm::value_ptr(vec));
  return vec;
}

float Framebuffer::hizFarMax() const
{
  return m_hizUpdate.farInfo.getSizeMax();
}

void Framebuffer::initHiz(const vko::Device& device, glm::uvec2 vpSize)
{
  m_hiz.setupUpdateInfos(m_hizUpdate, vpSize.x, vpSize.y, s_depthFormat,
                         VK_IMAGE_ASPECT_DEPTH_BIT);

  // Create HiZ far image
  m_hizFar.emplace(device,
                   VkImageCreateInfo{
                       .sType     = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                       .pNext     = nullptr,
                       .flags     = 0,
                       .imageType = VK_IMAGE_TYPE_2D,
                       .format    = m_hizUpdate.farInfo.format,
                       .extent = {m_hizUpdate.farInfo.width, m_hizUpdate.farInfo.height, 1},
                       .mipLevels   = m_hizUpdate.farInfo.mipLevels,
                       .arrayLayers = 1,
                       .samples     = VK_SAMPLE_COUNT_1_BIT,
                       .tiling      = VK_IMAGE_TILING_OPTIMAL,
                       .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                       .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
                       .queueFamilyIndexCount = 0,
                       .pQueueFamilyIndices   = nullptr,
                       .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                   },
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, *m_allocator);

  m_hizUpdate.sourceImage = depthImage();
  m_hizUpdate.farImage    = m_hizFar->image;
  m_hizUpdate.nearImage   = VK_NULL_HANDLE;

  m_hiz.initUpdateViews(m_hizUpdate);
  m_hiz.updateDescriptorSet(m_hizUpdate, 0);

  // Initial resource transitions
  {
    vko::simple::ImmediateCommandBuffer cmd(device, m_commandPool, m_queue);

    VkImageMemoryBarrier barrier{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext         = nullptr,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout     = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = m_hizFar->image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS,
                             0, VK_REMAINING_ARRAY_LAYERS},
    };
    device.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                nullptr, 0, nullptr, 1, &barrier);
  }
}
