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
#include <sample_vulkan_objects.hpp>

RendererCommon::RendererCommon(ResourceAllocator* allocator, SampleGlslCompiler&, VkCommandPool, VkQueue, const SceneVK&)
    : m_traversalParams(initialTraversalParams())
    , m_skyParams(nvvkhl_shaders::initSimpleSkyParameters())
    , m_bFrameInfo(allocator, 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    , m_bSkyParams(allocator, 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
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

void RendererCommon::cmdUpdateParams(Framebuffer& framebuffer, nvh::CameraManipulator& camera, float maxWorldDiagonalInObjectSpace, VkCommandBuffer cmd)
{
  auto  imageSize       = framebuffer.size();
  float viewAspectRatio = float(imageSize.width) / float(imageSize.height);

  // Update the uniform buffer containing frame info
  shaders::FrameParams frameInfo{};
  const auto&          clip = camera.getClipPlanes();
  frameInfo.view            = camera.getMatrix();
  frameInfo.proj            = glm::perspectiveRH_ZO(glm::radians(camera.getFov()), viewAspectRatio, clip.x, clip.y);
  frameInfo.proj[1][1] *= -1;
  frameInfo.projInv         = glm::inverse(frameInfo.proj);
  frameInfo.viewInv         = glm::inverse(frameInfo.view);
  frameInfo.viewProj        = frameInfo.proj * frameInfo.view;
  frameInfo.viewLast        = m_lastFrameInfo.view;
  frameInfo.camPos          = camera.getEye();
  vkCmdUpdateBuffer(cmd, m_bFrameInfo, 0, sizeof(frameInfo), &frameInfo);

  // Update the sky
  vkCmdUpdateBuffer(cmd, m_bSkyParams, 0, sizeof(m_skyParams), &m_skyParams);

  // first frame use same
  if(!m_frameIndex)
    m_lastFrameInfo = frameInfo;

  // Update traversal parameters
  if(!m_frameIndex || !m_config.lockLodCamera)
  {
    m_traversalParams.viewTransform = frameInfo.view;
    m_traversalParams.viewPosition  = glm::vec3(frameInfo.viewInv[3]);
    m_traversalParams.hizViewProj   = m_lastFrameInfo.viewProj;
  }

  float errorOverDistanceThreshold =
      nvclusterlodErrorOverDistance(m_config.lodTargetPixelError, glm::radians(camera.getFov()), float(imageSize.height));

  m_traversalParams.errorOverDistanceThreshold = errorOverDistanceThreshold;
  m_traversalParams.distanceToUNorm32 =
      float(double(std::numeric_limits<uint32_t>::max()) / double(maxWorldDiagonalInObjectSpace));
  m_traversalParams.useOcclusion      = m_config.useOcclusion ? 1 : 0;
  m_traversalParams.hizSizeFactors    = framebuffer.hizFarFactors();
  m_traversalParams.hizSizeMax        = framebuffer.hizFarMax();
  m_traversalParams.hizViewport       = glm::vec2(framebuffer.size().width, framebuffer.size().height);

  m_lastFrameInfo = frameInfo;
  m_frameIndex++;
}

bool RendererCommon::uiLod(const Scene&, const Framebuffer&)
{
  using namespace ImGuiH;
  bool changed = false;

  ImGui::Text("Level of Detail (View)");
  PropertyEditor::begin();
  changed = PropertyEditor::entry(
                "Max. Pixel Error",
                [&] { return ImGui::SliderFloat("Max. Pixel Error", &m_config.lodTargetPixelError, 0.5f, 1000.0f); })
            || changed;
  PropertyEditor::entry("Lock Camera", [&] { return ImGui::Checkbox("Lock Camera", &m_config.lockLodCamera); });
  PropertyEditor::entry("Use Occlusion", [&] { return ImGui::Checkbox("Use Occlusion", &m_config.useOcclusion); });
  PropertyEditor::end();

  return changed;
}

bool RendererCommon::uiSky()
{
  using namespace ImGuiH;
  bool changed = false;
  ImGui::Text("Sun Orientation");
  PropertyEditor::begin();
  glm::vec3 dir                = m_skyParams.directionToLight;
  changed =
      PropertyEditor::entry("Sun Brightness",
                            [&] { return ImGui::SliderFloat("Sun Brightness", &m_skyParams.sunIntensity, 0.0f, 100.0f); })
      || changed;
  changed                      = ImGuiH::azimuthElevationSliders(dir, false) || changed;
  m_skyParams.directionToLight = dir;
  m_skyParams.lightRadiance    = m_skyParams.sunColor * m_skyParams.sunIntensity;
  PropertyEditor::end();
  return changed;
}

TonemapPipeline::TonemapPipeline(VkDevice device, nvvk::ResourceAllocator* allocator)
    : m_tonemapper(std::make_unique<nvvkhl::TonemapperPostProcess>(device, allocator))
{
  m_tonemapper->createComputePipeline();
}

TonemapPipeline::~TonemapPipeline() {}

Framebuffer::Framebuffer(ResourceAllocator* allocator, SampleGlslCompiler& glslCompiler, TonemapPipeline& persistantTonemap, glm::uvec2 vpSize)
    : m_gBuffer(allocator->getDevice(), allocator, VkExtent2D{vpSize.x, vpSize.y}, {c_colorFormat, c_colorLDRFormat}, s_depthFormat)
    , m_allocator(allocator)
    , m_hiz(allocator->getDevice(), glslCompiler)
    , m_tonemap(persistantTonemap)
{
  initHiz(vpSize);
  m_tonemap->updateComputeDescriptorSets(m_gBuffer.getDescriptorImageInfo(0), m_gBuffer.getDescriptorImageInfo(1));
}

Framebuffer::~Framebuffer()
{
  m_hiz.deinitUpdateViews(m_hizUpdate);
  m_allocator->destroy(m_imgHizFar);
}

void Framebuffer::cmdTonemap(VkCommandBuffer cmd)
{
  m_tonemap->runCompute(cmd, {size().width, size().height});
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

void Framebuffer::tonemapUI()
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

void Framebuffer::initHiz(glm::uvec2 vpSize)
{
  m_hiz.setupUpdateInfos(m_hizUpdate, vpSize.x, vpSize.y, s_depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

  // hiz
  VkImageCreateInfo hizImageInfo = {};
  hizImageInfo.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  hizImageInfo.imageType         = VK_IMAGE_TYPE_2D;
  hizImageInfo.format            = m_hizUpdate.farInfo.format;
  hizImageInfo.extent.width      = m_hizUpdate.farInfo.width;
  hizImageInfo.extent.height     = m_hizUpdate.farInfo.height;
  hizImageInfo.mipLevels         = m_hizUpdate.farInfo.mipLevels;
  hizImageInfo.extent.depth      = 1;
  hizImageInfo.arrayLayers       = 1;
  hizImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
  hizImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
  hizImageInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  hizImageInfo.flags             = 0;
  hizImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;


  m_imgHizFar = m_allocator->createImage(hizImageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  m_hizUpdate.sourceImage = gbuffer().getDepthImage();
  m_hizUpdate.farImage    = m_imgHizFar.image;
  m_hizUpdate.nearImage   = VK_NULL_HANDLE;

  m_hiz.initUpdateViews(m_hizUpdate);
  m_hiz.updateDescriptorSet(m_hizUpdate, 0);

  // initial resource transitions

  // fixme gbuffer and this should do it differently
  nvvk::CommandPool cpool(m_allocator->getDevice(), 0);
  VkCommandBuffer   cmd = cpool.createCommandBuffer();
  nvvk::cmdBarrierImageLayout(cmd, m_imgHizFar.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
  cpool.submitAndWait(cmd);
}
