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

#include <acceleration_structures.hpp>
#include <bit>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui/imgui_helper.h>
#include <iomanip>
#include <lod_streaming_jobs.hpp>
#include <lod_traverser.hpp>
#include <memory>
#include <nvh/cameramanipulator.hpp>
#include <nvhiz_vk.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvkhl/shaders/dh_sky.h>
#include <queue>
#include <sample_garbage.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <shaders/shaders_frame_params.h>
#include <sstream>
#include <vulkan/vulkan_core.h>

inline std::string formatBytes(uint64_t bytes)
{
  constexpr const char* suffixes[]{"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};
  uint64_t              suffix  = std::bit_width(bytes) / 10llu;
  float                 decimal = suffix > 0 ? float(bytes >> (suffix * 10 - 10)) / 1024.0f : float(bytes);
  std::ostringstream    oss;  // TODO: replace with std::format() and make constexpr
  oss << std::setprecision(suffix > 0 ? 1 : 0) << std::fixed << decimal << suffixes[suffix];
  return oss.str();
}

class HiZ : public NVHizVK
{
public:
  HiZ(VkDevice device, SampleGlslCompiler& glslCompiler)
  {
    NVHizVK::Config config;
    config.msaaSamples             = 0;
    config.reversedZ               = false;
    config.supportsMinmaxFilter    = true;
    config.supportsSubGroupShuffle = true;

    init(device, config, 1);

    VkShaderModule shaderModules[NVHizVK::SHADER_COUNT];
    for(uint32_t i = 0; i < NVHizVK::SHADER_COUNT; i++)
    {
      shaderc::CompileOptions options = glslCompiler.defaultOptions();
      appendShaderDefines(i, options);

      m_hizShaders[i] = reloadUntilCompiling(device, glslCompiler, "nvhiz-update.comp.glsl",
                                             shaderc_shader_kind::shaderc_compute_shader, &options);

      assert(m_hizShaders[i]);

      shaderModules[i] = m_hizShaders[i];
    }

    initPipelines(shaderModules);
  }

  ~HiZ() { deinit(); }

private:
  vkobj::ShaderModule m_hizShaders[NVHizVK::SHADER_COUNT];
};

namespace nvvkhl {
struct TonemapperPostProcess;
};

class TonemapPipeline
{
public:
  TonemapPipeline(VkDevice device, nvvk::ResourceAllocator* allocator);
  ~TonemapPipeline();
  nvvkhl::TonemapperPostProcess* operator->() { return m_tonemapper.get(); }
  nvvkhl::TonemapperPostProcess& operator*() { return *m_tonemapper; }

  // Usage:
  // ->updateComputeDescriptorSets(inputImage, outputImage);
  // ->runCompute(cmd, {width, height});

private:
  // not copy/move-safe
  std::unique_ptr<nvvkhl::TonemapperPostProcess> m_tonemapper;
};

class Framebuffer
{
public:
  static const VkFormat c_colorFormat    = VK_FORMAT_R16G16B16A16_SFLOAT;
  static const VkFormat c_colorLDRFormat = VK_FORMAT_R8G8B8A8_SRGB;  // TODO: can imgui + nvpro_core do sRGB here??
  static const VkFormat s_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;

  Framebuffer(ResourceAllocator* allocator, SampleGlslCompiler& glslCompiler, TonemapPipeline& persistantTonemap, glm::uvec2 vpSize);

  ~Framebuffer();

  void cmdTonemap(VkCommandBuffer cmd);
  void tonemapUI();

  VkExtent2D             size() const { return m_gBuffer.getSize(); }
  VkImage                renderHdrImage() const { return m_gBuffer.getColorImage(0); }
  VkImageView            renderHdrView() const { return m_gBuffer.getColorImageView(0); }
  VkDescriptorImageInfo  renderHdrImageInfo() const { return m_gBuffer.getDescriptorImageInfo(0); }
  VkDescriptorSet        renderHdrDescriptorSet() const { return m_gBuffer.getDescriptorSet(0); }  // for rendering
  VkImage                displayLdrImage() const { return m_gBuffer.getColorImage(1); }
  VkImageView            displayLdrView() const { return m_gBuffer.getColorImageView(1); }
  VkDescriptorImageInfo  displayLdrImageInfo() const { return m_gBuffer.getDescriptorImageInfo(1); }
  VkDescriptorSet        displayLdrDescriptorSet() const { return m_gBuffer.getDescriptorSet(1); }  // to display
  const nvvkhl::GBuffer& gbuffer() const { return m_gBuffer; }

  HiZ&                         hiz() { return m_hiz; }
  const NVHizVK::Update&       hizUpdate() const { return m_hizUpdate; }
  const VkDescriptorImageInfo& hizFar() const { return m_hizUpdate.farImageInfo; }
  glm::vec4                    hizFarFactors() const;
  float                        hizFarMax() const;

  // None of this is copy/move safe
  Framebuffer(const Framebuffer& other) = delete;
  Framebuffer operator=(const Framebuffer& other) = delete;

private:
  nvvkhl::GBuffer    m_gBuffer;
  ResourceAllocator* m_allocator = nullptr;
  HiZ                m_hiz;
  nvvk::Image        m_imgHizFar = {};
  NVHizVK::Update    m_hizUpdate;
  TonemapPipeline& m_tonemap;  // tonemapper is owned externally so its config/UI persists between resizes (TODO: move outside)
  void initHiz(glm::uvec2 vpSize);
};

struct RendererConfig
{
  bool  useOcclusion        = false;
  bool  lockLodCamera       = false;
  float lodTargetPixelError = 1.0f;
};

struct RendererCommon
{
  RendererCommon(ResourceAllocator* allocator, SampleGlslCompiler& glslCompiler, VkCommandPool initPool, VkQueue initQueue, const SceneVK& sceneVk);

  void cmdUpdateParams(Framebuffer& framebuffer, nvh::CameraManipulator& camera, float maxWorldDiagonalInObjectSpace, VkCommandBuffer cmd);
  bool uiLod(const Scene& scene, const Framebuffer& framebuffer);
  bool uiSky();

  RendererConfig                                     m_config;
  shaders::TraversalParams                           m_traversalParams;
  nvvkhl_shaders::SimpleSkyParameters                m_skyParams = {};
  vkobj::Buffer<shaders::FrameParams>                m_bFrameInfo;
  vkobj::Buffer<nvvkhl_shaders::SimpleSkyParameters> m_bSkyParams;
  shaders::FrameParams                               m_lastFrameInfo = {};

  // TODO: more of an RT-without-denoise-only thing
  uint32_t m_frameAccumIndex = 0;

  uint64_t m_frameIndex = 0;
};

// A non-owning parameter pack for creating or updating the renderer
struct RenderInitParams
{
  VkInstance          instance;
  vkobj::Context&     context;
  SampleGlslCompiler& glslCompiler;
  RendererCommon&     common;
  const Scene&        scene;
  const SceneVK&      sceneVk;
  Framebuffer&        framebuffer;
};

// Trivial container for three commonly used queues
struct TimelineQueueContainer
{
  vkobj::TimelineQueue primary;
  vkobj::TimelineQueue compute;
  vkobj::TimelineQueue transfer;

  TimelineQueueContainer(nvvkhl::Application* app)
      : primary(app->getDevice(), app->getQueue(0).familyIndex, app->getQueue(0).queue)
      , compute(app->getDevice(), app->getQueue(1).familyIndex, app->getQueue(1).queue)
      , transfer(app->getDevice(), app->getQueue(2).familyIndex, app->getQueue(2).queue)
  {
  }
};

// A non-owning parameter pack for rendering
struct RenderParams
{
  VkInstance              instance;
  vkobj::Context&         context;
  RendererCommon&         common;
  Framebuffer&            framebuffer;
  nvvk::ProfilerVK&       profiler;
  std::queue<Garbage>&    garbage;
  TimelineQueueContainer& queueStates;
};

class RendererInterface
{
public:
  virtual ~RendererInterface()                                                                         = default;
  virtual void         updatedFrambuffer(const RenderParams& params)                                   = 0;
  virtual void         render(const RenderParams& params, const SceneVK& sceneVk, VkCommandBuffer cmd) = 0;
  virtual void         uiOverlay()                                                                     = 0;
  virtual void         uiInline(bool& recreateRenderer, bool& resetFrameAccumulation)                  = 0;
  virtual void         uiSection(bool& recreateRenderer, bool& resetFrameAccumulation)                 = 0;
  virtual VkDeviceSize deviceMemoryUsage() const                                                       = 0;
  virtual bool         requiresCLAS() const                                                            = 0;
};
