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
#include <camera.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iomanip>
#include <lod_streaming_jobs.hpp>
#include <lod_traverser.hpp>
#include <memory>
#include <nvhiz_vk.hpp>
#include <nvvkhl/shaders/dh_sky.h>
#include <queue>
#include <sample_garbage.hpp>
#include <sample_profiler.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <shaders_frame_params.h>
#include <sstream>
#include <vko/imgui_objects.hpp>
#include <vko/object.hpp>
#include <vko/shortcuts.hpp>
#include <vulkan/vulkan_core.h>

inline std::string formatBytes(uint64_t bytes)
{
  constexpr const char* suffixes[]{"B",   "KiB", "MiB", "GiB",
                                   "TiB", "PiB", "EiB"};
  uint64_t              suffix = uint64_t(std::bit_width(bytes) / 10);
  float decimal = suffix > 0 ? float(bytes >> (suffix * 10 - 10)) / 1024.0f : float(bytes);
  std::ostringstream oss;  // TODO: replace with std::format() and make constexpr
  oss << std::setprecision(suffix > 0 ? 1 : 0) << std::fixed << decimal
      << suffixes[suffix];
  return oss.str();
}

class HiZ : public NVHizVK
{
public:
  HiZ(const vko::Device& device, SampleGlslCompiler& glslCompiler)
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

      m_hizShaders[i] =
          reloadUntilCompiling(device, glslCompiler, "nvhiz-update.comp.glsl",
                               shaderc_shader_kind::shaderc_compute_shader, &options);

      assert(m_hizShaders[i].has_value());

      shaderModules[i] = m_hizShaders[i].value();
    }

    initPipelines(shaderModules);
  }

  ~HiZ() { deinit(); }

private:
  std::array<std::optional<vkobj::ShaderModule>, NVHizVK::SHADER_COUNT> m_hizShaders;
};

namespace nvvkhl {
struct TonemapperPostProcess;
};

class TonemapPipeline
{
public:
  TonemapPipeline(const vko::Device& device, SampleGlslCompiler& glslCompiler);
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
  static const VkFormat c_colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  static const VkFormat c_colorLDRFormat = VK_FORMAT_R8G8B8A8_UNORM;  // Can't use SRGB with STORAGE_BIT
  static const VkFormat s_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;

  Framebuffer(const vko::Device&   device,
              vko::vma::Allocator& allocator,
              VkCommandPool        commandPool,
              VkQueue              queue,
              SampleGlslCompiler&  glslCompiler,
              glm::uvec2           vpSize);

  ~Framebuffer();

  void cmdTonemap(VkCommandBuffer cmd);
  void renderUI();

  VkExtent2D size() const { return {m_size.x, m_size.y}; }

  // HDR render target accessors
  VkImage               renderHdrImage() const { return m_colorHDR.image; }
  VkImageView           renderHdrView() const { return m_colorHDR.view; }
  VkDescriptorImageInfo renderHdrImageInfo() const
  {
    return {m_sampler, m_colorHDR.view, VK_IMAGE_LAYOUT_GENERAL};
  }

  // LDR display target accessors
  VkImage               displayLdrImage() const { return m_colorLDR.image; }
  VkImageView           displayLdrView() const { return m_colorLDR.view; }
  VkDescriptorImageInfo displayLdrImageInfo() const
  {
    return {m_sampler, m_colorLDR.view, VK_IMAGE_LAYOUT_GENERAL};
  }
  ImTextureID displayImGuiTexture() const { return m_imguiTexture; }

  // Depth accessors
  VkImage     depthImage() const { return m_depth.image; }
  VkImageView depthView() const { return m_depth.view; }

  HiZ&                         hiz() { return m_hiz; }
  const NVHizVK::Update&       hizUpdate() const { return m_hizUpdate; }
  const VkDescriptorImageInfo& hizFar() const
  {
    return m_hizUpdate.farImageInfo;
  }
  glm::vec4 hizFarFactors() const;
  float     hizFarMax() const;

  // None of this is copy/move safe
  Framebuffer(const Framebuffer& other)           = delete;
  Framebuffer operator=(const Framebuffer& other) = delete;

private:
  glm::uvec2                        m_size;
  vko::ViewedImage<>                m_colorHDR;  // HDR render target
  vko::ViewedImage<>                m_colorLDR;  // LDR display target
  vko::ViewedImage<>                m_depth;     // Depth buffer
  vko::Sampler                      m_sampler;
  vko::imgui::Texture               m_imguiTexture;
  vko::vma::Allocator*              m_allocator   = nullptr;
  VkCommandPool                     m_commandPool = VK_NULL_HANDLE;
  VkQueue                           m_queue       = VK_NULL_HANDLE;
  HiZ                               m_hiz;
  std::optional<vko::ViewedImage<>> m_hizFar;
  NVHizVK::Update                   m_hizUpdate;
  TonemapPipeline                   m_tonemap;
  void initHiz(const vko::Device& device, glm::uvec2 vpSize);
};

struct RendererConfig
{
  bool  useOcclusion        = false;
  bool  lockLodCamera       = false;
  float lodTargetPixelError = 1.0f;
};

struct RendererCommon
{
  RendererCommon(const vko::Device&              device,
                 vko::vma::Allocator&            allocator,
                 vko::shared_obj<RendererConfig> config);

  void cmdUpdateParams(const vko::Device& device,
                       Framebuffer&       framebuffer,
                       Camera&            camera,
                       float              maxWorldDiagonalInObjectSpace,
                       VkCommandBuffer    cmd);
  bool uiLod(const Scene& scene, const Framebuffer& framebuffer);
  bool uiSky();

  vko::shared_obj<RendererConfig>                    m_config;
  shaders::TraversalParams                           m_traversalParams;
  nvvkhl_shaders::SimpleSkyParameters                m_skyParams = {};
  vkobj::Buffer<shaders::FrameParams>                m_bFrameInfo;
  vkobj::Buffer<nvvkhl_shaders::SimpleSkyParameters> m_bSkyParams;
  shaders::FrameParams                               m_lastFrameInfo = {};

  // TODO: more of an RT-without-denoise-only thing
  uint32_t m_frameAccumIndex = 0;

  uint64_t m_frameIndex = 0;
};

// A non-owning parameter pack for creating the renderer and its streaming
struct RenderInitParams
{
  vkobj::Context&     context;
  SampleGlslCompiler& glslCompiler;

  // Renderer resources
  RendererCommon& common;
  const Scene&    scene;
  SceneVK&        sceneVk;  // non-const: streaming writes to it
  Framebuffer&    framebuffer;
  bool            dlssAvailable = false;

  // Streaming resources
  VkQueue               queue;
  VkCommandPool         initPool;
  vkobj::TimelineQueue& initQueue;
  vkobj::TimelineQueue& transferQueue;
  VkDeviceSize          streamingBufferSize;
  uint32_t              streamingMaxResidentGroups;
  bool                  streamingGreedyUnload;
  SampleProfiler&       profiler;
};


// Forward declare SampleVulkanContext
struct SampleVulkanContext;

// Trivial container for three commonly used queues
struct TimelineQueueContainer
{
  vkobj::TimelineQueue primary;
  vkobj::TimelineQueue compute;
  vkobj::TimelineQueue transfer;
  vkobj::TimelineQueue asyncLoad;

  TimelineQueueContainer(const SampleVulkanContext& vkContext);
};

// A non-owning parameter pack for rendering
struct RenderParams
{
  vkobj::Context&         context;
  RendererCommon&         common;
  Camera&                 camera;
  Framebuffer&            framebuffer;
  SampleProfiler&         profiler;
  std::queue<Garbage>&    garbage;
  TimelineQueueContainer& queueStates;
};

// Note: RendererInterface removed - use std::variant<RaytraceRenderer, RasterizeRenderer> instead
// Each renderer must provide these methods:
//   void         updatedFrambuffer(const RenderParams& params);
//   void         render(const RenderParams& params, const SceneVK& sceneVk, StagingStream& staging);
//   void         uiOverlay();
//   void         uiInline(bool& recreateRenderer, bool& resetFrameAccumulation);
//   void         uiSection(bool& recreateRenderer, bool& resetFrameAccumulation);
//   VkDeviceSize deviceMemoryUsage() const;
//   static constexpr bool requiresCLAS();
//   streaming::StreamingSceneVk& streaming();
