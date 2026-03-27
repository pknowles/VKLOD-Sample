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

#include <lod_streaming_scene.hpp>
#include <lod_traverser.hpp>
#include <pathtrace_device_host.h>
#include <renderer_common.hpp>
#include <sample_glsl_compiler.hpp>
#include <vko/bindings.hpp>
#include <vko/nv_ngx.hpp>
#include <vko/object.hpp>
#include <vko/ray_tracing.hpp>
#include <vko/shortcuts.hpp>

struct GBufferRT
{
  GBufferRT(const vko::Device&        device,
            vko::vma::Allocator&      allocator,
            VkCommandBuffer           cmd,
            glm::uvec2                size,
            std::span<const VkFormat> colorFormats);

  VkExtent2D size() const { return {m_size.x, m_size.y}; }
  glm::uvec2 getSize() const { return m_size; }
  VkImage    getColorImage(size_t index) const
  {
    return m_colorImages[index].image;
  }
  VkImageView getColorImageView(size_t index) const
  {
    return m_colorImages[index].view;
  }
  VkFormat getColorFormat(size_t index) const { return m_colorFormats[index]; }
  VkDescriptorImageInfo getDescriptorImageInfo(size_t index) const
  {
    return {.sampler     = VK_NULL_HANDLE,
            .imageView   = m_colorImages[index].view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  }

private:
  glm::uvec2                      m_size;
  std::vector<vko::ViewedImage<>> m_colorImages;
  std::vector<VkFormat>           m_colorFormats;
};

// App-specific container for the raytracing pipeline and shader binding table
class PathtracingPipeline
{
public:
  PathtracingPipeline(SampleGlslCompiler&  glslCompiler,
                      const vko::Instance& instance,
                      VkPhysicalDevice     physicalDevice,
                      const vko::Device&   device,
                      vko::vma::Allocator& allocator,
                      VkCommandPool        commandPool,
                      VkQueue              queue,
                      uint32_t             textureCount);

  // Writes all bindings except for the gBuffer
  void writeDescriptorSetInitial(VkAccelerationStructureKHR tlas,
                                 VkBuffer                   frameInfo,
                                 VkBuffer                   skyParams,
                                 std::span<const VkDescriptorImageInfo> textures,
                                 VkDescriptorSet descriptorSet) const;

  // Write just the gBuffer bindings
  // DANGER: it's easy to forget to update the descriptorset when the framebuffer
  // changes, effectively leaving a dangling pointer
  void writeDescriptorSetFramebuffer(const GBufferRT& gBuffer,
                                     std::optional<VkDescriptorImageInfo> passthroughColor,
                                     VkDescriptorSet descriptorSet) const;

  void trace(VkCommandBuffer                   cmd,
             std::vector<VkDescriptorSet>      descriptorSets,
             const shaders::PathtraceConstant& pushConstant,
             glm::uvec2                        vpSize) const;

  VkDescriptorSetLayout descriptorSetLayout() const
  {
    return m_descriptorSetLayout;
  }
  const vko::BindingsAndFlags& bindings() const { return m_bindings; }

private:
  const vko::Device*                        m_device = nullptr;
  vko::BindingsAndFlags                     m_bindings;
  vko::DescriptorSetLayout                  m_descriptorSetLayout;
  std::optional<vko::PipelineLayout>        m_pipelineLayout;
  std::optional<vko::RayTracingPipelineKHR> m_pipeline;
  std::unique_ptr<vko::simple::ShaderBindingTables<vko::vma::Allocator>> m_sbt;
};

enum class DlssQuality : int32_t
{
  Disabled         = 0,
  MaxQuality       = 1,
  Balanced         = 2,
  Performance      = 3,
  UltraPerformance = 4
};

// Runtime mutable parameters for rendering
struct RaytraceConfig
{
  shaders::PathtraceConfig shaders = {
      .lodVisualization = VISUALIZE_CLUSTER_LOD,
      .sampleCountPixel = 1,
      .sampleCountAO    = 4,
      .maxDepth         = std::min(3, PATHTRACE_MAX_RGEN_RECURSION_DEPTH),
      .aoRadius         = 1000.0f,
      .pathtrace        = int32_t(true),
      .fogHeightOffset  = 0.0f,
      .fogDensity       = 0.001f,
  };
  DlssQuality dlssQuality          = DlssQuality::Balanced;
  bool        perInstanceTraversal = false;
  float memoryReserveScale = 7.0f;  // multiple of unique BLAS memory usage if perInstanceTraversal is true
  bool showGBufferDebug = false;
};

// Single instance ray tracing pipeline and LOD traverser, which has temporary
// job queue storage. This is somewhat scene dependent due to it supporting
// specific scene maximums (e.g. max. clusters per BLAS). It is re-created when
// the scene changes.
// GBuffer for ray tracing with multiple color attachments (for DLSS-RR)
class RaytraceRenderer
{
public:
  static constexpr bool requiresCLAS() { return true; }

  RaytraceRenderer(const RenderInitParams& params, vko::shared_obj<RaytraceConfig> config);

  void         updatedFrambuffer(const RenderParams& params);
  void         render(const RenderParams& params,
                      const SceneVK&      sceneVk,
                      vko::StagingStream<vko::vma::RecyclingStagingPool<vko::Device>>& staging);
  void         uiOverlay();
  void         uiInline(bool& recreateRenderer, bool& resetFrameAccumulation);
  void         uiSection(bool& recreateRenderer, bool& resetFrameAccumulation);
  VkDeviceSize deviceMemoryUsage() const;

  streaming::StreamingSceneVk&       streaming() { return *m_streaming; }
  const streaming::StreamingSceneVk& streaming() const { return *m_streaming; }

  // Default is not safe due to member dependency order
  RaytraceRenderer operator=(RaytraceRenderer&& other) = delete;

private:
  std::unique_ptr<streaming::StreamingSceneVk> m_streaming;
  vko::shared_obj<RaytraceConfig> m_config;  // shared to persist across recreation
  std::optional<LodInstanceTraverser> m_lodInstanceTraverser;  // must be one but not both (alt: use std::variant)
  std::optional<LodMeshTraverser> m_lodMeshTraverser;  // must be one but not both (alt: use std::variant)
  PathtracingPipeline                        m_rtPipeline;
  vko::SingleDescriptorSetPool               m_descriptorPool;
  vko::DescriptorSet                         m_rtDescriptorSet;
  std::optional<shaders::TraverseStats>      m_lastTraverseStats;
  SceneCounts                                m_sceneCounts;
  AABB                                       m_sceneAabb;
  vko::ngx::CapabilityParameter              m_ngxParameter;
  std::optional<vko::ngx::RayReconstruction> m_dlssRR;
  std::optional<GBufferRT>                   m_gBuffer;
  bool                                       m_gBufferStale = true;

  void blitGBufferDebugVisualization(const vko::Device& device,
                                     VkCommandBuffer    cmd,
                                     VkImage            outputImage,
                                     VkExtent2D         outputSize);
};
