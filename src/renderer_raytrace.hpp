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

#include <lod_traverser.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvvk/sbtwrapper_vk.hpp>
#include <renderer_common.hpp>
#include <sample_dlss_objects.hpp>
#include <sample_glsl_compiler.hpp>
#include <shaders/pathtrace_device_host.h>

// Self-destroying shader binding table
class SBT : public nvvk::SBTWrapper
{
public:
  using nvvk::SBTWrapper::SBTWrapper;
  ~SBT() { destroy(); }

  // nvvk::SBTWrapper is not copy or move-safe
  SBT(const SBT& other)            = delete;
  SBT& operator=(const SBT& other) = delete;
};

// App-specific container for the raytracing pipeline and shader binding table
class PathtracingPipeline
{
public:
  PathtracingPipeline(SampleGlslCompiler&                glslCompiler,
                      ResourceAllocator*                 allocator,
                      uint32_t                           queueGCT,
                      std::vector<VkDescriptorSetLayout> descriptorSetLayouts);

  static std::unique_ptr<nvvk::DescriptorSetContainer> makeDescriptorSet(VkDevice device, uint32_t textureCount);

  // Writes all bindings except for the gBuffer
  static void writeDescriptorSetInitial(VkDevice                               device,
                                        VkAccelerationStructureKHR             tlas,
                                        VkBuffer                               frameInfo,
                                        VkBuffer                               skyParams,
                                        std::span<const VkDescriptorImageInfo> textures,
                                        nvvk::DescriptorSetContainer&          descriptorSet);

  // Write just the gBuffer bindings
  // DANGER: it's easy to forget to update the descriptorset when the framebuffer
  // changes, effectively leaving a dangling pointer
  static void writeDescriptorSetFramebuffer(VkDevice                             device,
                                            const nvvkhl::GBuffer&               gBuffer,
                                            std::optional<VkDescriptorImageInfo> passthroughColor,
                                            nvvk::DescriptorSetContainer&        descriptorSet);

  void trace(VkCommandBuffer                   cmd,
             std::vector<VkDescriptorSet>      descriptorSets,
             const shaders::PathtraceConstant& pushConstant,
             glm::uvec2                        vpSize) const;

private:
  vkobj::PipelineLayout m_pipelineLayout;
  vkobj::Pipeline       m_pipeline;
  std::unique_ptr<SBT>  m_sbt;
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
  float       memoryReserveScale    = 7.0f;  // multiple of unique BLAS memory usage if perInstanceTraversal is true
  bool        showGBufferDebug     = false;
  uint32_t    maxTotalClusterGroups = 0;
};

// Single instance ray tracing pipeline and LOD traverser, which has temporary
// job queue storage. This is somewhat scene dependent due to it supporting
// specific scene maximums (e.g. max. clusters per BLAS). It is re-created when
// the scene changes.
class RaytraceRenderer : public RendererInterface
{
public:
  RaytraceRenderer(const RenderInitParams& params, RaytraceConfig& config);
  virtual void         updatedFrambuffer(const RenderParams& params) override;
  virtual void         render(const RenderParams& params, const SceneVK& sceneVk, VkCommandBuffer cmd) override;
  virtual void         uiOverlay() override;
  virtual void         uiInline(bool& recreateRenderer, bool& resetFrameAccumulation) override;
  virtual void         uiSection(bool& recreateRenderer, bool& resetFrameAccumulation) override;
  virtual VkDeviceSize deviceMemoryUsage() const override;  // nvvk::GBuffer not included - no API to query it
  virtual bool         requiresCLAS() const override { return true; }

private:
  RaytraceConfig&                     m_config;  // owned by app to persist across recreation. TODO: shared_ptr
  std::optional<LodInstanceTraverser> m_lodInstanceTraverser;  // must be one but not both (alt: use std::variant)
  std::optional<LodMeshTraverser>     m_lodMeshTraverser;      // must be one but not both (alt: use std::variant)
  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtBinding;
  PathtracingPipeline                           m_rtPipeline;
  std::optional<shaders::TraverseStats>         m_lastTraverseStats;
  SceneCounts                                   m_sceneCounts;
  AABB                                          m_sceneAabb;
  ngx::CapabilityParameter                      m_ngxParameter;
  std::optional<ngx::RayReconstruction>         m_dlssRR;
  std::unique_ptr<nvvkhl::GBuffer>              m_gBuffer;
  bool                                          m_gBufferStale = true;
  TonemapPipeline                               m_tonemap;

  void blitGBufferDebugVisualization(VkCommandBuffer cmd, VkImage outputImage, VkExtent2D outputSize);
};
