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
#include <lod_streaming_scene.hpp>
#include <lod_traverser.hpp>
#include <rasterize_device_host.h>
#include <renderer_common.hpp>
#include <sample_glsl_compiler.hpp>

struct RasterizeConfig
{
  shaders::RasterizeConfig shaders                = {.lodVisualization = 0};
  uint32_t                 maxDrawableClusterBits = 20;
};

class RasterizeRenderer
{
public:
  static constexpr bool requiresCLAS() { return false; }

  RasterizeRenderer(const RenderInitParams& params);

  void         updatedFrambuffer(const RenderParams& params);
  void         render(const RenderParams& params,
                      const SceneVK&      sceneVk,
                      vko::StagingStream<vko::vma::RecyclingStagingPool<vko::Device>>& staging);
  void         uiOverlay();
  void         uiInline(bool&, bool&) {}
  void         uiSection(bool& recreateRenderer, bool& resetFrameAccumulation);
  VkDeviceSize deviceMemoryUsage() const { return 0; }

  streaming::StreamingSceneVk&       streaming() { return *m_streaming; }
  const streaming::StreamingSceneVk& streaming() const { return *m_streaming; }

private:
  std::unique_ptr<streaming::StreamingSceneVk> m_streaming;
  static constexpr uint64_t                    MAX_CYCLES = 4;

  RasterizeConfig m_config;

  struct Drawing
  {
    std::optional<vko::BindingsAndFlags>        bindings;
    std::optional<vko::DescriptorSetLayout>     descriptorSetLayout;
    std::optional<vko::PipelineLayout>          pipelineLayout;
    std::optional<vko::GraphicsPipeline>        pipeline;
    std::optional<vko::SingleDescriptorSetPool> descriptorPool;
    std::optional<vko::DescriptorSet>           descriptorSet;
  } m_drawing;

  struct DrawingData
  {
    std::optional<vkobj::Buffer<shaders::RasterizeConstants>> constants;

    std::optional<vkobj::Buffer<shaders::DrawCluster>>           drawClusters;
    std::optional<vkobj::Buffer<shaders::DrawMeshTasksIndirect>> drawIndirect;
    std::optional<vkobj::Buffer<shaders::DrawStats>>             drawStats;
    std::optional<vkobj::Buffer<shaders::DrawStats>> drawStatsHostVisible;
    std::optional<vkobj::BufferMapping<shaders::DrawStats>> drawStatsMapping;
  } m_drawingData;

  struct Traversal
  {
    std::optional<vko::BindingsAndFlags>        bindings;
    std::optional<vko::DescriptorSetLayout>     descriptorSetLayout;
    std::optional<vko::PipelineLayout>          pipelineLayout;
    std::optional<vko::ComputePipeline>         traversePipeline;
    std::optional<vko::ComputePipeline>         traverseInitPipeline;
    std::optional<vko::ComputePipeline>         traverseVerifyPipeline;
    std::optional<vko::SingleDescriptorSetPool> descriptorPool;
    std::optional<vko::DescriptorSet>           descriptorSet;
  } m_traversal;

  struct TraversalData
  {
    std::optional<vkobj::Buffer<shaders::TraversalConstants>> constants;
    std::optional<vkobj::Buffer<shaders::EncodedNodeJob>>     nodeQueue;
    std::optional<vkobj::Buffer<shaders::EncodedClusterJob>>  clusterQueue;
    std::optional<vkobj::Buffer<shaders::JobStatus>>          jobStatus;
  } m_traversalData;

  uint64_t m_frame = 0;

  void initDrawingPipeline(const vko::Device&    device,
                           SampleGlslCompiler&   glslCompiler,
                           const RendererCommon& common,
                           const Scene&          scene,
                           const SceneVK&        sceneVk,
                           Framebuffer&          framebuffer);

  void initTraversalPipeline(const vko::Device&  device,
                             SampleGlslCompiler& glslCompiler,
                             Framebuffer&        framebuffer);

  void resizeTraversalData(const RenderParams& params, const SceneVK& sceneVk, VkCommandBuffer cmd);
  void resizeDrawingData(const RenderParams& params);
};
