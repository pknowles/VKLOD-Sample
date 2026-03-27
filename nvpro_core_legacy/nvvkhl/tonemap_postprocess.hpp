/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <nvvkhl/shaders/dh_tonemap.h>
#include <vko/bindings.hpp>
#include <vko/handles.hpp>
#include <optional>
#include <span>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace nvvkhl {

// Tonemapper using vulkan_objects with RAII design
struct TonemapperPostProcess
{
  TonemapperPostProcess() = default;
  ~TonemapperPostProcess();

  TonemapperPostProcess(const TonemapperPostProcess&) = delete;
  TonemapperPostProcess& operator=(const TonemapperPostProcess&) = delete;
  TonemapperPostProcess(TonemapperPostProcess&&) = default;
  TonemapperPostProcess& operator=(TonemapperPostProcess&&) = default;

  // Create graphics pipeline for tonemapping
  void createGraphicPipeline(const vko::Device&        device,
                             VkFormat                  colorFormat,
                             VkFormat                  depthFormat,
                             std::span<const uint32_t> vertexShader,
                             std::span<const uint32_t> fragmentShader);

  // Create compute pipeline for tonemapping
  void createComputePipeline(const vko::Device& device, std::span<const uint32_t> computeShader);

  // Update descriptor sets for graphics (push descriptors)
  void updateGraphicDescriptorSets(VkDescriptorImageInfo inImage);

  // Update descriptor sets for compute (push descriptors)
  void updateComputeDescriptorSets(VkDescriptorImageInfo inImage, VkDescriptorImageInfo outImage);

  // Execute graphics tonemapping
  void runGraphic(VkCommandBuffer cmd);

  // Execute compute tonemapping
  void runCompute(VkCommandBuffer cmd, const VkExtent2D& size);

  // Display UI for tonemapper settings
  bool onUI();

  void                        setSettings(const nvvkhl_shaders::Tonemapper& settings) { m_settings = settings; }
  nvvkhl_shaders::Tonemapper& settings() { return m_settings; }

private:
  // Graphics pipeline resources
  vko::BindingsAndFlags                  m_graphicsBindings;
  std::optional<vko::DescriptorSetLayout> m_graphicsLayout;
  std::optional<vko::PipelineLayout>     m_graphicsPipelineLayout;
  VkPipeline                              m_graphicsPipeline{VK_NULL_HANDLE};
  const vko::Device*                      m_device{nullptr};

  // Compute pipeline resources
  vko::BindingsAndFlags                  m_computeBindings;
  std::optional<vko::DescriptorSetLayout> m_computeLayout;
  std::optional<vko::PipelineLayout>     m_computePipelineLayout;
  VkPipeline                              m_computePipeline{VK_NULL_HANDLE};

  nvvkhl_shaders::Tonemapper m_settings{nvvkhl_shaders::defaultTonemapper()};

  // Push descriptor writes (built on update, used in run)
  VkDescriptorImageInfo             m_iimage{};
  VkDescriptorImageInfo             m_oimage{};
  std::vector<VkWriteDescriptorSet> m_writes;

  enum class TmMode
  {
    eNone,
    eGraphic,
    eCompute
  } m_mode{TmMode::eNone};
};

}  // namespace nvvkhl
