/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvvkhl/tonemap_postprocess.hpp>
#include <vulkan/vulkan_core.h>
#include <cassert>
#include <imgui.h>

namespace nvvkhl {

TonemapperPostProcess::~TonemapperPostProcess()
{
  if(m_device)
  {
    if(m_graphicsPipeline != VK_NULL_HANDLE)
    {
      m_device->vkDestroyPipeline(*m_device, m_graphicsPipeline, nullptr);
    }
    if(m_computePipeline != VK_NULL_HANDLE)
    {
      m_device->vkDestroyPipeline(*m_device, m_computePipeline, nullptr);
    }
  }
}

static VkExtent2D getGroupCounts(const VkExtent2D& size)
{
  constexpr uint32_t groupSize = 8;
  return {(size.width + groupSize - 1) / groupSize, (size.height + groupSize - 1) / groupSize};
}

void TonemapperPostProcess::createGraphicPipeline(const vko::Device& device,
                                                  VkFormat colorFormat,
                                                  VkFormat depthFormat,
                                                  std::span<const uint32_t> vertexShader,
                                                  std::span<const uint32_t> fragmentShader)
{
  m_mode   = TmMode::eGraphic;
  m_device = &device;

  // Create descriptor set layout for push descriptors
  m_graphicsBindings = vko::makeBindings({
      {{nvvkhl_shaders::eTonemapperInput, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, 0},
  });
  m_graphicsLayout = vko::makeDescriptorSetLayout(device, m_graphicsBindings, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

  // Create pipeline layout with push constants
  VkPushConstantRange pushConstantRange{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(nvvkhl_shaders::Tonemapper)};
  VkDescriptorSetLayout layoutHandle = static_cast<VkDescriptorSetLayout>(*m_graphicsLayout);
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext                  = nullptr,
      .flags                   = 0,
      .setLayoutCount         = 1,
      .pSetLayouts            = &layoutHandle,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstantRange,
  };
  m_graphicsPipelineLayout.emplace(device, pipelineLayoutInfo);

  // Create shader modules
  vko::ShaderModule vertexModule(device, VkShaderModuleCreateInfo{
                                               .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                               .pNext    = nullptr,
                                               .flags    = 0,
                                               .codeSize = vertexShader.size_bytes(),
                                               .pCode    = vertexShader.data(),
                                           });
  vko::ShaderModule fragmentModule(device, VkShaderModuleCreateInfo{
                                                .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                                .pNext    = nullptr,
                                                .flags    = 0,
                                                .codeSize = fragmentShader.size_bytes(),
                                                .pCode    = fragmentShader.data(),
                                            });

  // Create graphics pipeline
  VkPipelineShaderStageCreateInfo stages[] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = nullptr, .flags = 0, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertexModule, .pName = "main", .pSpecializationInfo = nullptr},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = nullptr, .flags = 0, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragmentModule, .pName = "main", .pSpecializationInfo = nullptr},
  };

  VkPipelineRenderingCreateInfo renderingInfo{
      .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .pNext                    = nullptr,
      .viewMask                 = 0,
      .colorAttachmentCount     = 1,
      .pColorAttachmentFormats  = &colorFormat,
      .depthAttachmentFormat    = depthFormat,
      .stencilAttachmentFormat  = VK_FORMAT_UNDEFINED,
  };

  VkPipelineVertexInputStateCreateInfo vertexInput{
      .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .pNext                           = nullptr,
      .flags                           = 0,
      .vertexBindingDescriptionCount   = 0,
      .pVertexBindingDescriptions      = nullptr,
      .vertexAttributeDescriptionCount = 0,
      .pVertexAttributeDescriptions    = nullptr,
  };

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .pNext                  = nullptr,
      .flags                  = 0,
      .topology                = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable  = VK_FALSE,
  };

  VkPipelineRasterizationStateCreateInfo rasterization{
      .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext                   = nullptr,
      .flags                   = 0,
      .depthClampEnable         = VK_FALSE,
      .rasterizerDiscardEnable  = VK_FALSE,
      .polygonMode              = VK_POLYGON_MODE_FILL,
      .cullMode                 = VK_CULL_MODE_NONE,
      .frontFace                = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .depthBiasEnable          = VK_FALSE,
      .depthBiasConstantFactor  = 0.0f,
      .depthBiasClamp           = 0.0f,
      .depthBiasSlopeFactor     = 0.0f,
      .lineWidth                 = 1.0f,
  };

  VkPipelineColorBlendAttachmentState colorBlendAttachment{
      .blendEnable         = VK_FALSE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
      .colorBlendOp        = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp        = VK_BLEND_OP_ADD,
      .colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  };

  VkPipelineColorBlendStateCreateInfo colorBlend{
      .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .pNext           = nullptr,
      .flags           = 0,
      .logicOpEnable    = VK_FALSE,
      .logicOp          = VK_LOGIC_OP_COPY,
      .attachmentCount  = 1,
      .pAttachments    = &colorBlendAttachment,
      .blendConstants  = {0.0f, 0.0f, 0.0f, 0.0f},
  };

  VkGraphicsPipelineCreateInfo pipelineInfo{
      .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext               = &renderingInfo,
      .flags                = 0,
      .stageCount           = 2,
      .pStages              = stages,
      .pVertexInputState    = &vertexInput,
      .pInputAssemblyState  = &inputAssembly,
      .pTessellationState   = nullptr,
      .pViewportState       = nullptr,
      .pRasterizationState  = &rasterization,
      .pMultisampleState    = nullptr,
      .pDepthStencilState   = nullptr,
      .pColorBlendState     = &colorBlend,
      .pDynamicState        = nullptr,
      .layout               = *m_graphicsPipelineLayout,
      .renderPass            = VK_NULL_HANDLE,
      .subpass              = 0,
      .basePipelineHandle   = VK_NULL_HANDLE,
      .basePipelineIndex     = -1,
  };

  if(device.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_graphicsPipeline) != VK_SUCCESS)
  {
    m_graphicsPipeline = VK_NULL_HANDLE;
  }
}

void TonemapperPostProcess::createComputePipeline(const vko::Device& device,
                                                  std::span<const uint32_t> computeShader)
{
  m_mode   = TmMode::eCompute;
  m_device = &device;

  // Create descriptor set layout for push descriptors
  m_computeBindings = vko::makeBindings({
      {{nvvkhl_shaders::eTonemapperInput, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 0},
      {{nvvkhl_shaders::eTonemapperOutput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 0},
  });
  m_computeLayout = vko::makeDescriptorSetLayout(device, m_computeBindings, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

  // Create pipeline layout with push constants
  VkPushConstantRange pushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(nvvkhl_shaders::Tonemapper)};
  VkDescriptorSetLayout layoutHandle = static_cast<VkDescriptorSetLayout>(*m_computeLayout);
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext                  = nullptr,
      .flags                   = 0,
      .setLayoutCount         = 1,
      .pSetLayouts            = &layoutHandle,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstantRange,
  };
  m_computePipelineLayout.emplace(device, pipelineLayoutInfo);

  // Create shader module
  vko::ShaderModule computeModule(device, VkShaderModuleCreateInfo{
                                               .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                               .pNext    = nullptr,
                                               .flags    = 0,
                                               .codeSize = computeShader.size_bytes(),
                                               .pCode    = computeShader.data(),
                                           });

  // Create compute pipeline
  VkPipelineShaderStageCreateInfo stage{
      .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext   = nullptr,
      .flags   = 0,
      .stage   = VK_SHADER_STAGE_COMPUTE_BIT,
      .module  = computeModule,
      .pName   = "main",
      .pSpecializationInfo = nullptr,
  };
  VkComputePipelineCreateInfo pipelineInfo{
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext   = nullptr,
      .flags   = 0,
      .stage   = stage,
      .layout  = *m_computePipelineLayout,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex  = -1,
  };

  if(device.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_computePipeline) != VK_SUCCESS)
  {
    m_computePipeline = VK_NULL_HANDLE;
  }
}

void TonemapperPostProcess::updateGraphicDescriptorSets(VkDescriptorImageInfo inImage)
{
  assert(m_mode == TmMode::eGraphic && m_graphicsLayout);
  m_iimage = inImage;
  vko::WriteDescriptorSetBuilder builder;
  builder.push_back<VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER>(
      VK_NULL_HANDLE,  // Push descriptor - not used
      m_graphicsBindings.bindings[0],
      0,
      m_iimage);
  m_writes.assign(builder.writes().begin(), builder.writes().end());
}

void TonemapperPostProcess::updateComputeDescriptorSets(VkDescriptorImageInfo inImage, VkDescriptorImageInfo outImage)
{
  assert(m_mode == TmMode::eCompute && m_computeLayout);
  m_iimage = inImage;
  m_oimage = outImage;
  vko::WriteDescriptorSetBuilder builder;
  builder.push_back<VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER>(
      VK_NULL_HANDLE,  // Push descriptor - not used
      m_computeBindings.bindings[0],
      0,
      m_iimage);
  builder.push_back<VK_DESCRIPTOR_TYPE_STORAGE_IMAGE>(
      VK_NULL_HANDLE,  // Push descriptor - not used
      m_computeBindings.bindings[1],
      0,
      m_oimage);
  m_writes.assign(builder.writes().begin(), builder.writes().end());
  
  // Fix up pointers to point to our member variables instead of builder's temporary storage
  m_writes[0].pImageInfo = &m_iimage;
  m_writes[1].pImageInfo = &m_oimage;
}

void TonemapperPostProcess::runGraphic(VkCommandBuffer cmd)
{
  assert(m_mode == TmMode::eGraphic && m_graphicsPipeline != VK_NULL_HANDLE && m_graphicsPipelineLayout);
  // Note: These are global Vulkan functions, not device methods
  m_device->vkCmdPushConstants(cmd, static_cast<VkPipelineLayout>(*m_graphicsPipelineLayout), VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof(nvvkhl_shaders::Tonemapper), &m_settings);
  m_device->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
  m_device->vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, static_cast<VkPipelineLayout>(*m_graphicsPipelineLayout), 0,
                            static_cast<uint32_t>(m_writes.size()), m_writes.data());
  m_device->vkCmdDraw(cmd, 3, 1, 0, 0);
}

void TonemapperPostProcess::runCompute(VkCommandBuffer cmd, const VkExtent2D& size)
{
  assert(m_mode == TmMode::eCompute && m_computePipeline != VK_NULL_HANDLE && m_computePipelineLayout);
  m_device->vkCmdPushConstants(cmd, static_cast<VkPipelineLayout>(*m_computePipelineLayout), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(nvvkhl_shaders::Tonemapper), &m_settings);
  m_device->vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, static_cast<VkPipelineLayout>(*m_computePipelineLayout), 0,
                            static_cast<uint32_t>(m_writes.size()), m_writes.data());
  m_device->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
  VkExtent2D group_counts = getGroupCounts(size);
  m_device->vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
}

bool TonemapperPostProcess::onUI()
{
  bool changed{false};

  const char* items[] = {"Filmic", "Uncharted 2", "Clip", "ACES", "AgX", "Khronos PBR"};

  changed |= ImGui::Combo("Method", &m_settings.method, items, IM_ARRAYSIZE(items));
  changed |= ImGui::Checkbox("Active", reinterpret_cast<bool*>(&m_settings.isActive));
  changed |= ImGui::SliderFloat("Exposure", &m_settings.exposure, 0.1F, 15.0F, "%.3f", ImGuiSliderFlags_Logarithmic);
  changed |= ImGui::SliderFloat("Brightness", &m_settings.brightness, 0.0F, 2.0F);
  changed |= ImGui::SliderFloat("Contrast", &m_settings.contrast, 0.0F, 2.0F);
  changed |= ImGui::SliderFloat("Saturation", &m_settings.saturation, 0.0F, 2.0F);
  changed |= ImGui::SliderFloat("Vignette", &m_settings.vignette, 0.0F, 1.0F);

  if(ImGui::SmallButton("reset"))
  {
    m_settings = nvvkhl_shaders::defaultTonemapper();
    changed    = true;
  }
  return changed;
}

}  // namespace nvvkhl
