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

#include <chrono>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <print>
#include <sample_vulkan_objects.hpp>
#include <shaderc/shaderc.hpp>
#include <stdexcept>
#include <thread>
#include <variant>
#include <vko/bindings.hpp>
#include <vko/shaderc_compiler.hpp>
#include <vulkan/vulkan_core.h>

// Returns the ceiling of an integer division. Assumes positive values!
template <std::integral T>
T div_ceil(const T& a, const T& b)
{
  return (a + b - 1) / b;
}

// A shaderc compiler with include dirs and default options, similar to
// nvvk::GlslCompiler but without using nvh::findFile
class SampleGlslCompiler
{
public:
  SampleGlslCompiler(std::initializer_list<std::filesystem::path> includeDirs)
      : m_includeDirs(includeDirs)
  {
    m_options.SetIncluder(std::make_unique<vko::shaderc::FileIncluder>(
        std::span<const std::filesystem::path>(includeDirs)));
    m_options.SetTargetSpirv(shaderc_spirv_version_1_6);
    m_options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
#ifndef NDEBUG
    m_options.SetGenerateDebugInfo();
#endif
  }
  std::filesystem::path find(const std::filesystem::path& filename)
  {
    for(const auto& dir : m_includeDirs)
    {
      auto candidate = dir / filename;
      if(std::filesystem::exists(candidate))
        return candidate;
    }
    std::println("Failed to find '{}'. Searched:", filename.string());
    for(const auto& dir : m_includeDirs)
      std::println(" - {}", dir.string());
    return {};
  }

  template <vko::device_and_commands DeviceAndCommands>
  vkobj::ShaderModule compile(const DeviceAndCommands&       device,
                              const std::filesystem::path&   path,
                              shaderc_shader_kind            shaderKind,
                              const shaderc::CompileOptions* options = nullptr)
  {
    std::ifstream file(path, std::ios::binary);
    if(!file)
      throw std::runtime_error("Failed to open shader source file: " + path.string());
    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    vko::shaderc::SpirvBinary binary(m_compiler, source, shaderKind, path.string(),
                                     "main", options ? *options : m_options);

#if 0
    auto spvFilePath = std::filesystem::path(".") / path.filename().replace_extension(".spv");
    std::ofstream spvFile(spvFilePath, std::ios::binary);
    if(!spvFile.good())
      throw std::runtime_error("Cannot write spv file " + spvFilePath.string());
    spvFile.write(reinterpret_cast<const char*>(binary.span().data()), binary.span().size_bytes());
#endif

    VkShaderModuleCreateInfo shaderModuleCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = binary.span().size_bytes(),
        .pCode    = reinterpret_cast<const uint32_t*>(binary.span().data()),
    };
    return vko::ShaderModule(device, shaderModuleCreateInfo);
  }
  const shaderc::CompileOptions& defaultOptions() const { return m_options; }
  std::vector<std::filesystem::path> m_includeDirs;
  shaderc::CompileOptions            m_options;
  shaderc::Compiler                  m_compiler;
};

// Simple polling-based file change detector. Returns when the file's last write
// time changes.
inline void waitForFileChange(const std::filesystem::path& path)
{
  namespace fs       = std::filesystem;
  auto lastWriteTime = fs::last_write_time(path);
  while(true)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto currentWriteTime = fs::last_write_time(path);
    if(currentWriteTime != lastWriteTime)
      return;
  }
}

// A utility mostly for development. Spins printing error messages and waiting
// for file changes until the shader compiles.
template <vko::device_and_commands DeviceAndCommands>
inline vkobj::ShaderModule reloadUntilCompiling(const DeviceAndCommands& device,
                                                SampleGlslCompiler& glslCompiler,
                                                const std::filesystem::path& path,
                                                shaderc_shader_kind shaderKind,
                                                const shaderc::CompileOptions* options = nullptr)
{
  std::optional<std::filesystem::path> fullPath = glslCompiler.find(path);
  if(!fullPath)
    throw std::runtime_error("file not found: " + path.string());
  for(;;)
  {
    try
    {
      return glslCompiler.compile(device, *fullPath, shaderKind, options);
    }
    catch(const vko::Exception& e)
    {
      std::println("Exception: {}", e.what());
      std::println("Waiting for changes in {}", fullPath->string());
      waitForFileChange(*fullPath);
    }
  }
}

namespace vkobj {

// Simple wrapper around vko::SingleDescriptorSet that can be immediately
// written to at construction time, similar to the old nvvk API.
// For more control, use vko::SingleDescriptorSet, vko::BindingsAndFlags, and
// vko::WriteDescriptorSetBuilder directly.
class SingleDescriptorSet
{
public:
  struct Binding
  {
    uint32_t                                                    index;
    VkDescriptorType                                            descriptorType;
    std::variant<VkDescriptorBufferInfo, VkDescriptorImageInfo> descriptorInfo;
  };

  template <vko::device_and_commands DeviceAndCommands>
  SingleDescriptorSet(const DeviceAndCommands&       device,
                      VkShaderStageFlags             stageFlags,
                      std::initializer_list<Binding> bindingInfos)
  {
    // Build the bindings and flags
    vko::BindingsAndFlags bindings;
    bindings.bindings.reserve(bindingInfos.size());
    bindings.flags.reserve(bindingInfos.size());

    for(const Binding& binding : bindingInfos)
    {
      bindings.bindings.push_back(VkDescriptorSetLayoutBinding{
          .binding            = binding.index,
          .descriptorType     = binding.descriptorType,
          .descriptorCount    = 1,
          .stageFlags         = stageFlags,
          .pImmutableSamplers = nullptr,
      });
      bindings.flags.push_back(0);  // No special flags
    }

    // Create the descriptor set
    m_descriptorSet.emplace(device, bindings, 0,
                            VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

    // Write descriptor data immediately
    vko::WriteDescriptorSetBuilder writes;
    size_t                         bindingIndex = 0;
    for(const Binding& binding : bindingInfos)
    {
      const auto& layoutBinding = bindings.bindings[bindingIndex];

      if(std::holds_alternative<VkDescriptorBufferInfo>(binding.descriptorInfo))
      {
        const auto& info = std::get<VkDescriptorBufferInfo>(binding.descriptorInfo);
        if(binding.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
          writes.push_back<VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER>(m_descriptorSet->set,
                                                              layoutBinding, 0, info);
        else if(binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
          writes.push_back<VK_DESCRIPTOR_TYPE_STORAGE_BUFFER>(m_descriptorSet->set,
                                                              layoutBinding, 0, info);
      }
      else if(std::holds_alternative<VkDescriptorImageInfo>(binding.descriptorInfo))
      {
        const auto& info = std::get<VkDescriptorImageInfo>(binding.descriptorInfo);
        if(binding.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
          writes.push_back<VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER>(
              m_descriptorSet->set, layoutBinding, 0, info);
        else if(binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
          writes.push_back<VK_DESCRIPTOR_TYPE_STORAGE_IMAGE>(m_descriptorSet->set,
                                                             layoutBinding, 0, info);
        else if(binding.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
          writes.push_back<VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE>(m_descriptorSet->set,
                                                             layoutBinding, 0, info);
      }
      bindingIndex++;
    }

    device.vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.writes().size()),
                                  writes.writes().data(), 0, nullptr);
  }

  VkDescriptorSetLayout layout() const { return m_descriptorSet->layout; }
  operator VkDescriptorSet() const { return m_descriptorSet->set; }
  const VkDescriptorSet* ptr() const { return m_descriptorSet->set.ptr(); }

private:
  std::optional<vko::SingleDescriptorSet> m_descriptorSet;
};

// Simpler version of nvvk::PushComputeDispatcher, without integrated binding or
// barrier support. This is quite application-specific and does not support
// sharing the pipeline layout.
template <class PushConstants = void>
struct SimpleComputePipeline
{
  SimpleComputePipeline() = default;

  // Constructor for only push constants
  template <vko::device_and_commands DeviceAndCommands>
  SimpleComputePipeline(const DeviceAndCommands&       device,
                        SampleGlslCompiler&            glslCompiler,
                        const std::filesystem::path&   path,
                        const shaderc::CompileOptions* options = nullptr)
    requires(!std::is_void_v<PushConstants>)
      : pipelineLayout([&]() {
        VkPushConstantRange pushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                              sizeof(PushConstants)};
        VkPipelineLayoutCreateInfo createInfo = {
            .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext          = nullptr,
            .flags          = 0,
            .setLayoutCount = 0u,
            .pSetLayouts    = nullptr,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pushConstantRange,
        };
        return vkobj::PipelineLayout(device, createInfo);
      }())
      , pipeline([&, layout = &pipelineLayout /* WAR C4355 capturing 'this' */]() {
        auto module = reloadUntilCompiling(device, glslCompiler, path,
                                           shaderc_glsl_compute_shader, options);
        VkComputePipelineCreateInfo createInfo = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage =
                VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext               = nullptr,
                    .flags               = 0,
                    .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module              = module,
                    .pName               = "main",
                    .pSpecializationInfo = nullptr,
                },
            .layout             = *layout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex  = 0,
        };
        return vkobj::ComputePipeline(device, createInfo);
      }())
  {
  }

  // Constructor that includes a descriptor set layout, possibly also with push
  // constants
  template <vko::device_and_commands DeviceAndCommands>
  SimpleComputePipeline(const DeviceAndCommands&       device,
                        SampleGlslCompiler&            glslCompiler,
                        const std::filesystem::path&   path,
                        VkDescriptorSetLayout          descriptorsetLayout,
                        const shaderc::CompileOptions* options = nullptr)
      : pipelineLayout([&]() {
        if constexpr(std::is_void_v<PushConstants>)
        {
          VkPipelineLayoutCreateInfo createInfo = {
              .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
              .pNext          = nullptr,
              .flags          = 0,
              .setLayoutCount = descriptorsetLayout ? 1u : 0u,
              .pSetLayouts    = &descriptorsetLayout,
              .pushConstantRangeCount = 0,
              .pPushConstantRanges    = nullptr,
          };
          return vkobj::PipelineLayout(device, createInfo);
        }
        else
        {
          VkPushConstantRange pushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                                sizeof(PushConstants)};
          VkPipelineLayoutCreateInfo createInfo = {
              .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
              .pNext          = nullptr,
              .flags          = 0,
              .setLayoutCount = descriptorsetLayout ? 1u : 0u,
              .pSetLayouts    = &descriptorsetLayout,
              .pushConstantRangeCount = 1,
              .pPushConstantRanges    = &pushConstantRange,
          };
          return vkobj::PipelineLayout(device, createInfo);
        }
      }())
      , pipeline([&, layout = &pipelineLayout /* WAR C4355 capturing 'this' */]() {
        auto module = reloadUntilCompiling(device, glslCompiler, path,
                                           shaderc_glsl_compute_shader, options);
        VkComputePipelineCreateInfo createInfo = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage =
                VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext               = nullptr,
                    .flags               = 0,
                    .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module              = module,
                    .pName               = "main",
                    .pSpecializationInfo = nullptr,
                },
            .layout             = *layout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex  = 0,
        };
        return vkobj::ComputePipeline(device, createInfo);
      }())
  {
  }

  // Convenience overloads for a const ref to compiler options
  template <vko::device_and_commands DeviceAndCommands>
  SimpleComputePipeline(const DeviceAndCommands&       device,
                        SampleGlslCompiler&            glslCompiler,
                        const std::filesystem::path&   path,
                        const shaderc::CompileOptions& options)
      : SimpleComputePipeline(device, glslCompiler, path, &options)
  {
  }
  template <vko::device_and_commands DeviceAndCommands>
  SimpleComputePipeline(const DeviceAndCommands&       device,
                        SampleGlslCompiler&            glslCompiler,
                        const std::filesystem::path&   path,
                        VkDescriptorSetLayout          descriptorsetLayout,
                        const shaderc::CompileOptions& options)
      : SimpleComputePipeline(device, glslCompiler, path, descriptorsetLayout, &options)
  {
  }

  operator VkPipeline() const { return pipeline; }
  vkobj::PipelineLayout  pipelineLayout;
  vkobj::ComputePipeline pipeline;
};

}  // namespace vkobj
