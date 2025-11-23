/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <mesh_util.hpp>
#include <numeric>
#include <sample_glsl_compiler.hpp>
#include <sample_progress.hpp>
#include <sample_vulkan_objects.hpp>
#include <shaders_mesh_gen.h>
#include <vector>

struct SimpleMesh
{
  std::vector<glm::uvec3> triangles;
  std::vector<glm::vec3>  positions;
  std::vector<glm::vec3>  normals;
};

SimpleMesh generateMesh(const vko::Device&   device,
                        vko::vma::Allocator& allocator,
                        vkobj::Staging&      staging,
                        vkobj::SimpleComputePipeline<shaders::MeshGenConstants>& pipeline,
                        uint32_t seed,
                        uint32_t maxTriangles);

inline shaderc::CompileOptions withDefines(shaderc::CompileOptions options,
                                           std::initializer_list<std::pair<std::string, std::string>> macros)
{
  for(const auto& [name, value] : macros)
    options.AddMacroDefinition(name, value);
  return options;
}

struct GeneratorPipelines
{
  GeneratorPipelines(const vko::Device& device, SampleGlslCompiler& glslCompiler)
      : terrain(device,
                glslCompiler,
                "generate_mesh.comp.glsl",
                withDefines(glslCompiler.defaultOptions(), {{"TERRAIN", "1"}}))
      , mountain(device,
                 glslCompiler,
                 "generate_mesh.comp.glsl",
                 withDefines(glslCompiler.defaultOptions(), {{"MOUNTAIN", "1"}}))
      , rock1(device,
              glslCompiler,
              "generate_mesh.comp.glsl",
              withDefines(glslCompiler.defaultOptions(), {{"ROCK1", "1"}}))
      , rock2(device,
              glslCompiler,
              "generate_mesh.comp.glsl",
              withDefines(glslCompiler.defaultOptions(), {{"ROCK2", "1"}}))
      , rock3(device,
              glslCompiler,
              "generate_mesh.comp.glsl",
              withDefines(glslCompiler.defaultOptions(), {{"ROCK3", "1"}}))
      , rock4(device,
              glslCompiler,
              "generate_mesh.comp.glsl",
              withDefines(glslCompiler.defaultOptions(), {{"ROCK4", "1"}}))
  {
  }
  vkobj::SimpleComputePipeline<shaders::MeshGenConstants> terrain;
  vkobj::SimpleComputePipeline<shaders::MeshGenConstants> mountain;
  vkobj::SimpleComputePipeline<shaders::MeshGenConstants> rock1;
  vkobj::SimpleComputePipeline<shaders::MeshGenConstants> rock2;
  vkobj::SimpleComputePipeline<shaders::MeshGenConstants> rock3;
  vkobj::SimpleComputePipeline<shaders::MeshGenConstants> rock4;
};

struct GeneratedScene
{
  std::vector<SimpleMesh>                     meshes;
  std::vector<std::pair<uint32_t, glm::mat4>> instances;
};

GeneratedScene makeTerrainAndRocksScene(const vko::Device&   device,
                                        SampleGlslCompiler&  glslCompiler,
                                        vko::vma::Allocator& allocator,
                                        vkobj::Staging&      staging,
                                        float                detailScale,
                                        TaskProgress&        progress);