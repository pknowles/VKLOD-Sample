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

#ifndef SHADERS_MESH_GEN_H
#define SHADERS_MESH_GEN_H

#include "shaders_buffer_ref.h"

#define MESH_GEN_WORKGROUP_SIZE 8

#ifdef __cplusplus
namespace shaders {
#endif  // __cplusplus

struct MeshGenConstants
{
  DEVICE_ADDRESS(u8vec2) edgeVertexIndices;
  DEVICE_ADDRESS(uint16_t) edgeMasks;
  DEVICE_ADDRESS(uint16_t) triangleOffsets;
  DEVICE_ADDRESS(uint8_t) triangleTable;
  DEVICE_ADDRESS(uvec3) outTriangles;
  DEVICE_ADDRESS(vec3) outPositions;
  DEVICE_ADDRESS(vec3) outNormals;
  DEVICE_ADDRESS(uint32_t) outTriangleCount;
  DEVICE_ADDRESS(uint32_t) outVertexCount;
  uint32_t maxTriangles;
  uint32_t maxVertices;
  uint32_t seed;
};

#ifdef __cplusplus
}  // namespace shaders
#endif  // __cplusplus

#endif  // SHADERS_MESH_GEN_H