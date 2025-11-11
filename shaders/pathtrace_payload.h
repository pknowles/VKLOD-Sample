/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PAYLOAD_H
#define PAYLOAD_H

#ifdef __cplusplus
namespace shaders {
using vec3 = nvmath::vec3f;
#endif  // __cplusplus

#define EVENT_TYPE_GEN 0
#define EVENT_TYPE_MISS 1
#define EVENT_TYPE_DIFFUSE 2
#define EVENT_TYPE_SPECULAR 3
#define EVENT_TYPE_TRANSPARENCY 4
#define EVENT_TYPE_SHADOW 5

struct HitPayload
{
  vec3 radiance;
  vec3 transmittance;
  vec3 origin;
  vec3 direction;
  uint seed;
  int  depth;
  int  eventType;
  u8vec4 albedoMetalness;
  u8vec4 normalRoughness;
};

HitPayload initPayload(vec3 pos, vec3 dir, uint seed)
{
  HitPayload p;
  p.radiance      = vec3(0);
  p.transmittance = vec3(1);
  p.origin        = pos;
  p.direction     = dir;
  p.depth         = 0;
  p.eventType     = EVENT_TYPE_GEN;
  p.seed          = seed;
  return p;
}

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif  // PAYLOAD_H
