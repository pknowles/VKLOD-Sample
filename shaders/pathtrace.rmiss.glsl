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

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable

#include "dh_bindings.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/random.h"
#include "pathtrace_device_host.h"
#include "pathtrace_payload.h"

layout(set = 0, binding = BRtTlas ) uniform accelerationStructureEXT topLevelAS;
layout(location = 0) rayPayloadInEXT HitPayload payload;
layout(set = 0, binding = BRtSkyParam) uniform SkyInfo_
{
  SimpleSkyParameters skyInfo;
};
layout(push_constant) uniform RtxPushConstant_ { PathtraceConstant pc; };

#include "pathtrace_common.h"

void main()
{
  payload.depth += 1;

  // special case for shadowRay() rays
  if(payload.eventType == EVENT_TYPE_SHADOW)
    return;

  payload.eventType = EVENT_TYPE_MISS;

  SimpleSkyParameters p = skyInfo;
  if(payload.depth > 1)
    p.lightRadiance = vec3(0.0);  // Sunlight is added in the closest hit shader
  vec3 sky_color = evalSimpleSky(p, gl_WorldRayDirectionEXT);
  //sky_color = vec3(0.5); // white furnace test

#if UNSHADOWED_FOG
  payload.radiance += payload.transmittance * sky_color;
#else
  // brute force single-scattering fog approximation
  const float maxDistance      = 0.5 / max(0.1, pc.config.fogDensity);
  const int   maxShadowSamples = 3;
  const float ranges[3]        = {0.05, 0.2 - 0.05, 0.7999};
  float       randOffset       = rand(payload.seed);
  vec3        lastPos          = gl_WorldRayOriginEXT;
  float       lastDist         = 0.0;
  for(int i = 0; i < maxShadowSamples; ++i)
  {
    float d = lastDist + ranges[i] * maxDistance * randOffset;
    vec3  p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * d;
    float t = fogTransmittance(lastPos, p, d - lastDist);
    if(shadowRay(p, normalize(skyInfo.directionToLight), 100000.0))
      payload.radiance += payload.transmittance * (1.0 - t) * skyInfo.horizonColor;
    payload.transmittance *= t;
    lastPos  = p;
    lastDist = d;
  }
  payload.radiance += payload.transmittance * sky_color;

#if 0
  payload.albedoMetalness = u8vec4(sky_color * 255.0, 0.0);
  payload.normalRoughness = u8vec4((-gl_WorldRayDirectionEXT * 0.5 + 0.5) * 255.0, 0.0);
#endif
#endif
}
