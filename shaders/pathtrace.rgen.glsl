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
#include "nvvkhl/shaders/constants.h"
#include "nvvkhl/shaders/random.h"
#include "pathtrace_device_host.h"
#include "pathtrace_payload.h"

// clang-format off
layout(location = 0) rayPayloadEXT HitPayload payload;
layout(set = DSRt, binding = BRtTlas) uniform accelerationStructureEXT topLevelAS;
layout(set = DSRt, binding = BRtOutBaseColor_Metalness, rgba8) uniform image2D imageBaseColor_Metalness;  // eGBufBaseColor_Metalness
layout(set = DSRt, binding = BRtOutSpecAlbedo, rgba8) uniform image2D imageSpecAlbedo;                    // eGBufSpecAlbedo
layout(set = DSRt, binding = BRtOutSpecHitDist, r16f) uniform image2D imageSpecHitDist;                   // eGBufSpecHitDist
layout(set = DSRt, binding = BRtOutNormalRoughness, rgba16f) uniform image2D imageNormalRoughness;        // eGBufNormalRoughness
layout(set = DSRt, binding = BRtOutMotionVectors, rg16f) uniform image2D imageMotionVectors;              // eGBufMotionVectors
layout(set = DSRt, binding = BRtOutViewZ, r16f) uniform image2D imageViewZ;                               // eGBufViewZ
layout(set = DSRt, binding = BRtOutColor, rgba16f) uniform image2D imageColor;                            // eGBufColor
layout(set = DSRt, binding = BRtFrameInfo) uniform FrameParams_ { FrameParams frameInfo; };
// clang-format on

layout(push_constant) uniform RtxPushConstant_
{
  PathtraceConstant pc;
};

#define DLSS_INF_DISTANCE 65504.0  // FP16 max number

vec2 eyeToImage(vec4 eyeVec)
{
  vec4 csPos = frameInfo.proj * eyeVec;
  vec3 ndc   = csPos.xyz / csPos.w;
  return (ndc.xy * 0.5 + 0.5) * imageSize(imageMotionVectors);
}

vec2 computeMotionVector(vec4 worldSpaceVec)
{
  return eyeToImage(frameInfo.viewLast * worldSpaceVec) - eyeToImage(frameInfo.view * worldSpaceVec);
}

void main()
{
  uint seed = xxhash32(uvec3(gl_LaunchIDEXT.xy, pc.frame));

  bool hitGeometry = false;
  HitPayload firstHit;
  firstHit.albedoMetalness = u8vec4(0.0);
  firstHit.normalRoughness = u8vec4(0.0);
  float specularHitDist = 0.0; //DLSS_INF_DISTANCE;
  u8vec3 specularAlbedo = u8vec3(0.0);

  int32_t sampleCount   = pc.config.sampleCountPixel;
  vec3    sampleAverage = vec3(0);
  for(int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
  {
    const vec2 jitter     = pc.dlssEnabled == 1 ? pc.jitter : vec2(rand(seed), rand(seed));
    const vec2 pixelCoord = vec2(gl_LaunchIDEXT.xy + jitter) / vec2(gl_LaunchSizeEXT.xy);
    const vec3 esTarget   = vec3(frameInfo.projInv * vec4(pixelCoord * 2.0 - 1.0, 0.0, 1.0));
    const vec3 wsPos      = vec3(frameInfo.viewInv * vec4(0.0, 0.0, 0.0, 1.0));
    const vec3 wsDir      = normalize(mat3(frameInfo.viewInv) * esTarget);

    firstHit.direction = wsDir;

    payload = initPayload(wsPos, wsDir, seed);
    do
    {
      const uint rayFlags = 0;        // e.g. gl_RayFlagsCullBackFacingTrianglesEXT;
      traceRayEXT(topLevelAS,         // acceleration structure
                  rayFlags,           // rayFlags
                  0xFF,               // cullMask
                  0,                  // sbtRecordOffset
                  0,                  // sbtRecordStride
                  0,                  // missIndex
                  payload.origin,     // ray origin
                  0.0,                // ray t_min
                  payload.direction,  // ray direction
                  INFINITE,           // ray t_max
                  0                   // payload (location = 0)
      );
      if(payload.depth == 1)
      {
        firstHit = payload;
        hitGeometry = payload.eventType != EVENT_TYPE_MISS;
      }
      else if(payload.depth == 2 && firstHit.eventType == EVENT_TYPE_SPECULAR)
      {
        float rayLength = length(payload.origin - firstHit.origin);
        specularHitDist = length(firstHit.origin - wsPos) + rayLength;
        specularAlbedo = u8vec3(255.0);
        firstHit.origin += wsDir * rayLength;
      }
    } while(pc.config.pathtrace != 0 &&
            payload.depth < pc.config.maxDepth &&
            payload.depth < PATHTRACE_MAX_RGEN_RECURSION_DEPTH &&
            payload.eventType != EVENT_TYPE_MISS &&
            any(greaterThan(payload.transmittance, vec3(0.5 / 255.0))));

    // Avoid fireflies
    payload.radiance = min(payload.radiance, vec3(50.0));

    // Accumulate
    sampleAverage += payload.radiance;

    // Keep sampling...
    seed = payload.seed;
  }
  sampleAverage /= sampleCount;

  //sampleAverage *= 10.0; // hard coded exposure

  vec2 motionVector = computeMotionVector(hitGeometry ? vec4(firstHit.origin, 1.0) : vec4(firstHit.direction, 0.0));
  float viewZ = DLSS_INF_DISTANCE;
  if(hitGeometry)
    viewZ = -(frameInfo.view * vec4(firstHit.origin, 1.0)).z;
  imageStore(imageBaseColor_Metalness, ivec2(gl_LaunchIDEXT.xy), vec4(firstHit.albedoMetalness) / 255.0);
  imageStore(imageMotionVectors, ivec2(gl_LaunchIDEXT.xy), vec4(motionVector, 0.0, 0.0));
  imageStore(imageSpecAlbedo, ivec2(gl_LaunchIDEXT.xy), vec4(specularAlbedo, 255.0) / 255.0);
  imageStore(imageSpecHitDist, ivec2(gl_LaunchIDEXT.xy), vec4(specularHitDist));
  imageStore(imageNormalRoughness, ivec2(gl_LaunchIDEXT.xy),
             vec4((vec3(firstHit.normalRoughness.xyz) / 255.0) * 2.0 - 1.0, firstHit.normalRoughness.w / 255.0));
  imageStore(imageViewZ, ivec2(gl_LaunchIDEXT.xy), vec4(viewZ));
  imageStore(imageColor, ivec2(gl_LaunchIDEXT.xy), vec4(sampleAverage, 1.0F));
}
