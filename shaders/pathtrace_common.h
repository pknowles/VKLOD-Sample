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

#ifndef PATHTRACE_COMMON_H
#define PATHTRACE_COMMON_H

#define UNSHADOWED_FOG 0

float fogTransmittance(vec3 p0, vec3 p1, float t /* redundant distance from p0 to p1*/)
{
  float h_0 = min(p0.y, p1.y) - pc.config.fogHeightOffset;
  float h_1 = max(p0.y, p1.y) - pc.config.fogHeightOffset;
  float a   = 8.0 * pc.config.fogDensity;  // adjusts fog gradient
  float b   = 1.0 * pc.config.fogDensity;

  // https://www.wolframalpha.com/input?i=limit%28product%28e%5E%28-b+t+%28e%5E%28-a+*%28%28k%2Fn%29+h_1%2B%281-k%2Fn%29+h_0%29%29%2Fn%29%29%2Ck%2C1%2Cn%29%2Cn%2Cinf%29
  // TODO: degenerate division fix with 'max()' results in no fog at equal heights
  return min(1.0, exp((b * (exp(-a * h_1) - exp(-a * h_0)) * t) / max(1e-6, a * (h_1 - h_0))));
}

// Return true if there is no occluder, meaning that the light is visible from P toward L
bool shadowRay(vec3 P, vec3 L, float maxDist)
{
  const uint rayFlags     = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
  HitPayload savedPayload = payload;
  payload.depth           = 0;
  payload.eventType       = EVENT_TYPE_SHADOW;
  traceRayEXT(topLevelAS, rayFlags, 0xFF, 0, 0, 0, P, 0.0, L, maxDist, 0);
  bool visible = (payload.depth != 0);
  payload      = savedPayload;
  return visible;
}

#endif  // PATHTRACE_COMMON_H