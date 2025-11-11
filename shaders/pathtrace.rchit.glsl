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
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_ray_tracing_position_fetch : require
#extension GL_EXT_spirv_intrinsics : require

spirv_decorate(extensions = ["SPV_NV_cluster_acceleration_structure"], capabilities = [5437], 11, 5436) in int gl_ClusterIDNV_;

#include "dh_bindings.h"
#include "nvvkhl/shaders/bsdf_functions.h"
#include "nvvkhl/shaders/dh_scn_desc.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/random.h"
#include "pathtrace_device_host.h"
#include "pathtrace_payload.h"
#include "shaders_scene.h"
#include "traverse_device_host.h"

layout(set = 0, binding = BRtTextures) uniform sampler2D[] texturesMap;

// Barycentric coordinates of hit location relative to triangle vertices
hitAttributeEXT vec2 hitBaryCoord;

// must be after dh_scn_desc.h (missing include)
#include "nvvkhl/shaders/pbr_mat_eval.h"

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;
layout(set = 0, binding = BRtTlas ) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = BRtFrameInfo) uniform FrameParams_ { FrameParams frameInfo; };
layout(set = 0, binding = BRtSkyParam) uniform SkyInfo_ { SimpleSkyParameters skyInfo; };
layout(push_constant) uniform RtxPushConstant_ { PathtraceConstant pc; };
// clang-format on

#include "pathtrace_common.h"

const float pi = 3.14159265f;

// Version for refraction
float fresnelSchlickApprox(vec3 incident, vec3 normal, float ior)
{
  float r0 = (ior - 1.0) / (ior + 1.0);
  r0 *= r0;
  float cosX = -dot(normal, incident);
  if(ior > 1.0)
  {
    float sinT2 = ior * ior * (1.0 - cosX * cosX);
    // Total internal reflection
    if(sinT2 > 1.0)
      return 1.0;
    cosX = sqrt(1.0 - sinT2);
  }
  float x   = 1.0 - cosX;
  float ret = r0 + (1.0 - r0) * x * x * x * x * x;
  return ret;
}

float schlickMaskingTerm(float NdotL, float NdotV, float roughness)
{
  // Karis notes they use alpha / 2 (or roughness^2 / 2)
  float k = roughness * roughness / 2;

  // Compute G(v) and G(l).  These equations directly from Schlick 1994
  //     (Though note, Schlick's notation is cryptic and confusing.)
  float g_v = NdotV / (NdotV * (1 - k) + k);
  float g_l = NdotL / (NdotL * (1 - k) + k);
  return g_v * g_l;
}

float distributionGGX(float NdotH, float alphaRoughness)
{
  float alphaSqr = max(alphaRoughness * alphaRoughness, 1e-07);

  float NdotHSqr = NdotH * NdotH;
  float denom    = NdotHSqr * (alphaSqr - 1.0) + 1.0;

  return alphaSqr / (pi * denom * denom);
}

float schlickFresnelMetalness(float metalness, float LdotH)
{
  float f90 = 1.0;
  float f0Dielectric = 0.04;  // Standard dielectric F0
  float f0Metallic = 0.5;     // Approximate average for metals

  // Two different hard coded f0 values for metallic and non-metallic materials.
  // Ideally metalness would blend the result of two BRDFs. See:
  // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
  return mix(schlickFresnel(f0Dielectric, f90, LdotH), schlickFresnel(f0Metallic, f90, LdotH), metalness);
}

vec3 ggxDirect(vec3 N, vec3 V, vec3 L, float roughness, float metalness, vec3 lightIndensity, vec3 albedo)
{
  // Compute half vectors and additional dot products for GGX
  vec3  H     = normalize(V + L);
  float NdotH = max(0.001, dot(N, H));
  float LdotH = max(0.001, dot(L, H));
  float NdotV = max(0.001, dot(N, V));
  float NdotL = max(0.001, dot(N, L));

  // Evaluate terms for our GGX BRDF model
  float D = distributionGGX(NdotH, roughness);
  float G = schlickMaskingTerm(NdotL, NdotV, roughness);
  float F = schlickFresnelMetalness(metalness, LdotH);

  vec3 specularAlbedo = mix(vec3(1.0), albedo, metalness);

  // Evaluate the Cook-Torrance Microfacet BRDF model
  //     Cancel NdotL here to avoid catastrophic numerical precision issues.
  float ggxTerm = D * G * F / (4 * NdotV /* * NdotL */);

  return lightIndensity * (/* NdotL * */ ggxTerm * pi * specularAlbedo + NdotL * albedo * (1.0 - metalness));
}

// Utility function to get a vector perpendicular to an input vector
//    (from "Efficient Construction of Perpendicular Vectors Without Branching")
vec3 getPerpendicularVector(vec3 u)
{
  vec3 a  = abs(u);
  uint xm = ((a.x - a.y) < 0 && (a.x - a.z) < 0) ? 1 : 0;
  uint ym = (a.y - a.z) < 0 ? (1 ^ xm) : 0;
  uint zm = 1 ^ (xm | ym);
  return cross(u, vec3(xm, ym, zm));
}

vec3 getCosHemisphereSample(inout uint seed, vec3 hitNorm)
{
  // Get 2 random numbers to select our sample with
  vec2 randVal = vec2(rand(seed), rand(seed));

  // Cosine weighted hemisphere sample from RNG
  vec3  bitangent = getPerpendicularVector(hitNorm);
  vec3  tangent   = cross(bitangent, hitNorm);
  float r         = sqrt(randVal.x);
  float phi       = 2.0f * pi * randVal.y;

  // Get our cosine-weighted hemisphere lobe sample direction
  return tangent * (r * cos(phi).x) + bitangent * (r * sin(phi)) + hitNorm.xyz * sqrt(1 - randVal.x);
}

// When using this function to sample, the probability density is:
//      pdf = D * NdotH / (4 * HdotV)
vec3 getGGXMicrofacetDirection(inout uint seed, float roughness, vec3 hitNorm)
{
  // Get our uniform random numbers
  vec2 randVal = vec2(rand(seed), rand(seed));

  // Get an orthonormal basis from the normal
  vec3 B = getPerpendicularVector(hitNorm);
  vec3 T = cross(B, hitNorm);

  // GGX NDF sampling
  float a2        = roughness * roughness;
  float cosThetaH = sqrt(max(0.0f, (1.0 - randVal.x) / ((a2 - 1.0) * randVal.x + 1)));
  float sinThetaH = sqrt(max(0.0f, 1.0f - cosThetaH * cosThetaH));
  float phiH      = randVal.y * pi * 2.0f;

  // Get our GGX NDF sample (i.e., the half vector)
  return T * (sinThetaH * cos(phiH)) + B * (sinThetaH * sin(phiH)) + hitNorm * cosThetaH;
}

// utility for temperature
float fade(float low, float high, float value)
{
  float mid   = (low + high) * 0.5;
  float range = (high - low) * 0.5;
  float x     = 1.0 - clamp(abs(mid - value) / range, 0.0, 1.0);
  return smoothstep(0.0, 1.0, x);
}

vec3 hsv2rgb(vec3 c)
{
  vec4 K = {1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0};
  vec3 p = abs(fract(vec3(c.x) + vec3(K)) * 6.0f - vec3(K.w));
  return c.z * mix(vec3(K.x), clamp(p - vec3(K.x), vec3(0.0), vec3(1.0)), c.y);
};

vec3 heatmapsv(float f, float saturation, float value)
{
  return hsv2rgb(vec3((2.0 - f * 2.0) / 3.0, saturation, value));
}

// 0 to 1 returns blue to red
vec3 heatmap(float f)
{
  return heatmapsv(f, 1.0, 1.0);
}

// random hsv([0,1], [.5,1], [.25,1])
vec3 colorHash(uint h)
{
  h ^= h >> 13;
  h *= 0x5bd1e995;
  h ^= h >> 15;
  return hsv2rgb(vec3(float(h & 0xffff) / 65535.f, float((h >> 16) & 0xff) / 340.f + 0.25, float((h >> 24) & 0xff) / 340.f + 0.25));
};

void wireframe(inout float wire, float width, vec3 bary)
{
  float minBary = min(bary.x, min(bary.y, bary.z));
  wire          = min(wire, smoothstep(width, width + 0.002F, minBary));
}

vec2 worldToImage(vec3 wsPos)
{
  vec4 csPos = frameInfo.proj * frameInfo.view * vec4(wsPos, 1.0);
  vec3 ndc   = csPos.xyz / csPos.w;
  return (ndc.xy * 0.5 + 0.5) * gl_LaunchSizeEXT.xy;
}

vec2 mixBary(vec2 a, vec2 b, vec2 c, vec2 bary)
{
  return a * (1.0 - bary.x - bary.y) + b * bary.x + c * bary.y;
}

vec3 mixBary(vec3 a, vec3 b, vec3 c, vec2 bary)
{
  return a * (1.0 - bary.x - bary.y) + b * bary.x + c * bary.y;
}

void main()
{
  payload.depth += 1;

#if 0
  payload.radiance = vec3(1, 0, 0);
  return;
#endif

#if 0
  payload.radiance = colorHash(gl_ClusterIDNV_);
  //payload.radiance = colorHash(gl_PrimitiveID);
  return;
#endif

#if 0
  InstanceArray     instances = InstanceArray(pc.instancesAddress);
  Instance          instance  = instances.array[gl_InstanceID];
  uint32_t          meshIndex = instance.meshIndex;
#else
  uint32_t meshIndex = uint32_t(gl_GeometryIndexEXT);
#endif
  MeshArray         meshes = MeshArray(pc.meshesAddress);
  Mesh              mesh   = meshes.array[meshIndex];
  ClusterGroupArray groups = ClusterGroupArray(mesh.groupsAddress);

  // Group and cluster-within-group indices are encoded in the cluster ID
  uint32_t clusterIndex = uint32_t(gl_ClusterIDNV_) & CLUSTER_ID_CLUSTER_MASK;
  uint32_t groupIndex   = uint32_t(gl_ClusterIDNV_) >> CLUSTER_ID_GROUP_SHIFT;

  ClusterGroup         group                   = groups.array[groupIndex];
  ClusterGeometryArray clusterGeometries       = ClusterGeometryArray(group.clusterGeometryAddressesAddress);
  ClusterGeometry      clusterGeometry         = clusterGeometries.array[clusterIndex];
  U8Vec3Array          clusterTriangleVertices = U8Vec3Array(clusterGeometry.triangleVerticesAddress);
  Vec3Array            clusterVertexPositions  = Vec3Array(clusterGeometry.vertexPositionsAddress);
  Vec3Array            clusterVertexNormals    = Vec3Array(clusterGeometry.vertexNormalsAddress);
  Vec2Array            clusterVertexTexcoords  = Vec2Array(clusterGeometry.vertexTexcoordsAddress);

  u8vec3 tri            = clusterTriangleVertices.array[gl_PrimitiveID];
  vec3   v0             = clusterVertexPositions.array[tri.x];
  vec3   v1             = clusterVertexPositions.array[tri.y];
  vec3   v2             = clusterVertexPositions.array[tri.z];
  vec3   interpPosition = mixBary(clusterVertexPositions.array[tri.x], clusterVertexPositions.array[tri.y],
                                  clusterVertexPositions.array[tri.z], hitBaryCoord);
  vec3   interpNormal   = mixBary(clusterVertexNormals.array[tri.x], clusterVertexNormals.array[tri.y],
                                  clusterVertexNormals.array[tri.z], hitBaryCoord);
  vec2   interpTexCoord = clusterGeometry.vertexTexcoordsAddress == 0 ?
                              vec2(0.0) :
                              mixBary(clusterVertexTexcoords.array[tri.x], clusterVertexTexcoords.array[tri.y],
                                      clusterVertexTexcoords.array[tri.z], hitBaryCoord);

  mat3 normalMatrix = transpose(mat3(gl_WorldToObjectEXT));  // WorldToObject = inverse(ObjectToWorld)
  vec3 wPos         = vec3(gl_ObjectToWorldEXT * vec4(interpPosition, 1.0));
  vec3 wGeomNorm    = normalize(normalMatrix * cross(v1 - v0, v2 - v0));
  vec3 wNorm        = normalize(normalMatrix * interpNormal);

  vec4  albedo    = mesh.material.albedo;
  float metalness = mesh.material.metallic;
  float roughness = mesh.material.roughness;
  if(mesh.material.albedoTexture != -1)
    albedo *= texture(texturesMap[mesh.material.albedoTexture], interpTexCoord);
  if(mesh.material.metallicRoughnessTexture != -1)
  {
    vec4 metallicRoughness = texture(texturesMap[mesh.material.metallicRoughnessTexture], interpTexCoord);
    roughness = metallicRoughness.g;
    metalness = metallicRoughness.b;
  }

  //wNorm = wGeomNorm;
  //albedo = vec4(1.0);
  //albedo.xyz  = wNorm*0.5+0.5;
  //roughness = 0.0;
  //metalness = 1.0;
  //albedo = vec4(0.0, roughness, metalness, 1.0);

  vec3 wDir   = normalize(gl_WorldRayDirectionEXT);
  vec3 wEye   = -wDir;
  vec3 wLight = normalize(skyInfo.directionToLight);

  bool isFront = gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT;
  if(!isFront)
  {
    wNorm      = -wNorm;
    wGeomNorm  = -wGeomNorm;
  }

  // Clamp the interpolated normal so it is not facing away from the viewer
  if(dot(wNorm, wEye) < 1e-6)
    wNorm = normalize(cross(cross(wEye, wNorm), wEye) + wEye * 1e-6);

  SimpleSkyParameters p = skyInfo;
  p.horizonColor *= 0.0;  // atmospheric light is added in the miss shader
  p.groundColor *= 0.0;
  p.skyColor *= 0.0;
  vec3 lightIntensity = evalSimpleSky(p, wLight);

  // Debug visualization
#if 0
  payload.depth = 999;
  payload.radiance = wNorm * 0.5 + 0.5;
  return;
#endif

  NodeArray  nodes                  = NodeArray(mesh.nodesAddress);
  UintArray  clusterGenratingGroups = UintArray(group.clusterGeneratingGroupsAddress);
  FloatArray groupQuadricErrors     = FloatArray(mesh.groupQuadricErrorsAddress);
  Vec4Array  groupBoundingSpheres   = Vec4Array(mesh.groupBoundingSpheresAddress);
  Uint8Array groupLodLevels         = Uint8Array(mesh.groupLodLevelsAddress);
  uint8_t    groupLodLevel          = groupLodLevels.array[groupIndex];
  uint32_t   generatingGroupIndex   = clusterGenratingGroups.array[clusterIndex];
  float      groupQuadricError      = groupQuadricErrors.array[groupIndex];
  vec4       groupBoundingSphere    = groupBoundingSpheres.array[groupIndex];
  float generatingGroupQuadricError = generatingGroupIndex == 0xffffffff ? 0.0f : groupQuadricErrors.array[generatingGroupIndex];
  vec4 generatingGroupBoundingSphere =
      generatingGroupIndex == 0xffffffff ? vec4(0.0f) : groupBoundingSpheres.array[generatingGroupIndex];

  // Use flat shading for visualizations
  if(pc.config.lodVisualization != VISUALIZE_NONE)
    wNorm = wGeomNorm;

  if(pc.config.lodVisualization == VISUALIZE_TRIANGLE_COLORS)
    albedo = mix(albedo, vec4(colorHash(gl_PrimitiveID), 1.0), 0.6);
  if(pc.config.lodVisualization == VISUALIZE_CLUSTER_COLORS)
    albedo = mix(albedo, vec4(colorHash(gl_ClusterIDNV_), 1.0), 0.6);
  if(pc.config.lodVisualization == VISUALIZE_GENERATING_GROUP_COLORS)
    albedo = vec4(colorHash(generatingGroupIndex), 1.0);
  if(pc.config.lodVisualization == VISUALIZE_MESH_COLORS)
    albedo = vec4(colorHash(meshIndex), 1.0);
  if(pc.config.lodVisualization == VISUALIZE_INSTANCE_COLORS)
    albedo = vec4(colorHash(gl_InstanceID), 1.0);
  if(pc.config.lodVisualization == VISUALIZE_CLUSTER_LOD)
  {
    uint tmp         = gl_ClusterIDNV_;
    uint clusterHash = pcg(tmp);  // modifies input
    albedo           = mix(albedo,
                           vec4(heatmapsv(1.0 - groupLodLevel * 0.125, groupLodLevel <= 6 ? 1.0 : 0.0, (clusterHash & 0x7f) / 190.0 + 0.2), 1.0),
                           0.6);
  }
  if(pc.config.lodVisualization == VISUALIZE_TRIANGLE_AREA)
  {
    vec2  iv0            = worldToImage(gl_ObjectToWorldEXT * vec4(v0, 1.0));
    vec2  iv1            = worldToImage(gl_ObjectToWorldEXT * vec4(v1, 1.0));
    vec2  iv2            = worldToImage(gl_ObjectToWorldEXT * vec4(v2, 1.0));
    vec2  e0             = iv1 - iv0;
    vec2  e1             = iv2 - iv0;
    float triangleAreaPx = abs(cross(vec3(e0, 0.0), vec3(e1, 0.0)).z * 0.5);
    float blueAreaPx     = 2.0 * 2.0;
    albedo               = vec4(heatmap(1.0 - min(1.0, triangleAreaPx / blueAreaPx)), 1.0);
  }
  if(pc.config.lodVisualization == VISUALIZE_TARGET_PIXEL_ERROR)
  {
    InstanceArray instances = InstanceArray(pc.instancesAddress);
    Instance      instance  = instances.array[gl_InstanceID];
    const vec3    wCamera   = vec3(frameInfo.viewInv * vec4(0.0, 0.0, 0.0, 1.0));
    float         distance  = length(wCamera - wPos);
    float decimatedMinDist  = max(0.01, length(wCamera - gl_ObjectToWorldEXT * vec4(groupBoundingSphere.xyz, 1.0))
                                            - groupBoundingSphere.w * instance.uniformScale);
    float clusterMaxEOD     = instance.uniformScale * generatingGroupQuadricError / distance;
    float decimatedMaxEOD   = instance.uniformScale * groupQuadricError / decimatedMinDist;

    albedo = vec4(heatmap(0.1 + 0.8 * smoothstep(clusterMaxEOD, decimatedMaxEOD, pc.errorOverDistanceThreshold)), 1.0);

    // Add triangle colors as density is important to see too
    albedo.rgb = mix(albedo.rgb, colorHash(gl_PrimitiveID), 0.6);

    // Error if over-detailed
    if(decimatedMaxEOD < pc.errorOverDistanceThreshold)
      albedo = mix(albedo, vec4(1, 0, 0, 1), 0.8);

    // Final sanity check: the rendered cluster's quadric error over distance
    // must be below the threshold. If it's not, draw pink. This may happen
    // before streaming fetches the appropriately detailed geometry.
    if(!(clusterMaxEOD < pc.errorOverDistanceThreshold))
      albedo = mix(albedo, vec4(1, 0, 1, 1), 0.8);
  }

  vec3 newOrigin = wPos + wGeomNorm * 1e-4 + wEye * 1e-4;

  bool directLightVisible = shadowRay(newOrigin, wLight, 100000.0);

  // Very approximate atmospheric fog, just for primary rays
  // TODO: handle shadowing in the miss shader too!
  if(pc.config.fogDensity != 0.0 && payload.depth <= 1)
  {
    vec3 fogAmbient = skyInfo.horizonColor;  // artificially blend fog with the miss shader radiance
#if UNSHADOWED_FOG
    float t                          = fogTransmittance(gl_WorldRayOriginEXT, wPos, gl_HitTEXT);
    payload.radiance += payload.transmittance * (1.0 - t) * fogAmbient;  // no direct in-scattering or shadows :(
    payload.transmittance *= t;
#else
    // brute force single-scattering fog approximation
    #if 1
    const int   maxShadowSamples = 3;
    const float ranges[maxShadowSamples] = {0.05, 0.2 - 0.05, 0.7999};
    #else
    const int   maxShadowSamples = 8;
    const float distances[maxShadowSamples] = {0.002, 0.0078, 0.0234, 0.0543, 0.1086, 0.2063, 0.3781, 0.9999};
    const float ranges[maxShadowSamples]    = {distances[0],
                                               distances[1] - distances[0],
                                               distances[2] - distances[1],
                                               distances[3] - distances[2],
                                               distances[4] - distances[3],
                                               distances[5] - distances[4],
                                               distances[6] - distances[5],
                                               distances[7] - distances[6]};
    #endif

    float randOffset = rand(payload.seed);
    vec3  lastPos          = gl_WorldRayOriginEXT;
    float fogStartHeight = pc.config.fogHeightOffset + pc.config.fogDensity * 0.3;
    float jumpTo = max(0.0, (fogStartHeight - gl_WorldRayOriginEXT.y) / min(-0.01, wDir.y)) / gl_HitTEXT;
    float lastDist         = jumpTo * gl_HitTEXT;
    float fogTraceDist = (1.0 - jumpTo) * gl_HitTEXT;
    for(int i = 0; i < maxShadowSamples; ++i)
    {
      float d = lastDist + ranges[i] * fogTraceDist * randOffset;
      vec3  p = gl_WorldRayOriginEXT + wDir * d;
      float t = fogTransmittance(lastPos, p, d - lastDist);
      if(shadowRay(p, wLight, 100000.0))
        payload.radiance += payload.transmittance * (1.0 - t) * fogAmbient;
      payload.transmittance *= t;
      lastPos = p;
      lastDist = d;
    }
    float t = fogTransmittance(lastPos, wPos, gl_HitTEXT - lastDist);
    if(directLightVisible)
      payload.radiance += payload.transmittance * (1.0 - t) * fogAmbient * 0.5;  // half the last in-scattering to compensate for sampling bias towards directLightVisible
    payload.transmittance *= t;
#endif
  }

  // Artificially emissive for visualizations
  if(pc.config.lodVisualization != VISUALIZE_NONE)
    payload.radiance += payload.transmittance * albedo.xyz * 0.3;

  payload.albedoMetalness = u8vec4(albedo.xyz * 255.0, metalness * 255.0);
  payload.normalRoughness = u8vec4((wNorm * 0.5 + 0.5) * 255.0, roughness * 255.0);

  if(directLightVisible)
    payload.radiance += payload.transmittance * albedo.w * ggxDirect(wNorm, wEye, wLight, roughness, metalness, lightIntensity, albedo.xyz);

  if(albedo.w < 0.3)  // rand(payload.seed)
  {
    payload.origin = wPos + wDir * 1e-4;
    // payload.direction unchanged
    payload.transmittance *= 1.0 - albedo.w;
    payload.eventType = EVENT_TYPE_TRANSPARENCY; // stop tracing
    return;
  }

  payload.eventType = EVENT_TYPE_DIFFUSE;
  if(pc.config.pathtrace != 0)
  {
    // Largely based off https://cwyman.org/code/dxrTutors/tutors/Tutor14/tutorial14.md.html
    float diffuseProbability = roughness * roughness;  // should be based on specular albedo
    bool  chooseDiffuse      = (rand(payload.seed) < diffuseProbability);
    if(chooseDiffuse)
    {
      vec3  N     = wNorm;
      vec3  L     = getCosHemisphereSample(payload.seed, N);
      float NdotL = max(0, dot(N, L));

      // Shoot a randomly selected cosine-sampled diffuse ray.
      // Accumulate the color: (NdotL * incomingLight * dif / pi)
      // Probability of sampling this ray:  (NdotL / pi) * probDiffuse
      payload.origin    = newOrigin;
      payload.direction = L;
      payload.transmittance *= albedo.xyz * (1.0 - metalness) / diffuseProbability;
    }
    else
    {
      payload.eventType = EVENT_TYPE_SPECULAR;

      // Randomly sample the NDF to get a microfacet in our BRDF
      vec3 H = normalize(getGGXMicrofacetDirection(payload.seed, roughness, wNorm));

      // Compute outgoing direction based on this (perfectly reflective) facet
      vec3 V = wEye;
      vec3 L = normalize(reflect(-V, H));

      // Compute some dot products needed for shading
      vec3 N = wNorm;
      float NdotL = max(0.001, dot(N, L));
      float NdotH = max(0.001, dot(N, H));
      float LdotH = max(0.001, dot(L, H));
      float NdotV = max(0.001, dot(N, V));

      // Evaluate our BRDF using a microfacet BRDF model
      float D       = distributionGGX(NdotH, roughness);
      float G       = schlickMaskingTerm(NdotL, NdotV, roughness);
      float F       = schlickFresnelMetalness(metalness, LdotH);
      float ggxTerm = D * G * F / (4 * NdotL * NdotV);

      vec3 specularAlbedo = mix(vec3(1.0), albedo.xyz, metalness);

      // What's the probability of sampling vector H from getGGXMicrofacet()?
      float ggxProb = D * NdotH / (4 * LdotH);  // LdotH == HdotV

      // Compute our color by tracing a ray in this direction
      // Accumulate color:  ggx-BRDF * lightIn * NdotL / probability-of-sampling
      //    -> Note: Should really cancel and simplify the math above
      payload.origin    = newOrigin;
      payload.direction = L;
      float ggx = clamp(ggxTerm / ggxProb, 0.0, 1.0);
      payload.transmittance *= NdotL * specularAlbedo * ggx / (1.0f - diffuseProbability);
    }

    if(dot(wGeomNorm, payload.direction) < 0.0)
    {
      payload.direction = -reflect(payload.direction, wGeomNorm);
    }
  }
  else  // pc.config.pathtrace
  {
    // Ambient occlusion only
    SimpleSkyParameters p = skyInfo;
    p.lightRadiance       = vec3(0.0);  // Direct sun has already been added
    float sampleWeight    = 1.0 / pc.config.sampleCountAO;
    vec3  ambientRadiance = vec3(0.0);
    for(int i = 0; i < pc.config.sampleCountAO; ++i)
    {
      vec3 N = wNorm;  //wGeomNorm;
      vec3 L = getCosHemisphereSample(payload.seed, N);
      if(shadowRay(newOrigin, L, pc.config.aoRadius))
      {
        vec3 ambientIntensity = evalSimpleSky(p, L);
        // Sample with transmittance NdotL cancles with probability NdotL.
        float NdotL = max(0, dot(wNorm, L));
#if 1
        ambientRadiance +=
            sampleWeight * ggxDirect(wNorm, wEye, L, mesh.material.roughness, metalness, ambientIntensity, albedo.xyz);
#else
        ambientRadiance += sampleWeight * albedo.xyz * ambientIntensity * NdotL;
#endif
      }
    }
    payload.radiance += payload.transmittance * ambientRadiance;
    payload.eventType = EVENT_TYPE_MISS; // stop tracing
    payload.origin = wPos; // for motion vectors
  }
}
