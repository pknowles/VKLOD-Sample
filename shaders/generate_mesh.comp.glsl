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

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable

#include "shaders_mesh_gen.h"

layout(local_size_x = MESH_GEN_WORKGROUP_SIZE, local_size_y = MESH_GEN_WORKGROUP_SIZE, local_size_z = MESH_GEN_WORKGROUP_SIZE) in;

layout(push_constant, scalar) uniform MeshGenConstants_
{
  MeshGenConstants pc;
};

uint pcg_hash(uint v)
{
  v += pc.seed;  // artificially introduce a seed. probably bad
  uint state = v * 747796405u + 2891336453u;
  uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

float pcg1df(uint x)
{
  return float(pcg_hash(x)) / 4294967296.0;  // Divide by 2^32
}

float pcg_noise3D(uint x, uint y, uint z)
{
  uint n = (x << 20) + (y << 10) + z;
  n += pc.seed;  // artificially introduce a seed. probably bad
  n = n * 747796405u + 1u;
  n = ((n >> ((n >> 28) + 4u)) ^ n) * 277803737u;
  n = (n >> 22) ^ n;
  return float(n) / 4294967295.0;
}

vec3 pcg3dfs(uint i)
{
  return vec3(pcg1df(i * 3u + 0u), pcg1df(i * 3u + 1u), pcg1df(i * 3u + 2u)) * 2.0 - 1.0;
}

// note: input is not clamped to 0-1 range
float smootherstep(float t)
{
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float smoothNoise3D(vec3 p)
{
  uvec3 pI = uvec3(floor(p) + 1048576.0);
  //vec3 pF = fract(p);
  vec3  pF   = vec3(smootherstep(fract(p.x)), smootherstep(fract(p.y)), smootherstep(fract(p.z)));
  float n000 = pcg_noise3D(pI.x, pI.y, pI.z);
  float n010 = pcg_noise3D(pI.x, pI.y + 1u, pI.z);
  float n100 = pcg_noise3D(pI.x + 1u, pI.y, pI.z);
  float n110 = pcg_noise3D(pI.x + 1u, pI.y + 1u, pI.z);
  float n001 = pcg_noise3D(pI.x, pI.y, pI.z + 1u);
  float n011 = pcg_noise3D(pI.x, pI.y + 1u, pI.z + 1u);
  float n101 = pcg_noise3D(pI.x + 1u, pI.y, pI.z + 1u);
  float n111 = pcg_noise3D(pI.x + 1u, pI.y + 1u, pI.z + 1u);

  float n00 = mix(n000, n001, pF.z);
  float n01 = mix(n010, n011, pF.z);
  float n10 = mix(n100, n101, pF.z);
  float n11 = mix(n110, n111, pF.z);

  float n0 = mix(n00, n01, pF.y);
  float n1 = mix(n10, n11, pF.y);

  return mix(n0, n1, pF.x);
}

vec3 smoothNoiseV3D(vec3 p)
{
  return vec3(smoothNoise3D(p), smoothNoise3D(p + 7.921), smoothNoise3D(p + 31.309)) * 2.0 - 1.0;
}

mat3 rotateX(float theta)
{
  float c = cos(theta);
  float s = sin(theta);
  return mat3(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
}

mat3 rotateY(float theta)
{
  float c = cos(theta);
  float s = sin(theta);
  return mat3(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
}

float fbm3D(vec3 p, int scale, int octaves, float gain)
{
  float total      = 0.0;
  float frequency  = 1.0 / float(scale);
  float amplitude  = gain;
  float lacunarity = 2.0;

  for(int i = 0; i < octaves; ++i)
  {
#if 0
        total += smoothNoise3D(p * frequency) * amplitude;
#else
    mat3 pr = rotateX(pcg1df(uint(i)) * 1024.0) * rotateY(pcg1df(uint(i + 97)) * 1024.0);
    total += smoothNoise3D((pr * p) * frequency) * amplitude;
#endif
    frequency *= lacunarity;
    amplitude *= gain;
  }

  return total;
}

vec3 hash3(vec3 p)
{
  p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p + vec3(1.0, 57.0, 113.0)));
}

// returns {interpolatedPosition, greyscaleCellColor}
vec4 voronoi3D(vec3 pos, float smoothness)
{
  ivec3 cell     = ivec3(floor(pos));
  vec3  localPos = fract(pos);
  float minDist  = 10.0;
  vec3  accum    = vec3(0.0);
  float accum2   = 0.0;
  float weight   = 0.0;

  vec3  hashValues[27];
  float distances[27];
  int   index = 0;

  for(int z = -1; z <= 1; z++)
  {
    for(int y = -1; y <= 1; y++)
    {
      for(int x = -1; x <= 1; x++)
      {
        ivec3 offset       = ivec3(x, y, z);
        ivec3 neighborCell = cell + offset;
        vec3  randomPoint  = hash3(vec3(neighborCell));
        vec3  diff         = vec3(offset) + randomPoint - localPos;
        float dist         = dot(diff, diff);
        hashValues[index]  = randomPoint;
        distances[index]   = dist;
        minDist            = min(minDist, dist);
        index++;
      }
    }
  }

  float sqrtMinDist = sqrt(minDist);

  index = 0;
  for(int z = -1; z <= 1; z++)
  {
    for(int y = -1; y <= 1; y++)
    {
      for(int x = -1; x <= 1; x++)
      {
        ivec3 offset      = ivec3(x, y, z);
        vec3  randomPoint = hashValues[index];
        float dist        = distances[index];
        float w           = pow(smoothstep(sqrtMinDist + smoothness, sqrtMinDist - 0.1, sqrt(dist)), 20.0);
        accum += (vec3(cell) + vec3(offset) + randomPoint) * w;
        accum2 += randomPoint.x * w;
        weight += w;
        index++;
      }
    }
  }

  return vec4(accum / weight, accum2 / weight);
}

float cut(vec3 p, vec3 plane, float a, float b, float c)
{
  float d = max(0.0, dot(p, plane) + c);
  return d * d * a + d * b;
}

// cumulative step function, with random steps
float layers(float x, float w)
{
  float s = pcg1df(uint(x + 1024)) * (1.0 - w);
  return max(0.0, floor(x) + smoothstep(s, s + w, fract(x)));
}

// random spikes in each integer interval
float layerCracks(float x, float rndRange, float width)
{
  float w = 0.15;
  float s = pcg1df(uint(x + 1024)) * rndRange;
  float c = max(0.0, width - abs(fract(x) - s)) * (1.0 / width);
  return c * c;
}

// smooth bump around 0.0
float bump(float x)
{
  return 1.0 / (1.0 + x * x);
}

// transition from 0 to 1 after plane boundary
float plane(vec3 n, float d, vec3 p)
{
  float dist = dot(p, n) - d;
  //return dist * dist * dist;
  return smoothstep(0.0, 0.1, dist);
}

// 3D to 3D perlin noise
vec3 warp(vec3 p)
{
  int s = 1;
  return vec3(fbm3D(p * float(s), s, 1, 0.5), fbm3D((p + vec3(7.0)) * float(s), s, 1, 0.5),
              fbm3D((p + vec3(23.0)) * float(s), s, 1, 0.5));
}

// ignores edges/corners
float cubeDist(vec3 p)
{
  p = abs(p);
  return max(max(p.x, p.y), p.z) + dot(p, p) * 0.2;
}

// layerCracks() in Z and randomly within X for each Z level
float layerCracksZ(vec3 p, float rndRange, float width)
{
  float s     = pcg1df(uint(p.z + 1024.0)) * rndRange;
  float c     = max(0.0, width - abs(fract(p.z) - s)) * (1.0 / width);
  uint  level = uint(p.z + 2048.0 - s);
  float ns    = pcg1df(level) * 1.0 + 0.5;
  return c * c + layerCracks(p.x * ns, rndRange, width * ns);
}

// layerCracks() in Y and randomly within Z then X for each Y level
float layerCracksY(vec3 p, float rndRange, float width)
{
  float s     = pcg1df(uint(p.y + 1024.0)) * rndRange;
  float c     = max(0.0, width - abs(fract(p.y) - s)) * (1.0 / width);
  uint  level = uint(p.y + 2048.0 - s);
  float ns    = pcg1df(level) * 0.7 + 0.5;
  return c * c + layerCracksZ(p * ns, rndRange, width * ns);
}

float scalarFieldTerrain(vec3 p)
{
  float n = -p.y * 0.65;

  float mag = fbm3D(p * 512.0, 512, 3, 0.5) * 1.0 + 0.5;

  float warp = smoothNoise3D(p);
  vec3  wp   = p + vec3(warp) * 0.6 * mag;

  float mn = fbm3D((wp) * 1024.0, 1024, 9, 0.5) * 1.4 * mag;
  n += mn;

  float other = smoothNoise3D(p * 1.0);
  float om    = other * other;
  n += layers((2.0 - p.y) * 2.0, 0.2) * 0.8 * om * om;

  // Turn down edges to avoid cracks between adjacent mountains
  vec2 peak = p.xz - p.xz * p.xz;
  n -= 0.002 * dot(peak, peak);

  return n;
}

float scalarFieldMountain(vec3 p)
{
    float iHeightDensity = 0.775;
    float iLayerFade = 0.85;
    float iLayerMag = 1.525;
    float iWarpMag = 1.175;
    float iFbmMag = 5.5;


    float n = 0.2 - p.y * iHeightDensity;
    
    vec3 warp = smoothNoiseV3D(p);
    
    float mag = fbm3D(p * 512.0, 512, 3, 0.5) * 0.5 + 0.2;

    vec3 wp = p + warp * 0.5 * mag * iWarpMag;

    float mn = fbm3D((wp) * 1024.0, 1024, 9, 0.5) * mag * iFbmMag;
    n += mn;
    
    float vm = max(0.0, -0.2 + smoothNoise3D(p * 0.4 + 5.0));
    vec4 v = voronoi3D(p * vec3(0.2, 0.05, 0.2), 0.4);
    n += v.w * vm * 8.0;

    float layerExtra = 1.0 / (0.1 + 0.5 * length(p.xz));
    float om = smoothNoise3D(rotateX(0.8) * p * 0.6) * iLayerFade;
    float om2 = smoothNoise3D(rotateX(1.8) * p * 0.4) * iLayerFade;
    n += layers((-1.0 - p.y) * 0.7, min(1.0, 0.2 + om * om)) * iLayerMag;
    n -= layers((1.0 + p.y) * 1.7, min(1.0, 0.1 + om2 * om2 * max(0.0, 1.0 - om))) * iLayerMag * layerExtra;

    // Mountain shape
    //n -= dot(p.xz, p.xz) * 0.04;
    //n -= exp(0.05 * dot(p.xz, p.xz));
    //n -= (0.5 * p.x * p.x + pow(p.x * p.x * p.x * 0.1 - p.z, 2)) * 0.15;
    //n += 2 / (2 + 0.2 * p.x * p.x + pow(0.2 * p.x * p.x * p.x - p.z, 2)) - 0.0;
    vec2 peak = p.xz - p.xz * p.xz;
    n -= 0.002 * dot(peak, peak);
    
    return n;
}

float scalarFieldBigCellRock(vec3 p)
{
  vec4  v  = voronoi3D(p * vec3(1.0, 0.2, 1.0) * 10.0, 0.4);
  float vl = max(abs(v.x * (0.4 + pcg1df(0) * 0.6)), max(abs(v.y), abs(v.z)));
  float d  = 0.0 - vl * (0.6 + pcg1df(pc.seed) * 0.6);

  vec4  v2  = voronoi3D(p * 20.0, 0.3);
  float v2d = abs(v2.w) * 1.1 / (0.1 + length(v2.xyz * vec3(1.0, 0.2, 1.0)));
  d += v2d * v.w;

  d += fbm3D(p * 512.0, 512, 8, 0.46) * 5.0;

  //vec3 up = vec3(0.1, 0.8, 0.1); // angled cracks - nicer but looks odd with others rotated
  vec3 up = vec3(0.0, 1.0, 0.0);
  d += layers(1.0 - dot(p, up) * 0.8, 0.2) * 0.5;
  d -= layerCracks(dot(p, up) * 4.0, 0.7, 0.3) * bump(p.y * 3.5 + 5.5) * 0.5;

  // Flatter top
  d -= pow(max(p.y - 0.8, 0.0), 0.25) * 2.0;

  return d;
}

float scalarFieldMediumRock(vec3 p)
{
  float cubes = 99.0;
  vec3  w     = warp(p);
  vec3  r     = vec3(0.0);
  for(int i = 0; i < 5; ++i)
  {
    r += pcg3dfs(i) * 1.0;
    cubes = min(cubes, cubeDist(rotateY(i * 0.3) * (p - r + w)));
  }
  float d = 1.0 - cubes;

#if 0
    for (int i = 0; i < 5; ++i)
    {
        d -= crack(rotateX(i * 0.2) * (p + w).y + i, 0.1) * 0.1;
    }
#endif

  vec3 w2 = warp(p * 0.2 + 10.0);
  d -= layerCracksY(p * 0.2 + w2 * 5.0, 0.5, 0.08) * 0.1;

  d += fbm3D(p * 512.0, 512, 8, 0.46) * 1.3;

  return d;
}

float scalarFieldSmallRock(vec3 p)
{
  float d = 0.8;

  vec3 w = warp(p * 0.6);

  float surf = fbm3D(p * 1024.0, 1024, 9, 0.4);

  for(int i = 0; i < 20; ++i)
  {
    vec3  n    = normalize(pcg3dfs(uint(i)));
    float dist = pcg1df(uint(i + 100)) * 0.5 + 1.0 + surf;
    d -= plane(n, dist, p + w) * 1.0;
  }

  return d;
}

float scalarFieldSmallCellRock(vec3 p)
{
  vec4  v  = voronoi3D(p * vec3(1, 0, 1), 0.8);
  float vl = max(abs(v.x * 0.6), max(abs(v.y), abs(v.z)));
  float d  = 2.0 - vl * 0.8 - pow(abs(p.y), 2.0);
  d -= fbm3D(p * 512, 512, 8, 0.46) * 0.5;
  return d;
}

float scalarField(vec3 p)
{
#if defined(TERRAIN)
  return scalarFieldTerrain(p * 4.0);
#elif defined(MOUNTAIN)
  return scalarFieldMountain((p + vec3(0, 0.6, 0)) * 8.6);
#elif defined(ROCK1)
  return scalarFieldBigCellRock(p * 3.0);
#elif defined(ROCK2)
  return scalarFieldMediumRock(p * 5.0);
#elif defined(ROCK3)
  return scalarFieldSmallRock(p * 5.0);
#elif defined(ROCK4)
  return scalarFieldSmallCellRock(p * 5.0);
#else
  return 1.0 - length(p);
#endif
}

vec3 getNormal(vec3 p, float epsilon)
{
  return -normalize(vec3(scalarField(p + vec3(epsilon, 0.0, 0.0)) - scalarField(p - vec3(epsilon, 0.0, 0.0)),
                         scalarField(p + vec3(0.0, epsilon, 0.0)) - scalarField(p - vec3(0.0, epsilon, 0.0)),
                         scalarField(p + vec3(0.0, 0.0, epsilon)) - scalarField(p - vec3(0.0, 0.0, epsilon))));
}

uint8_t triangleCounts(uint32_t i)
{
  Uint16Array triangleOffsets = Uint16Array(pc.triangleOffsets);
  return uint8_t(triangleOffsets.array[i + 1] - triangleOffsets.array[i]);
}

uint8_t triangleTable(uint32_t caseIndex, uint32_t jLocal)
{
  Uint16Array triangleOffsets = Uint16Array(pc.triangleOffsets);
  Uint8Array  table           = Uint8Array(pc.triangleTable);
  uint        idx             = triangleOffsets.array[caseIndex] + jLocal;  // 4-bit packed indices
  uint        byteIndex       = idx >> 1;
  uint        nibbleShift     = (idx & 1u) * 4u;
  return uint8_t((table.array[byteIndex] >> nibbleShift) & 0xFu);
}

void processCell(uvec3 cell, vec3 cellOffset, vec3 cellSize)
{
  U8Vec2Array    edgeVertexIndices = U8Vec2Array(pc.edgeVertexIndices);
  Uint16Array    edgeMasks         = Uint16Array(pc.edgeMasks);
  MutUVec3Array  outTriangles      = MutUVec3Array(pc.outTriangles);
  MutVec3Array   outPositions      = MutVec3Array(pc.outPositions);
  MutVec3Array   outNormals        = MutVec3Array(pc.outNormals);
  MutUint32Array outTriangleCount  = MutUint32Array(pc.outTriangleCount);
  MutUint32Array outVertexCount    = MutUint32Array(pc.outVertexCount);

  // Corner positions
  vec3  cornerPos[8];
  float cornerVal[8];
  vec3  offsets[8] = vec3[8](vec3(0, 0, 0), vec3(1, 0, 0), vec3(1, 0, 1), vec3(0, 0, 1), vec3(0, 1, 0), vec3(1, 1, 0),
                            vec3(1, 1, 1), vec3(0, 1, 1));
  for(int c = 0; c < 8; ++c)
  {
    cornerPos[c] = cellOffset + (cell + offsets[c]) * cellSize;
    cornerVal[c] = scalarField(cornerPos[c]);
  }

  // Build case index
  uint caseIndex = 0u;
  for(uint c = 0u; c < 8u; ++c)
  {
    caseIndex |= (cornerVal[c] < 0.0) ? (1u << c) : 0u;
  }

  uint16_t mask = edgeMasks.array[caseIndex];
  if(mask == 0u)
    return;

  // Number of triangle indices for this case; each triangle uses 3 indices
  uint triIndexBase  = Uint16Array(pc.triangleOffsets).array[caseIndex];
  uint triIndexCount = uint(triangleCounts(caseIndex));
  uint triCount      = triIndexCount / 3u;

  for(uint t = 0; t < triCount; ++t)
  {
    uint vidx[3];
    for(uint k = 0; k < 3u; ++k)
    {
      uint   edge = triangleTable(caseIndex, t * 3u + k);
      u8vec2 ends = edgeVertexIndices.array[edge];
      uint   a    = uint(ends.x);
      uint   b    = uint(ends.y);
      float  va   = cornerVal[a];
      float  vb   = cornerVal[b];

      // Binary search for a better isosurface crossing, largely to get a nicer
      // normal
      float tMin = 0.0;
      float tMax = 1.0;
      for(int i = 0; i < 8; ++i)
      {
        float tMid = (tMin + tMax) * 0.5;
        vec3  pMid = mix(cornerPos[a], cornerPos[b], tMid);
        float vMid = scalarField(pMid);
        if((vMid < 0.0) == (cornerVal[a] < 0.0))
          tMin = tMid;
        else
          tMax = tMid;
      }
      float tEdge = (tMin + tMax) * 0.5;

      vec3 pos    = mix(cornerPos[a], cornerPos[b], tEdge);
      vec3 normal = getNormal(pos, cellSize.x * 0.01);

      uint vOut = atomicAdd(outVertexCount.array[0], 1u);
      if(vOut < pc.maxVertices)
      {
        outPositions.array[vOut] = pos;
        outNormals.array[vOut]   = normal;
      }
      vidx[k] = vOut;
    }

    uint triOut = atomicAdd(outTriangleCount.array[0], 1u);
    if(triOut < pc.maxTriangles)
    {
      outTriangles.array[triOut] = uvec3(vidx[0], vidx[2], vidx[1]);
    }
  }
}

void main()
{
  uvec3 cell  = gl_GlobalInvocationID.xyz;
  uvec3 cells = gl_NumWorkGroups * gl_WorkGroupSize;
  processCell(cell, vec3(-1.0), vec3(2.0) / vec3(cells));
}
