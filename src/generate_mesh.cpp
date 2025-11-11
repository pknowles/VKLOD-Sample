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

#include <algorithm>
#include <array>
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <generate_mesh.hpp>
#include <glm/gtc/noise.hpp>
#include <map>
#include <sample_vulkan_objects.hpp>
#include <vulkan/vulkan_core.h>

// Scoped profiler for quick and coarse results
// https://stackoverflow.com/questions/31391914/timing-in-an-elegant-way-in-c
class Stopwatch
{
public:
  Stopwatch(std::string name)
      : m_name(std::move(name))
      , m_beg(std::chrono::high_resolution_clock::now())
  {
  }
  ~Stopwatch()
  {
    try
    {
      auto end = std::chrono::high_resolution_clock::now();
      auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_beg);
      std::cout << m_name << " : " << dur.count() << " ms\n";
    }
    catch(const std::exception& e)
    {
      // Print and ignore, to satisfy static analysis
      std::cerr << "Error in Stopwatch: " << e.what() << "\n";
    }
  }

private:
  std::string                                                 m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
};

// Specialize std::less for GLM vector types (similar to how std::hash works)
namespace std {
template <>
struct less<glm::vec3>
{
  bool operator()(const glm::vec3& a, const glm::vec3& b) const
  {
    if(a.x != b.x)
      return a.x < b.x;
    if(a.y != b.y)
      return a.y < b.y;
    return a.z < b.z;
  }
};

template <>
struct less<glm::ivec3>
{
  bool operator()(const glm::ivec3& a, const glm::ivec3& b) const
  {
    if(a.x != b.x)
      return a.x < b.x;
    if(a.y != b.y)
      return a.y < b.y;
    return a.z < b.z;
  }
};
}  // namespace std

// 4-bit container for a marching cube table
template <size_t N>
struct uint4_array
{
  static constexpr size_t k_size  = N;
  static constexpr size_t k_bytes = (N + 1) / 2;
  template <std::integral T>
  constexpr uint4_array(const std::array<T, N>& init)
  {
    m_data.fill(0);
    for(size_t i = 0; i < N; ++i)
    {
      const auto& v = init[i];
      assert((v & 0xF) == v);
      if((i & 1) == 0)
        m_data[i >> 1] |= static_cast<uint8_t>(v & 0xF);
      else
        m_data[i >> 1] |= static_cast<uint8_t>((v & 0xF) << 4);
    }
  }
  constexpr uint8_t     operator[](size_t i) const { return (m_data[i >> 1] >> ((i & 1) << 2)) & 0xF; }
  constexpr const void* data() const { return static_cast<const void*>(m_data.data()); }
  constexpr size_t      size() const { return k_size; }
  constexpr size_t      size_bytes() const { return k_bytes; }

private:
  std::array<uint8_t, k_bytes> m_data{};
};

// Marching cube tables from https://github.com/nihaljn/marching-cubes/blob/main/include/marching_cubes.hpp

// Pairs of vertex indices for each edge on the cube
static constexpr glm::u8vec2 g_edgeVertexIndices[] = {
    {0, 1}, {1, 2}, {2, 3}, {0, 3}, {4, 5}, {5, 6}, {6, 7}, {4, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

// A table of cube edges to split (encoded in 12 bits) for each MC case/cube
// index. The case is the bitmask of 8 vertices that are solid/not-solid.
static constexpr uint16_t g_edgeMasks[] = {
    0x0,   0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99,  0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33,  0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa,  0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66,  0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff,  0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55,  0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,  0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc,  0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55,  0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0xff,  0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66,  0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa,  0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33,  0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99,  0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
};

// Offsets into g_triangleTable for each MC case/cube index, including the total
// count at the end.
static constexpr uint16_t g_triangleOffsets[] = {
    0,    0,    3,    6,    12,   15,   21,   27,   36,   39,   45,   51,   60,   66,   75,   84,   90,   93,   99,
    105,  114,  120,  129,  138,  150,  156,  165,  174,  186,  195,  207,  219,  228,  231,  237,  243,  252,  258,
    267,  276,  288,  294,  303,  312,  324,  333,  345,  357,  366,  372,  381,  390,  396,  405,  417,  429,  438,
    447,  459,  471,  480,  492,  507,  522,  528,  531,  537,  543,  552,  558,  567,  576,  588,  594,  603,  612,
    624,  633,  645,  657,  666,  672,  681,  690,  702,  711,  723,  735,  750,  759,  771,  783,  798,  810,  825,
    840,  852,  858,  867,  876,  888,  897,  909,  915,  924,  933,  945,  957,  972,  984,  999,  1008, 1014, 1023,
    1035, 1047, 1056, 1068, 1083, 1092, 1098, 1110, 1125, 1140, 1152, 1167, 1173, 1185, 1188, 1191, 1197, 1203, 1212,
    1218, 1227, 1236, 1248, 1254, 1263, 1272, 1284, 1293, 1305, 1317, 1326, 1332, 1341, 1350, 1362, 1371, 1383, 1395,
    1410, 1419, 1425, 1437, 1446, 1458, 1467, 1482, 1488, 1494, 1503, 1512, 1524, 1533, 1545, 1557, 1572, 1581, 1593,
    1605, 1620, 1632, 1647, 1662, 1674, 1683, 1695, 1707, 1716, 1728, 1743, 1758, 1770, 1782, 1791, 1806, 1812, 1827,
    1839, 1845, 1848, 1854, 1863, 1872, 1884, 1893, 1905, 1917, 1932, 1941, 1953, 1965, 1980, 1986, 1995, 2004, 2010,
    2019, 2031, 2043, 2058, 2070, 2085, 2100, 2106, 2118, 2127, 2142, 2154, 2163, 2169, 2181, 2184, 2193, 2205, 2217,
    2232, 2244, 2259, 2268, 2280, 2292, 2307, 2322, 2328, 2337, 2349, 2355, 2358, 2364, 2373, 2382, 2388, 2397, 2409,
    2415, 2418, 2427, 2433, 2445, 2448, 2454, 2457, 2460, 2460, 2460,
};

// Linearized lists of triangle triples, separated by g_triangleOffsets
static constexpr uint4_array g_triangleTable{std::to_array(
    {0,  8,  3,  0,  1,  9,  1,  8,  3,  9,  8,  1,  1,  2,  10, 0,  8,  3,  1,  2,  10, 9,  2,  10, 0,  2,  9,  2,  8,
     3,  2,  10, 8,  10, 9,  8,  3,  11, 2,  0,  11, 2,  8,  11, 0,  1,  9,  0,  2,  3,  11, 1,  11, 2,  1,  9,  11, 9,
     8,  11, 3,  10, 1,  11, 10, 3,  0,  10, 1,  0,  8,  10, 8,  11, 10, 3,  9,  0,  3,  11, 9,  11, 10, 9,  9,  8,  10,
     10, 8,  11, 4,  7,  8,  4,  3,  0,  7,  3,  4,  0,  1,  9,  8,  4,  7,  4,  1,  9,  4,  7,  1,  7,  3,  1,  1,  2,
     10, 8,  4,  7,  3,  4,  7,  3,  0,  4,  1,  2,  10, 9,  2,  10, 9,  0,  2,  8,  4,  7,  2,  10, 9,  2,  9,  7,  2,
     7,  3,  7,  9,  4,  8,  4,  7,  3,  11, 2,  11, 4,  7,  11, 2,  4,  2,  0,  4,  9,  0,  1,  8,  4,  7,  2,  3,  11,
     4,  7,  11, 9,  4,  11, 9,  11, 2,  9,  2,  1,  3,  10, 1,  3,  11, 10, 7,  8,  4,  1,  11, 10, 1,  4,  11, 1,  0,
     4,  7,  11, 4,  4,  7,  8,  9,  0,  11, 9,  11, 10, 11, 0,  3,  4,  7,  11, 4,  11, 9,  9,  11, 10, 9,  5,  4,  9,
     5,  4,  0,  8,  3,  0,  5,  4,  1,  5,  0,  8,  5,  4,  8,  3,  5,  3,  1,  5,  1,  2,  10, 9,  5,  4,  3,  0,  8,
     1,  2,  10, 4,  9,  5,  5,  2,  10, 5,  4,  2,  4,  0,  2,  2,  10, 5,  3,  2,  5,  3,  5,  4,  3,  4,  8,  9,  5,
     4,  2,  3,  11, 0,  11, 2,  0,  8,  11, 4,  9,  5,  0,  5,  4,  0,  1,  5,  2,  3,  11, 2,  1,  5,  2,  5,  8,  2,
     8,  11, 4,  8,  5,  10, 3,  11, 10, 1,  3,  9,  5,  4,  4,  9,  5,  0,  8,  1,  8,  10, 1,  8,  11, 10, 5,  4,  0,
     5,  0,  11, 5,  11, 10, 11, 0,  3,  5,  4,  8,  5,  8,  10, 10, 8,  11, 9,  7,  8,  5,  7,  9,  9,  3,  0,  9,  5,
     3,  5,  7,  3,  0,  7,  8,  0,  1,  7,  1,  5,  7,  1,  5,  3,  3,  5,  7,  9,  7,  8,  9,  5,  7,  10, 1,  2,  10,
     1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3,  8,  0,  2,  8,  2,  5,  8,  5,  7,  10, 5,  2,  2,  10, 5,  2,  5,  3,
     3,  5,  7,  7,  9,  5,  7,  8,  9,  3,  11, 2,  9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7,  11, 2,  3,  11, 0,  1,
     8,  1,  7,  8,  1,  5,  7,  11, 2,  1,  11, 1,  7,  7,  1,  5,  9,  5,  8,  8,  5,  7,  10, 1,  3,  10, 3,  11, 5,
     7,  0,  5,  0,  9,  7,  11, 0,  1,  0,  10, 11, 10, 0,  11, 10, 0,  11, 0,  3,  10, 5,  0,  8,  0,  7,  5,  7,  0,
     11, 10, 5,  7,  11, 5,  10, 6,  5,  0,  8,  3,  5,  10, 6,  9,  0,  1,  5,  10, 6,  1,  8,  3,  1,  9,  8,  5,  10,
     6,  1,  6,  5,  2,  6,  1,  1,  6,  5,  1,  2,  6,  3,  0,  8,  9,  6,  5,  9,  0,  6,  0,  2,  6,  5,  9,  8,  5,
     8,  2,  5,  2,  6,  3,  2,  8,  2,  3,  11, 10, 6,  5,  11, 0,  8,  11, 2,  0,  10, 6,  5,  0,  1,  9,  2,  3,  11,
     5,  10, 6,  5,  10, 6,  1,  9,  2,  9,  11, 2,  9,  8,  11, 6,  3,  11, 6,  5,  3,  5,  1,  3,  0,  8,  11, 0,  11,
     5,  0,  5,  1,  5,  11, 6,  3,  11, 6,  0,  3,  6,  0,  6,  5,  0,  5,  9,  6,  5,  9,  6,  9,  11, 11, 9,  8,  5,
     10, 6,  4,  7,  8,  4,  3,  0,  4,  7,  3,  6,  5,  10, 1,  9,  0,  5,  10, 6,  8,  4,  7,  10, 6,  5,  1,  9,  7,
     1,  7,  3,  7,  9,  4,  6,  1,  2,  6,  5,  1,  4,  7,  8,  1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7,  8,  4,
     7,  9,  0,  5,  0,  6,  5,  0,  2,  6,  7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9,  3,  11, 2,  7,
     8,  4,  10, 6,  5,  5,  10, 6,  4,  7,  2,  4,  2,  0,  2,  7,  11, 0,  1,  9,  4,  7,  8,  2,  3,  11, 5,  10, 6,
     9,  2,  1,  9,  11, 2,  9,  4,  11, 7,  11, 4,  5,  10, 6,  8,  4,  7,  3,  11, 5,  3,  5,  1,  5,  11, 6,  5,  1,
     11, 5,  11, 6,  1,  0,  11, 7,  11, 4,  0,  4,  11, 0,  5,  9,  0,  6,  5,  0,  3,  6,  11, 6,  3,  8,  4,  7,  6,
     5,  9,  6,  9,  11, 4,  7,  9,  7,  11, 9,  10, 4,  9,  6,  4,  10, 4,  10, 6,  4,  9,  10, 0,  8,  3,  10, 0,  1,
     10, 6,  0,  6,  4,  0,  8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1,  10, 1,  4,  9,  1,  2,  4,  2,  6,  4,  3,  0,
     8,  1,  2,  9,  2,  4,  9,  2,  6,  4,  0,  2,  4,  4,  2,  6,  8,  3,  2,  8,  2,  4,  4,  2,  6,  10, 4,  9,  10,
     6,  4,  11, 2,  3,  0,  8,  2,  2,  8,  11, 4,  9,  10, 4,  10, 6,  3,  11, 2,  0,  1,  6,  0,  6,  4,  6,  1,  10,
     6,  4,  1,  6,  1,  10, 4,  8,  1,  2,  1,  11, 8,  11, 1,  9,  6,  4,  9,  3,  6,  9,  1,  3,  11, 6,  3,  8,  11,
     1,  8,  1,  0,  11, 6,  1,  9,  1,  4,  6,  4,  1,  3,  11, 6,  3,  6,  0,  0,  6,  4,  6,  4,  8,  11, 6,  8,  7,
     10, 6,  7,  8,  10, 8,  9,  10, 0,  7,  3,  0,  10, 7,  0,  9,  10, 6,  7,  10, 10, 6,  7,  1,  10, 7,  1,  7,  8,
     1,  8,  0,  10, 6,  7,  10, 7,  1,  1,  7,  3,  1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7,  2,  6,  9,  2,  9,
     1,  6,  7,  9,  0,  9,  3,  7,  3,  9,  7,  8,  0,  7,  0,  6,  6,  0,  2,  7,  3,  2,  6,  7,  2,  2,  3,  11, 10,
     6,  8,  10, 8,  9,  8,  6,  7,  2,  0,  7,  2,  7,  11, 0,  9,  7,  6,  7,  10, 9,  10, 7,  1,  8,  0,  1,  7,  8,
     1,  10, 7,  6,  7,  10, 2,  3,  11, 11, 2,  1,  11, 1,  7,  10, 6,  1,  6,  7,  1,  8,  9,  6,  8,  6,  7,  9,  1,
     6,  11, 6,  3,  1,  3,  6,  0,  9,  1,  11, 6,  7,  7,  8,  0,  7,  0,  6,  3,  11, 0,  11, 6,  0,  7,  11, 6,  7,
     6,  11, 3,  0,  8,  11, 7,  6,  0,  1,  9,  11, 7,  6,  8,  1,  9,  8,  3,  1,  11, 7,  6,  10, 1,  2,  6,  11, 7,
     1,  2,  10, 3,  0,  8,  6,  11, 7,  2,  9,  0,  2,  10, 9,  6,  11, 7,  6,  11, 7,  2,  10, 3,  10, 8,  3,  10, 9,
     8,  7,  2,  3,  6,  2,  7,  7,  0,  8,  7,  6,  0,  6,  2,  0,  2,  7,  6,  2,  3,  7,  0,  1,  9,  1,  6,  2,  1,
     8,  6,  1,  9,  8,  8,  7,  6,  10, 7,  6,  10, 1,  7,  1,  3,  7,  10, 7,  6,  1,  7,  10, 1,  8,  7,  1,  0,  8,
     0,  3,  7,  0,  7,  10, 0,  10, 9,  6,  10, 7,  7,  6,  10, 7,  10, 8,  8,  10, 9,  6,  8,  4,  11, 8,  6,  3,  6,
     11, 3,  0,  6,  0,  4,  6,  8,  6,  11, 8,  4,  6,  9,  0,  1,  9,  4,  6,  9,  6,  3,  9,  3,  1,  11, 3,  6,  6,
     8,  4,  6,  11, 8,  2,  10, 1,  1,  2,  10, 3,  0,  11, 0,  6,  11, 0,  4,  6,  4,  11, 8,  4,  6,  11, 0,  2,  9,
     2,  10, 9,  10, 9,  3,  10, 3,  2,  9,  4,  3,  11, 3,  6,  4,  6,  3,  8,  2,  3,  8,  4,  2,  4,  6,  2,  0,  4,
     2,  4,  6,  2,  1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8,  1,  9,  4,  1,  4,  2,  2,  4,  6,  8,  1,  3,  8,
     6,  1,  8,  4,  6,  6,  10, 1,  10, 1,  0,  10, 0,  6,  6,  0,  4,  4,  6,  3,  4,  3,  8,  6,  10, 3,  0,  3,  9,
     10, 9,  3,  10, 9,  4,  6,  10, 4,  4,  9,  5,  7,  6,  11, 0,  8,  3,  4,  9,  5,  11, 7,  6,  5,  0,  1,  5,  4,
     0,  7,  6,  11, 11, 7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5,  9,  5,  4,  10, 1,  2,  7,  6,  11, 6,  11, 7,  1,
     2,  10, 0,  8,  3,  4,  9,  5,  7,  6,  11, 5,  4,  10, 4,  2,  10, 4,  0,  2,  3,  4,  8,  3,  5,  4,  3,  2,  5,
     10, 5,  2,  11, 7,  6,  7,  2,  3,  7,  6,  2,  5,  4,  9,  9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7,  3,  6,
     2,  3,  7,  6,  1,  5,  0,  5,  4,  0,  6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8,  9,  5,  4,  10,
     1,  6,  1,  7,  6,  1,  3,  7,  1,  6,  10, 1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4,  4,  0,  10, 4,  10, 5,
     0,  3,  10, 6,  10, 7,  3,  7,  10, 7,  6,  10, 7,  10, 8,  5,  4,  10, 4,  8,  10, 6,  9,  5,  6,  11, 9,  11, 8,
     9,  3,  6,  11, 0,  6,  3,  0,  5,  6,  0,  9,  5,  0,  11, 8,  0,  5,  11, 0,  1,  5,  5,  6,  11, 6,  11, 3,  6,
     3,  5,  5,  3,  1,  1,  2,  10, 9,  5,  11, 9,  11, 8,  11, 5,  6,  0,  11, 3,  0,  6,  11, 0,  9,  6,  5,  6,  9,
     1,  2,  10, 11, 8,  5,  11, 5,  6,  8,  0,  5,  10, 5,  2,  0,  2,  5,  6,  11, 3,  6,  3,  5,  2,  10, 3,  10, 5,
     3,  5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2,  9,  5,  6,  9,  6,  0,  0,  6,  2,  1,  5,  8,  1,  8,  0,  5,
     6,  8,  3,  8,  2,  6,  2,  8,  1,  5,  6,  2,  1,  6,  1,  3,  6,  1,  6,  10, 3,  8,  6,  5,  6,  9,  8,  9,  6,
     10, 1,  0,  10, 0,  6,  9,  5,  0,  5,  6,  0,  0,  3,  8,  5,  6,  10, 10, 5,  6,  11, 5,  10, 7,  5,  11, 11, 5,
     10, 11, 7,  5,  8,  3,  0,  5,  11, 7,  5,  10, 11, 1,  9,  0,  10, 7,  5,  10, 11, 7,  9,  8,  1,  8,  3,  1,  11,
     1,  2,  11, 7,  1,  7,  5,  1,  0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2,  11, 9,  7,  5,  9,  2,  7,  9,  0,  2,
     2,  11, 7,  7,  5,  2,  7,  2,  11, 5,  9,  2,  3,  2,  8,  9,  8,  2,  2,  5,  10, 2,  3,  5,  3,  7,  5,  8,  2,
     0,  8,  5,  2,  8,  7,  5,  10, 2,  5,  9,  0,  1,  5,  10, 3,  5,  3,  7,  3,  10, 2,  9,  8,  2,  9,  2,  1,  8,
     7,  2,  10, 2,  5,  7,  5,  2,  1,  3,  5,  3,  7,  5,  0,  8,  7,  0,  7,  1,  1,  7,  5,  9,  0,  3,  9,  3,  5,
     5,  3,  7,  9,  8,  7,  5,  9,  7,  5,  8,  4,  5,  10, 8,  10, 11, 8,  5,  0,  4,  5,  11, 0,  5,  10, 11, 11, 3,
     0,  0,  1,  9,  8,  4,  10, 8,  10, 11, 10, 4,  5,  10, 11, 4,  10, 4,  5,  11, 3,  4,  9,  4,  1,  3,  1,  4,  2,
     5,  1,  2,  8,  5,  2,  11, 8,  4,  5,  8,  0,  4,  11, 0,  11, 3,  4,  5,  11, 2,  11, 1,  5,  1,  11, 0,  2,  5,
     0,  5,  9,  2,  11, 5,  4,  5,  8,  11, 8,  5,  9,  4,  5,  2,  11, 3,  2,  5,  10, 3,  5,  2,  3,  4,  5,  3,  8,
     4,  5,  10, 2,  5,  2,  4,  4,  2,  0,  3,  10, 2,  3,  5,  10, 3,  8,  5,  4,  5,  8,  0,  1,  9,  5,  10, 2,  5,
     2,  4,  1,  9,  2,  9,  4,  2,  8,  4,  5,  8,  5,  3,  3,  5,  1,  0,  4,  5,  1,  0,  5,  8,  4,  5,  8,  5,  3,
     9,  0,  5,  0,  3,  5,  9,  4,  5,  4,  11, 7,  4,  9,  11, 9,  10, 11, 0,  8,  3,  4,  9,  7,  9,  11, 7,  9,  10,
     11, 1,  10, 11, 1,  11, 4,  1,  4,  0,  7,  4,  11, 3,  1,  4,  3,  4,  8,  1,  10, 4,  7,  4,  11, 10, 11, 4,  4,
     11, 7,  9,  11, 4,  9,  2,  11, 9,  1,  2,  9,  7,  4,  9,  11, 7,  9,  1,  11, 2,  11, 1,  0,  8,  3,  11, 7,  4,
     11, 4,  2,  2,  4,  0,  11, 7,  4,  11, 4,  2,  8,  3,  4,  3,  2,  4,  2,  9,  10, 2,  7,  9,  2,  3,  7,  7,  4,
     9,  9,  10, 7,  9,  7,  4,  10, 2,  7,  8,  7,  0,  2,  0,  7,  3,  7,  10, 3,  10, 2,  7,  4,  10, 1,  10, 0,  4,
     0,  10, 1,  10, 2,  8,  7,  4,  4,  9,  1,  4,  1,  7,  7,  1,  3,  4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1,
     4,  0,  3,  7,  4,  3,  4,  8,  7,  9,  10, 8,  10, 11, 8,  3,  0,  9,  3,  9,  11, 11, 9,  10, 0,  1,  10, 0,  10,
     8,  8,  10, 11, 3,  1,  10, 11, 3,  10, 1,  2,  11, 1,  11, 9,  9,  11, 8,  3,  0,  9,  3,  9,  11, 1,  2,  9,  2,
     11, 9,  0,  2,  11, 8,  0,  11, 3,  2,  11, 2,  3,  8,  2,  8,  10, 10, 8,  9,  9,  10, 2,  0,  9,  2,  2,  3,  8,
     2,  8,  10, 0,  1,  8,  1,  10, 8,  1,  10, 2,  1,  3,  8,  9,  1,  8,  0,  9,  1,  0,  3,  8})};

// Generates a mesh using the given compiled compute shader
SimpleMesh generateMesh(ResourceAllocator*                                       allocator,
                        vkobj::SimpleComputePipeline<shaders::MeshGenConstants>& pipeline,
                        VkCommandPool                                            pool,
                        vkobj::TimelineQueue&                                    queue,
                        uint32_t                                                 seed,
                        uint32_t                                                 maxTriangles)
{
  // Upload marching cubes lookup tables into device-local buffers
  vkobj::ImmediateCommandBuffer cmd(allocator->getDevice(), pool, queue, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
  constexpr VkBufferUsageFlags  kLutUsage =
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  constexpr VkMemoryPropertyFlags kLutMem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  vkobj::Buffer<glm::u8vec2>      edgeVertexIndicesBuffer(allocator, g_edgeVertexIndices, kLutUsage, kLutMem, cmd);
  vkobj::Buffer<uint16_t>         edgeMasksBuffer(allocator, g_edgeMasks, kLutUsage, kLutMem, cmd);
  vkobj::Buffer<uint16_t>         triangleOffsetsBuffer(allocator, g_triangleOffsets, kLutUsage, kLutMem, cmd);
  std::span<const uint8_t>        triangleTableBytes(reinterpret_cast<const uint8_t*>(g_triangleTable.data()),
                                                     g_triangleTable.size_bytes());
  vkobj::Buffer<uint8_t>          triangleTableBuffer(allocator, triangleTableBytes, kLutUsage, kLutMem, cmd);

  uint32_t maxVertices = maxTriangles * 3;

  // Buffers to hold the written mesh
  vkobj::Buffer<glm::uvec3> trianglesBuffer(allocator, maxTriangles,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkobj::Buffer<glm::vec3>  positionsBuffer(allocator, maxVertices,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkobj::Buffer<glm::vec3>  normalsBuffer(allocator, maxVertices,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkobj::Buffer<uint32_t>   triangleCountsBuffer(allocator, 1,
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                     | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkobj::Buffer<uint32_t>   vertexCountsBuffer(allocator, 1,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                   | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Initialize counters to zero
  vkCmdFillBuffer(cmd, triangleCountsBuffer, 0, sizeof(uint32_t), 0);
  vkCmdFillBuffer(cmd, vertexCountsBuffer, 0, sizeof(uint32_t), 0);

  // Barrier: LUT uploads -> compute read
  memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  float resolutionGuess = sqrtf(float(maxTriangles)) * 0.3f;
  resolutionGuess       = glm::clamp(resolutionGuess, 32.0f, 1024.0f);
  uint32_t resolution   = div_ceil<uint32_t>(uint32_t(resolutionGuess), MESH_GEN_WORKGROUP_SIZE);

  shaders::MeshGenConstants constants{
      .edgeVertexIndices = edgeVertexIndicesBuffer.address(),
      .edgeMasks         = edgeMasksBuffer.address(),
      .triangleOffsets   = triangleOffsetsBuffer.address(),
      .triangleTable     = triangleTableBuffer.address(),
      .outTriangles      = trianglesBuffer.address(),
      .outPositions      = positionsBuffer.address(),
      .outNormals        = normalsBuffer.address(),
      .outTriangleCount  = triangleCountsBuffer.address(),
      .outVertexCount    = vertexCountsBuffer.address(),
      .maxTriangles      = maxTriangles,
      .maxVertices       = maxVertices,
      .seed              = seed,
  };

  // Execute the compute shader
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdPushConstants(cmd, pipeline.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
  vkCmdDispatch(cmd, resolution, resolution, resolution);

  // Barrier: compute write -> transfer/host read
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  std::span<const uint32_t> triangleCountMapping = vkobj::cmdStagedDownload(*allocator->getStaging(), cmd, triangleCountsBuffer);
  std::span<const uint32_t> vertexCountMapping = vkobj::cmdStagedDownload(*allocator->getStaging(), cmd, vertexCountsBuffer);

  // Submit the command buffer now just to get the counts and make a new command
  // buffer to download the mesh. The move assignment operator would trigger a
  // submit anyway, but good to be explicit.
  cmd.submit();
  cmd = vkobj::ImmediateCommandBuffer(allocator->getDevice(), pool, queue, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  if(triangleCountMapping[0] > maxTriangles)
  {
    LOGW("Triangle count %d exceeds maxTriangles %d\n", triangleCountMapping[0], maxTriangles);
  }
  if(vertexCountMapping[0] > maxVertices)
  {
    LOGW("Vertex count %d exceeds maxVertices %d\n", vertexCountMapping[0], maxVertices);

    // unrecoverable error
    throw std::runtime_error("Mesh generation failed");
  }

  uint32_t triangleCount = std::min<uint32_t>(triangleCountMapping[0], maxTriangles);
  uint32_t vertexCount   = std::min<uint32_t>(vertexCountMapping[0], maxVertices);

  std::span<const glm::vec3> positions = vkobj::cmdStagedDownload(*allocator->getStaging(), cmd, positionsBuffer, vertexCount);
  std::span<const glm::vec3> normals = vkobj::cmdStagedDownload(*allocator->getStaging(), cmd, normalsBuffer, vertexCount);
  std::span<const glm::uvec3> triangles = vkobj::cmdStagedDownload(*allocator->getStaging(), cmd, trianglesBuffer, triangleCount);

  // Final submit to transfer mesh data
  cmd.submit();

  // Uniquify vertices. Just based on position as there are no sharp edges, i.e.
  // vertices at the same position have the same normal
  std::map<glm::vec3, uint32_t> uniquePositions;
  std::vector<glm::uvec3>       mutableTriangles(triangles.begin(), triangles.end());
  std::vector<glm::vec3>        uniqueNormalsVector;
  std::vector<glm::vec3>        uniquePositionsVector;
  for(auto& t : mutableTriangles)
  {
    for(uint32_t i = 0; i < 3; ++i)
    {
      auto [it, inserted] = uniquePositions.try_emplace(positions[t[i]], uint32_t(uniquePositionsVector.size()));
      if(inserted)
      {
        uniquePositionsVector.push_back(positions[t[i]]);
        uniqueNormalsVector.push_back(normals[t[i]]);
      }
      t[i] = it->second;
    }
  }

  printf("  Desired triangles: %u, Actual: %zu (%0.2f%%)\n", maxTriangles, mutableTriangles.size(),
         float(mutableTriangles.size()) / float(maxTriangles) * 100.0f);

  return SimpleMesh{
      .triangles = std::move(mutableTriangles),
      .positions = std::move(uniquePositionsVector),
      .normals   = std::move(uniqueNormalsVector),
  };
}

// 3D Fractal Brownian Motion for placing instances
float fbm(glm::vec3 p, int octaves, float lacunarity, float gain)
{
  float total         = 0.0f;
  float frequency     = 1.0f;
  float amplitude     = 1.0f;
  float normalisation = 0.0f;

  for(int i = 0; i < std::max(octaves, 1); ++i)
  {
    total += glm::perlin(p * frequency) * amplitude;
    normalisation += amplitude;
    frequency *= lacunarity;
    amplitude *= gain;
  }

  // Prevent division by zero
  if(normalisation == 0.0f)
  {
    return 0.0f;
  }

  return total / normalisation;
}

// Poisson disc distribution using Bridson's algorithm (AI generated)
std::vector<glm::vec2> poissonDisc(float radius, glm::vec2 bounds, uint32_t maxAttempts = 30)
{
  std::vector<glm::vec2> points;
  std::vector<glm::vec2> activeList;

  float cellSize = radius / std::sqrt(2.0f);

  std::unordered_map<glm::ivec2, int> grid;  // Maps grid cell -> point index

  auto gridCell = [cellSize](glm::vec2 p) -> glm::ivec2 { return glm::ivec2(glm::floor(p / cellSize)); };

  // Start with random point
  glm::vec2 initial = glm::vec2(glm::linearRand(0.0f, bounds.x), glm::linearRand(0.0f, bounds.y));
  points.push_back(initial);
  activeList.push_back(initial);
  grid[gridCell(initial)] = 0;

  while(!activeList.empty())
  {
    int       activeIndex = glm::linearRand(0, int(activeList.size()) - 1);
    glm::vec2 activePoint = activeList[activeIndex];
    bool      found       = false;

    for(uint32_t attempt = 0; attempt < maxAttempts; ++attempt)
    {
      float     angle     = glm::linearRand(0.0f, glm::two_pi<float>());
      float     distance  = glm::linearRand(radius, 2.0f * radius);
      glm::vec2 candidate = activePoint + glm::vec2(std::cos(angle), std::sin(angle)) * distance;

      if(candidate.x < 0 || candidate.x >= bounds.x || candidate.y < 0 || candidate.y >= bounds.y)
        continue;

      bool       valid = true;
      glm::ivec2 cell  = gridCell(candidate);

      for(int dy = -2; dy <= 2 && valid; ++dy)
      {
        for(int dx = -2; dx <= 2 && valid; ++dx)
        {
          glm::ivec2 neighborCell = cell + glm::ivec2(dx, dy);
          auto       it           = grid.find(neighborCell);
          if(it != grid.end() && glm::distance(points[it->second], candidate) < radius)
          {
            valid = false;
          }
        }
      }

      if(valid)
      {
        points.push_back(candidate);
        activeList.push_back(candidate);
        grid[gridCell(candidate)] = int(points.size()) - 1;
        found                     = true;
        break;
      }
    }

    if(!found)
    {
      activeList.erase(activeList.begin() + activeIndex);
    }
  }

  return points;
}

// Hashed grid of vertex normal samples, used to find flat places to spawn
// instances
struct GeometrySampler
{
  struct Samples
  {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
  };

  GeometrySampler(const SimpleMesh& mesh, float cellSize, std::vector<glm::mat4>&& instanceTransforms)
      : instanceTransforms(std::move(instanceTransforms))
      , cellSize(cellSize)
  {
    // Sample a small selection of normals and add them to a grid
    size_t                        maxSamples    = 16;
    size_t                        expectedCells = size_t(2.0f / cellSize);
    std::map<glm::ivec3, Samples> cellSamples;
    size_t skip = std::max(mesh.positions.size() / (maxSamples * expectedCells * expectedCells), size_t(1));
    for(size_t v = 0; v < mesh.positions.size(); v += skip)
    {
      for(int x = -1; x <= 1; ++x)
      {
        for(int y = -1; y <= 1; ++y)
        {
          for(int z = -1; z <= 1; ++z)
          {
            glm::ivec3 offset  = glm::ivec3(x, y, z);
            auto&      samples = cellSamples[posToGrid(mesh.positions[v]) + offset];

            // Randomly sample positions in the current cell
            if(offset == glm::ivec3(0, 0, 0))
            {
              if(samples.positions.size() < maxSamples)
              {
                samples.positions.push_back(mesh.positions[v]);
              }
              else
              {
                size_t index             = glm::linearRand(0u, uint32_t(samples.positions.size() - 1));
                samples.positions[index] = mesh.positions[v];
              }
            }

            // Randomly sample normals in the current and surrounding cells
            {
              if(samples.normals.size() < maxSamples)
              {
                samples.normals.push_back(mesh.normals[v]);
              }
              else
              {
                size_t index           = glm::linearRand(0u, uint32_t(samples.normals.size() - 1));
                samples.normals[index] = mesh.normals[v];
              }
            }
          }
        }
      }
    }

    // Compute the standard deviation of the normals in each cell and use it as a
    // flatness metric for places to spawn rocks
    for(const auto& [grid, samples] : cellSamples)
    {
      if(samples.positions.empty() || samples.normals.size() < 5)
        continue;
#if 0
      glm::vec3 mean = glm::vec3(0.0f);
      for(const auto& n : samples.normals)
        mean += n;
      mean = glm::normalize(mean);
#endif
      glm::vec3 up     = glm::vec3(0.0f, 1.0f, 0.0f);
      float     stdDev = 0.0f;
      for(const auto& n : samples.normals)
        stdDev += powf(acosf(std::fabs(dot(n, up))), 2.0f);  // abs() for either up or down
      stdDev          = sqrtf(stdDev / float(samples.normals.size()));
      float flattness = 1.0f / (powf(stdDev, 2.0f) + 1e-6f);
      if(flattness > 4.0f)  // filter out not-flat places
      {
        //printf("flattness: %f, mean: %f, %f, %f\n", flattness, mean.x, mean.y, mean.z);
        cellPositions.push_back(std::accumulate(samples.positions.begin(), samples.positions.end(), glm::vec3(0.0f))
                                / float(samples.positions.size()));

        // Artificially add some noise to the flatness to make instancing
        // marginally more interesting
        flattness *= fbm(cellPositions.back() * 512.0f, 8, 2.0f, 0.5f) + 1.0f;
        assert(flattness >= 0.0f);

        cellCumulativeFlatness.push_back(cellCumulativeFlatness.back() + flattness);
      }
    }

    if(cellPositions.empty())
    {
      throw std::runtime_error("No flat positions in mesh");
    }
  }

  // Pick a random grid cell, weighted towards places with flat terrain
  glm::vec3 pickFlatPosition() const
  {
    glm::vec3 pos{};

    // Pick a random instance transform
    assert(!instanceTransforms.empty());
    size_t    instanceIndex     = glm::linearRand(0u, uint32_t(instanceTransforms.size() - 1));
    glm::mat4 instanceTransform = instanceTransforms[instanceIndex];

    // Try multiple times since some may already be excluded by other instances.
    // Ideally those cells would be removed from the options, but this is
    // simpler to code.
    for(size_t i = 0; i < 16; ++i)
    {
      float  r     = glm::linearRand(0.0f, cellCumulativeFlatness.back() - 1e-6f);
      auto   it    = std::upper_bound(cellCumulativeFlatness.begin(), cellCumulativeFlatness.end(), r) - 1;
      size_t index = it - cellCumulativeFlatness.begin();
      assert(it != cellCumulativeFlatness.end());
      assert(index < cellPositions.size());
      pos = glm::vec3(instanceTransform * glm::vec4(cellPositions[index], 1.0f));

      // Try to avoid placing instances on top of each other
      std::optional<glm::vec4> excluded;
      for(size_t j = 0; j < 5; ++j)
      {
        excluded = excludedBy(pos);
        if(i < 10 || !excluded)
          break;

        // Move the position in a random direction to avoid the found overlap
        glm::vec2 offset = glm::circularRand(excluded.value().w + 1e-6f);
        pos              = glm::vec3(excluded.value().x + offset.x, pos.y, excluded.value().z + offset.y);
      }
      if(!excluded)
        break;

      if(i == 15)
        printf("WARNING: No good flat position found while generating instances\n");
    }
    return pos;
  };

  // Avoid placing instances on top of each other
  void exclude(const glm::vec3& pos, float radius) { excludedCylinders.push_back(glm::vec3(pos.x, pos.z, radius)); }

  // Brute force check for overlapping positions (just on the XZ plane)
  std::optional<glm::vec4> excludedBy(const glm::vec3& pos) const
  {
    for(const auto& cylinder : excludedCylinders)
    {
      if(glm::length(glm::vec2(pos.x, pos.z) - glm::vec2(cylinder)) < cylinder.z)
        return glm::vec4(cylinder.x, pos.y, cylinder.y, cylinder.z);
    }
    return std::nullopt;
  }

  glm::ivec3 posToGrid(const glm::vec3& pos) const { return glm::ivec3(glm::floor(pos / cellSize)); }
  glm::vec3  gridToPos(const glm::ivec3& grid) const { return (glm::vec3(grid) + 0.5f) * cellSize; }

  std::vector<glm::vec3> cellPositions;
  std::vector<float>     cellCumulativeFlatness{0.0f};
  std::vector<glm::vec3> excludedCylinders;
  std::vector<glm::mat4> instanceTransforms;
  float                  cellSize;
};

// Random rotation around Y, with a small tilt around X and Z
glm::mat4 randomRotation(float maxTilt)
{
  float yaw   = glm::linearRand(0.0f, glm::two_pi<float>());
  float pitch = glm::linearRand(-maxTilt, maxTilt);
  float roll  = glm::linearRand(-maxTilt, maxTilt);
  return glm::eulerAngleXYZ(pitch, yaw, roll);
}

// Hard coded scene generation, using a compute shader to create terrain and
// rock meshes and then placing rock instances randomly on the terrain
GeneratedScene makeTerrainAndRocksScene(VkDevice              device,
                                        SampleGlslCompiler&   glslCompiler,
                                        ResourceAllocator*    allocator,
                                        VkCommandPool         pool,
                                        vkobj::TimelineQueue& queue,
                                        float                 detailScale,
                                        TaskProgress&         progress)
{
  Stopwatch          sw("Generate Scene");
  GeneratorPipelines m_generators(device, glslCompiler);
  GeneratedScene     result;
  std::srand(0);

  uint32_t numBaseTerrainTriangles = uint32_t(float(1u << 22) * detailScale);
  uint32_t numMountains            = 2;
  uint32_t numMountainTriangles    = uint32_t(float(1u << 22) * detailScale);
  uint32_t numRock1s               = 3;
  uint32_t numRock1Triangles       = uint32_t(float(1u << 21) * detailScale);
  uint32_t numRock2s               = 5;
  uint32_t numRock2Triangles       = uint32_t(float(1u << 20) * detailScale);
  uint32_t numRock3s               = 4;
  uint32_t numRock3Triangles       = uint32_t(float(1u << 19) * detailScale);
  uint32_t numRock4s               = 2;
  uint32_t numRock4Triangles       = uint32_t(float(1u << 19) * detailScale);

  // Attempts to convert a triangle count to a time-proportional estimate for
  // processing LOD. This equation is computed and printed by TaskProgress.
  auto workEstimate = [](uint32_t numTriangles) {
    // fits linear best: time = 8.5805584e-07 complexity + -0.34479564
    return 8.5805584e-07f * float(numTriangles) - 0.34479564f;
  };
  progress.startSubtask(workEstimate(numBaseTerrainTriangles) * 2.0f                       //
                        + workEstimate(numMountainTriangles) * float(numMountains) * 2.0f  //
                        + workEstimate(numRock1Triangles) * float(numRock1s)               //
                        + workEstimate(numRock2Triangles) * float(numRock2s)               //
                        + workEstimate(numRock3Triangles) * float(numRock3s)               //
                        + workEstimate(numRock4Triangles) * float(numRock4s));

  // Base terrain
  LOGI("Generating base terrain\n");
  progress.setSubtaskWorkName("terrain geometry generation");
  result.meshes.push_back(generateMesh(allocator, m_generators.terrain, pool, queue, 0, numBaseTerrainTriangles));
  progress.makeProgress(workEstimate(numBaseTerrainTriangles));
  result.instances.push_back({uint32_t(0), glm::mat4(1.0f)});

  // Samplers for picking rock positions on the base terrain tile
  progress.setSubtaskWorkName("populate terrain samplers");
  GeometrySampler              samplerTerrain(result.meshes.back(), 2.0f / 256.0f, {result.instances.back().second});
  std::vector<GeometrySampler> mountainSamplers;
  progress.makeProgress(workEstimate(numBaseTerrainTriangles));

  // Pick a random spot on either the base terrain or a mountain
  auto pickFlatPosition = [&](float excludeRadius) {
    uint32_t  index = glm::linearRand(0u, uint32_t(mountainSamplers.size()));
    glm::vec3 pos;
    if(index >= uint32_t(mountainSamplers.size()))
    {
      pos = samplerTerrain.pickFlatPosition();
      samplerTerrain.exclude(pos, excludeRadius);
    }
    else
    {
      pos = mountainSamplers[index].pickFlatPosition();
      mountainSamplers[index].exclude(pos, excludeRadius);
    }
    return pos;
  };

// Mountains
#if 1
  for(uint32_t i = 0; i < numMountains; ++i)
  {
    LOGI("Generating mountain %d\n", i);
    progress.setSubtaskWorkName(fmt::format("mountain {} geometry generation", i));
    result.meshes.push_back(generateMesh(allocator, m_generators.mountain, pool, queue, i, numMountainTriangles));
    progress.progressTo(workEstimate(numMountainTriangles), fmt::format("mountain {} sampler population", i));
    mountainSamplers.push_back(GeometrySampler(result.meshes.back(), 2.0f / 256.0f, {/* instances populated below */}));
    progress.makeProgress(workEstimate(numMountainTriangles));  // sampler population
  }
  AABB mountainAaabb = computeBounds(result.meshes.back().positions);
  for(glm::vec2 pos : poissonDisc(0.08f, glm::vec2(1.0f), 30))
  {
    pos = pos * 2.0f - glm::vec2(1.0f);
    pos *= 4.0f;

    float distance = glm::length(pos);
    float scaleMin = 0.4f + distance * 0.15f;
    float scaleMax = 0.5f + distance * 0.2f;
    float scale    = glm::linearRand(scaleMin, scaleMax);

    pos *= scaleMax;

    if(std::abs(pos.x) < 0.9f && std::abs(pos.y) < 0.9f)
      continue;
    if(glm::length(pos) > 4.0f)
      continue;
    float     height      = std::max(0.0f, (glm::length(pos) - 1.0f) * 0.1f);
    glm::vec3 translation = glm::vec3(pos.x, glm::linearRand(height * 0.7f, height) + 0.1f, pos.y);

    glm::mat4 model = glm::translate(glm::mat4(1.0f), translation);
    model *= randomRotation(0.0f);
    model = glm::scale(model, glm::vec3(scale));
    model = glm::translate(model, glm::vec3(0.0f, -mountainAaabb.min.y, 0.0f));
    size_t mountainIndex = glm::linearRand(0u, uint32_t(mountainSamplers.size() - 1));
    result.instances.push_back({uint32_t(result.meshes.size() - numMountains + mountainIndex), model});
    mountainSamplers[mountainIndex].instanceTransforms.push_back(model);
  }
#endif

// Big rocks
#if 1
  for(uint32_t i = 0; i < numRock1s; ++i)
  {
    LOGI("Generating rock1 %d\n", i);
    progress.setSubtaskWorkName(fmt::format("rock1-{} geometry generation", i));
    result.meshes.push_back(generateMesh(allocator, m_generators.rock1, pool, queue, i, numRock1Triangles));
    progress.progressTo(workEstimate(numRock1Triangles), fmt::format("rock1-{} instance placement", i));
    AABB     aabb      = computeBounds(result.meshes.back().positions);
    uint32_t instances = glm::linearRand(10u, 20u);
    for(uint32_t j = 0; j < instances; ++j)
    {
      float     scale       = glm::linearRand(0.3f, 0.4f);
      glm::vec3 translation = pickFlatPosition(aabb.size().x * scale * 0.1f);

      glm::mat4 model = glm::translate(glm::mat4(1.0f), translation);
      model *= randomRotation(0.08f);
      model = glm::scale(model, glm::vec3(scale));
      model = glm::translate(model, glm::vec3(0.0f, -aabb.min.y - aabb.size().y * 0.2f, 0.0f));
      result.instances.push_back({uint32_t(result.meshes.size() - 1), model});
    }
  }
#endif

// Medium rocks
#if 1
  for(uint32_t i = 0; i < numRock2s; ++i)
  {
    LOGI("Generating rock2 %d\n", i);
    progress.setSubtaskWorkName(fmt::format("rock2-{} geometry generation", i));
    result.meshes.push_back(generateMesh(allocator, m_generators.rock2, pool, queue, i, numRock2Triangles));
    progress.progressTo(workEstimate(numRock2Triangles), fmt::format("rock2-{} instance placement", i));
    AABB     aabb      = computeBounds(result.meshes.back().positions);
    uint32_t instances = glm::linearRand(50u, 200u);
    //static glm::vec2 clusterCenter = glm::linearRand(glm::vec2(-0.8f), glm::vec2(0.8f));
    for(uint32_t j = 0; j < instances; ++j)
    {
      float     scale       = glm::linearRand(0.03f, 0.2f);  // highly variable scale
      glm::vec3 translation = pickFlatPosition(aabb.size().x * scale * 0.1f);

      glm::mat4 model = glm::translate(glm::mat4(1.0f), translation);
      model *= randomRotation(0.5f);
      model = glm::scale(model, glm::vec3(scale));
      model = glm::translate(model, glm::vec3(0.0f, -aabb.min.y - aabb.size().y * 0.4f, 0.0f));
      result.instances.push_back({uint32_t(result.meshes.size() - 1), model});
    }
  }
#endif

// Many small rocks
#if 1
  for(uint32_t i = 0; i < numRock3s; ++i)
  {
    LOGI("Generating rock3 %d\n", i);
    progress.setSubtaskWorkName(fmt::format("rock3-{} geometry generation", i));
    result.meshes.push_back(generateMesh(allocator, m_generators.rock3, pool, queue, i, numRock3Triangles));
    progress.progressTo(workEstimate(numRock3Triangles), fmt::format("rock3-{} instance placement", i));
    AABB     aabb      = computeBounds(result.meshes.back().positions);
    uint32_t instances = glm::linearRand(1000u, 2000u);
    for(uint32_t j = 0; j < instances; ++j)
    {
      float     scale       = glm::linearRand(0.005f, 0.08f);  // highly variable scale
      glm::vec3 translation = pickFlatPosition(aabb.size().x * scale * 0.1f);

      glm::mat4 model = glm::translate(glm::mat4(1.0f), translation);
      model *= randomRotation(0.5f);
      model = glm::scale(model, glm::vec3(scale));
      model = glm::translate(model, glm::vec3(0.0f, -aabb.min.y - aabb.size().y * 0.4f, 0.0f));
      result.instances.push_back({uint32_t(result.meshes.size() - 1), model});
    }
  }
#endif

// Another cellular shaped rock
#if 1
  for(uint32_t i = 0; i < numRock4s; ++i)
  {
    LOGI("Generating rock4 %d\n", i);
    progress.setSubtaskWorkName(fmt::format("rock4-{} geometry generation", i));
    result.meshes.push_back(generateMesh(allocator, m_generators.rock4, pool, queue, i, numRock4Triangles));
    progress.progressTo(workEstimate(numRock4Triangles), fmt::format("rock4-{} instance placement", i));
    AABB     aabb      = computeBounds(result.meshes.back().positions);
    uint32_t instances = glm::linearRand(50u, 200u);
    for(uint32_t j = 0; j < instances; ++j)
    {
      float     scale       = glm::linearRand(0.04f, 0.07f);
      glm::vec3 translation = pickFlatPosition(aabb.size().x * scale * 0.1f);

      glm::mat4 model = glm::translate(glm::mat4(1.0f), translation);
      model *= randomRotation(0.5f);
      model = glm::scale(model, glm::vec3(scale));
      model = glm::translate(model, glm::vec3(0.0f, -aabb.min.y - aabb.size().y * 0.4f, 0.0f));
      result.instances.push_back({uint32_t(result.meshes.size() - 1), model});
    }
  }
#endif
  return result;
}
