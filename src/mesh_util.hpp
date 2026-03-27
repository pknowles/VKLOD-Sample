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

#include <glm/glm.hpp>

struct AABB
{
  glm::vec3 min, max;

  // Require an explicit make_empty()
  constexpr AABB() = delete;

  constexpr AABB(const glm::vec3& min_, const glm::vec3& max_) noexcept
      : min(min_)
      , max(max_)
  {
  }

  // Initialize with inverse bounds to allow subsequent unions with '+'
  static constexpr AABB make_empty() noexcept
  {
    return {glm::vec3(std::numeric_limits<float>::max()),
            glm::vec3(std::numeric_limits<float>::lowest())};
  }

  // Plus returns the union of bounding boxes.
  // [[nodiscard]] allows the compiler to warn if the return value is ignored,
  // which would be a bug. E.g. a + b; but should be a += b;
  [[nodiscard]] constexpr AABB operator+(const AABB& other) const noexcept
  {
    return {glm::min(min, other.min), glm::max(max, other.max)};
  }
  constexpr AABB& operator+=(const AABB& other) noexcept
  {
    return *this = *this + other;
  };

  [[nodiscard]] constexpr glm::vec3 size() const noexcept { return max - min; }
  [[nodiscard]] constexpr glm::vec3 center() const noexcept
  {
    return (min + max) * 0.5f;
  }
  [[nodiscard]] constexpr glm::vec3 positive_size() const noexcept
  {
    return glm::max(glm::vec3(0.0f), size());
  }
  [[nodiscard]] constexpr AABB positive() const noexcept
  {
    return {min, min + positive_size()};
  }
  [[nodiscard]] constexpr float half_area() const noexcept
  {
    auto s = size();
    return s.x * (s.y + s.z) + s.y * s.z;
  }
  [[nodiscard]] constexpr AABB intersect(const AABB& other) const noexcept
  {
    return AABB{glm::max(min, other.min), glm::min(max, other.max)}.positive();
  }
};

template <std::ranges::input_range Positions>
AABB computeBounds(Positions&& positions) noexcept
{
  AABB result = AABB::make_empty();
  for(const auto& position : positions)
    result = result + AABB{position, position};
  return result;
}
