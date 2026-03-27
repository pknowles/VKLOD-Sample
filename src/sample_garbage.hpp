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

#include <any>
#include <sample_vulkan_objects.hpp>

// Moves the given object into a std::shared_ptr. This is necessary to house
// move-only objects inside an std::any. A nicer implementation would use a
// move-only version of std::any since shared_ptr is expensive.
template <class T>
  requires(!std::is_lvalue_reference_v<T> && std::is_move_constructible_v<T>)
std::any moveAny(T&& obj)
{
  return std::make_shared<std::remove_reference_t<T>>(std::move(obj));
}

struct Garbage
{
  std::any object;
  vkobj::SemaphoreValue semaphoreState;  // don't free the object until this is signalled
};

inline void emptyUnusedGarbage(std::queue<Garbage>& garbage, const vko::Device& device)
{
  while(!garbage.empty() && garbage.front().semaphoreState.isSignaled(device))
  {
    garbage.pop();
  }
}

inline void waitAndEmptyGarbage(std::queue<Garbage>& garbage, const vko::Device& device)
{
  while(!garbage.empty())
  {
    garbage.front().semaphoreState.wait(device);
    garbage.pop();
  }
}
