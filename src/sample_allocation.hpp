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

#include "vk_mem_alloc.h"
#include <assert.h>
#include <bit>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vulkan/vulkan_core.h>

// A remote allocator that suballocates a given range of memory. Uses a sorted
// binary tree free list keyed by size. Tables index start and end ranges so
// freed allocations can be joined with existing items in the free list.
// Disclaimer: I'm not an allocation expert. There are a lot of well understood
// solutions out there.
class PoolAllocator
{
  struct Range
  {
    VkDeviceAddress address;
    VkDeviceSize    size;
    VkDeviceAddress end() const { return address + size; }
    bool            operator<(const Range& other) const
    {
      return size == other.size ? address < other.address : size < other.size;
    }
  };

public:
  static constexpr VkDeviceSize ExternalFragmentationThreshold = 128ull << 10;

  PoolAllocator(VkDeviceAddress address, VkDeviceSize size)
      : m_base(address)
      , m_bytesTotal(size)
      , m_fragmentationExternal(size < ExternalFragmentationThreshold ? size : 0)
      , m_freeList({{m_base, m_bytesTotal}})
      , m_freeListStarts({{m_base, {m_base, m_bytesTotal}}})
      , m_freeListEnds({{m_base + m_bytesTotal, {m_base, m_bytesTotal}}})
  {
  }

  ~PoolAllocator()
  {
    if(m_freeList.size() != 1 || m_freeList.begin()->size != m_bytesTotal)
      std::println(stderr, "PoolAllocator destroyed before its allocations were freed");
  }

  VkDeviceAddress allocate(VkDeviceSize userSize, VkDeviceSize align)
  {
    assert(userSize >= align);  // alignment should always be at least the size
    std::lock_guard lk(m_mutex);

    VkDeviceSize allocSize = adjustSize(userSize);

    //std::println("Allocating {} bytes for request of size {}", allocSize, userSize);

    // Binary search to find free allocation
    auto it = m_freeList.lower_bound({0, allocSize});
    for(; it != m_freeList.end(); ++it)
    {
      assert(allocSize <= it->size);
      void* resultVP = reinterpret_cast<void*>(it->address);
      static_assert(sizeof(void*) == sizeof(VkDeviceAddress));  // TODO: x32 ... ?
      size_t remaining = it->size;
      if(std::align(align, allocSize, resultVP, remaining))
      {
        VkDeviceAddress result = reinterpret_cast<VkDeviceAddress>(resultVP);
        // Reinsert any remaining space in the free block
        removeFreeRange(*it);
        assert(remaining >= allocSize);
        if(remaining > allocSize)
        {
          Range remainingRange = {result + allocSize, remaining - allocSize};
          insertFreeRange(remainingRange);
        }

        // Track memory usage, excluding alignment and fragmentation overhead
        m_bytesAllocated += userSize;
        m_fragmentationInternal += allocSize - userSize;
        return result;
      }
      // else, does not fit due to alignment
    }
    throw std::bad_alloc();
  }

  void deallocate(VkDeviceAddress address, VkDeviceSize userSize) noexcept
  {
    std::lock_guard lk(m_mutex);
    m_bytesAllocated -= userSize;
    Range range = {address, adjustSize(userSize)};  // match the adjustment done in allocate()
    assert(m_fragmentationInternal >= range.size - userSize);
    m_fragmentationInternal -= range.size - userSize;
    if(auto joinEnd = m_freeListStarts.find(range.end());
       joinEnd != m_freeListStarts.end())
    {
      Range joinWith = joinEnd->second;
      removeFreeRange(joinWith);
      range = {range.address, range.size + joinWith.size};
      //std::println("Joined end of range at {}", range.address);
    }
    if(auto joinStart = m_freeListEnds.find(range.address);
       joinStart != m_freeListEnds.end())
    {
      Range joinWith = joinStart->second;
      removeFreeRange(joinWith);
      range = {joinWith.address, range.size + joinWith.size};
      //std::println("Joined start of range at {}", range.address);
    }
    try
    {
      insertFreeRange(range);
    }
    catch(const std::bad_alloc&)
    {
      std::terminate();  // fatal. mostly for coverity. could also just leak
    }
    //fprintf(stderr, "Free list size: %zu\n", m_freeList.size());
  }

  VkDeviceSize offsetOf(VkDeviceAddress address) const
  {
    return address - m_base;
  }
  VkDeviceSize bytesAllocated() const
  {
    std::lock_guard lk(m_mutex);
    return m_bytesAllocated;
  }
  VkDeviceSize size() const { return m_bytesTotal; }
  VkDeviceSize internalFragmentation() const
  {
    std::lock_guard lk(m_mutex);
    return m_fragmentationInternal;
  }
  VkDeviceSize externalFragmentation() const
  {
    std::lock_guard lk(m_mutex);
    return m_fragmentationExternal;
  }
  VkDeviceSize fragmentation() const
  {
    return internalFragmentation() + externalFragmentation();
  }

private:
  static VkDeviceSize adjustSize(VkDeviceSize size)
  {
    if(size < (4ull << 10))
    {
      // Grow size to next power of two to reduce fragmentation
      return std::bit_ceil(size);
    }

    // Align to 4KiB
    constexpr VkDeviceSize alignment = 4ull << 10;
    return (size + alignment - 1) & ~(alignment - 1);
  }

  void insertFreeRange(const Range& range)
  {
    [[maybe_unused]] auto [it, inserted] = m_freeList.insert(range);
    assert(inserted);  // should never collide
    m_freeListStarts.insert({range.address, range});
    m_freeListEnds.insert({range.end(), range});
    if(range.size < ExternalFragmentationThreshold)
      m_fragmentationExternal += range.size;
  }

  void removeFreeRange(const Range& range)
  {
    m_freeList.erase(range);
    m_freeListStarts.erase(range.address);
    m_freeListEnds.erase(range.end());
    if(range.size < ExternalFragmentationThreshold)
    {
      assert(m_fragmentationExternal >= range.size);
      m_fragmentationExternal -= range.size;
    }
  }

  VkDeviceAddress                            m_base                  = 0;
  VkDeviceSize                               m_bytesAllocated        = 0;
  VkDeviceSize                               m_bytesTotal            = 0;
  VkDeviceSize                               m_fragmentationInternal = 0;
  VkDeviceSize                               m_fragmentationExternal = 0;
  std::set<Range>                            m_freeList;
  std::unordered_map<VkDeviceAddress, Range> m_freeListStarts;
  std::unordered_map<VkDeviceAddress, Range> m_freeListEnds;
  mutable std::mutex                         m_mutex;
};

// Move-only destructing memory allocation from PoolAllocator
class PoolMemory
{
public:
  PoolMemory()                        = default;
  PoolMemory(const PoolMemory& other) = delete;
  PoolMemory(PoolMemory&& other) noexcept
      : m_allocator(other.m_allocator)
      , m_address(other.m_address)
      , m_size(other.m_size)
  {
    other.m_allocator = nullptr;
  }
  PoolMemory& operator=(const PoolMemory& other) = delete;
  PoolMemory& operator=(PoolMemory&& other) noexcept
  {
    destroy();
    m_allocator = nullptr;
    std::swap(m_allocator, other.m_allocator);
    m_address = other.m_address;
    m_size    = other.m_size;
    return *this;
  }
  PoolMemory(PoolAllocator& allocator, VkDeviceSize size, VkDeviceSize align)
      : m_allocator(&allocator)
      , m_address(allocator.allocate(std::max(size, align), align))
      , m_size(size)
  {
  }
  ~PoolMemory() { destroy(); }
  operator VkDeviceAddress() const { return m_address; }

private:
  void destroy()
  {
    if(m_allocator)
      m_allocator->deallocate(m_address, m_size);
  }

  PoolAllocator*  m_allocator = nullptr;
  VkDeviceAddress m_address   = 0xffffffffffffffffull;
  VkDeviceSize    m_size      = 0;
};
