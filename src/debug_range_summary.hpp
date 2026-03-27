/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// This file contains rangeSummary() and rangeSummaryVk(), which simply print
// the start and end of a C++ range or GPU buffer. Really useful for debugging,
// provided you can submit and finish command buffers between stages and then
// use this to inspect the data.

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <ostream>
#include <sample_vulkan_objects.hpp>
#include <span>
#include <vko/allocator.hpp>
#include <vko/command_recording.hpp>
#include <vko/staging_memory.hpp>
#include <vulkan/vulkan_core.h>

namespace numerical_chars {
inline std::ostream& operator<<(std::ostream& os, char c)
{
  return std::is_signed<char>::value ? os << static_cast<int>(c) :
                                       os << static_cast<unsigned int>(c);
}

inline std::ostream& operator<<(std::ostream& os, signed char c)
{
  return os << static_cast<int>(c);
}

inline std::ostream& operator<<(std::ostream& os, unsigned char c)
{
  return os << static_cast<unsigned int>(c);
}
}  // namespace numerical_chars

template <char separator = ',', std::integral T>
std::string formatThousands(T number)
{
  std::string countStr = std::to_string(number);
  std::string result;
  result.reserve(countStr.length() + countStr.length() / 3);
  for(size_t i = 0; i < countStr.length(); ++i)
  {
    if(i > 0 && (countStr.length() - i) % 3 == 0)
      result += separator;
    result += countStr[i];
  }
  return result;
}

// clang-format off
namespace glm {
inline std::ostream & operator<<(std::ostream &os, const glm::vec2& v)    { return os << "{" << v.x << ", " << v.y << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::vec3& v)    { return os << "{" << v.x << ", " << v.y << ", " << v.z << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::vec4& v)    { return os << "{" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::uvec2& v)   { return os << "{" << v.x << ", " << v.y << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::uvec3& v)   { return os << "{" << v.x << ", " << v.y << ", " << v.z << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::uvec4& v)   { return os << "{" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::u16vec2& v) { return os << "{" << v.x << ", " << v.y << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::u16vec3& v) { return os << "{" << v.x << ", " << v.y << ", " << v.z << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::u16vec4& v) { return os << "{" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::u8vec2& v)  { using numerical_chars::operator<<; return os << "{" << v.x << ", " << v.y << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::u8vec3& v)  { using numerical_chars::operator<<; return os << "{" << v.x << ", " << v.y << ", " << v.z << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::u8vec4& v)  { using numerical_chars::operator<<; return os << "{" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "}"; }
}
// clang-format on

template <glm::length_t M, glm::length_t N, typename T, glm::qualifier Q>
inline std::ostream& operator<<(std::ostream& os, const glm::mat<M, N, T, Q>& v)
{
  return os << glm::to_string(v);
}

// A class to wrap a std::ostream and insert a prefix at the start of every newline
class PrefixedLines : public std::streambuf
{
public:
  PrefixedLines(std::streambuf* output, std::string_view prefix)
      : m_output(output)
      , m_prefix(prefix)
  {
  }

  virtual int sync() override { return m_output->pubsync(); }

  // slow individual virtual calls per character per ostream in the chain, but
  // short code :)
  virtual int overflow(int c) override
  {
    assert(c == traits_type::to_char_type(c));
    if(c == traits_type::eof())
    {
      sync();
      return traits_type::eof();
    }
    if(std::exchange(m_newline, c == '\n'))
      m_output->sputn(m_prefix.data(), std::streamsize(m_prefix.size()));
    return m_output->sputc(traits_type::to_char_type(c));
  }

private:
  std::streambuf*  m_output;
  std::string_view m_prefix;
  bool             m_newline = false;
};

// Print the start and end of a std range
template <class T>
std::ostream& rangeSummary(std::ostream& os, const T& range, size_t maxItems = 6, bool multiline = false)
{
  using numerical_chars::operator<<;
  constexpr bool isString =
      std::is_same_v<std::decay_t<decltype(*std::begin(range))>, char>;
  const char* separator = multiline ? ",\n  " : (isString ? "" : ", ");

  // Indent any newlines written during the range
  PrefixedLines indent(os.rdbuf(), "  ");
  std::ostream  ios(multiline ? &indent : os.rdbuf());

  // Specialize based on whether the size of the range can be computed
  if constexpr(std::is_base_of_v<std::random_access_iterator_tag,
                                 typename std::iterator_traits<decltype(std::begin(range))>::iterator_category>)
  {
    os << "{";
    size_t size = size_t(std::distance(std::begin(range), std::end(range)));
    if(multiline && size > 1)
    {
      ios << "\n";
    }
    if(size <= maxItems)
    {
      auto it = std::begin(range);
      if(it != std::end(range))
      {
        ios << *it;
        ++it;
      }
      for(; it != std::end(range); ++it)
      {
        ios << separator << *it;
      }
    }
    else
    {
      auto it = std::begin(range);
      for(size_t i = 0; i < maxItems / 2; ++i)
      {
        ios << *it++ << separator;
      }
      it += std::distance(it, std::end(range)) - std::intptr_t(maxItems) / 2;
      ios << "...";
      while(it != std::end(range))
      {
        ios << separator << *it++;
      }
    }
    os << "}[" << size << "]";
  }
  else
  {
    os << "{";
    // Unknown size
    auto it = std::begin(range);
    if(multiline && it != std::end(range))
    {
      ios << "\n  ";
    }
    if(it != std::end(range))
    {
      ios << *it;
      ++it;
    }
    for(size_t i = 1; i < maxItems && it != std::end(range); ++i, ++it)
    {
      ios << separator << *it;
    }
    if(it != std::end(range))
    {
      ios << "...";
    }
    os << "}";
  }
  return os;
}

// Debugging utility class to allow dumping objects in GPU memory recursively
// with rangeSummaryVk.
// Usage:
// {
//   BufferDownloader downloader(context.device, queueFamilyIndex, context.allocator->getStaging());
//   rangeSummaryVk<glm::uvec3>(std::cerr << "Triangles: ", triangleIndicesVkBuffer, triangleCount) << "\n";
// }
// Requires VK_NV_copy_memory_indirect to be enabled
class BufferDownloader
{
public:
  BufferDownloader(const vko::Device& device, uint32_t queueFamilyIndex, vko::vma::Allocator& allocator)
  {
    assert(s_device == nullptr);
    s_device           = &device;
    s_queueFamilyIndex = queueFamilyIndex;
    s_allocator        = &allocator;
  }
  ~BufferDownloader()
  {
    s_device           = VK_NULL_HANDLE;
    s_queueFamilyIndex = 0;
    s_allocator        = nullptr;
  }
  // Download from BufferSpan<T>
  template <typename T>
  static std::vector<std::remove_const_t<T>> download(vko::BufferSpan<T> src)
  {
    if(src.empty())
      return {};
    assert(s_device != VK_NULL_HANDLE);

    vko::TimelineQueue timelineQueue(*s_device, s_queueFamilyIndex, 0);
    vko::vma::RecyclingStagingPool stagingPool(*s_device, *s_allocator, 3, 5, 1 << 24,
                                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                                   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    vko::StagingStream stream(timelineQueue, std::move(stagingPool));

    vko::cmdMemoryBarrier(*s_device, stream.commandBuffer(),
                          {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT},
                          {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT});

    auto future = vko::download(stream, *s_device, src);
    stream.submit();
    return future.get(*s_device);
  }

  // Download from DeviceSpan<T>
  template <typename T>
  static std::vector<T> download(vko::DeviceSpan<const T> src)
  {
    if(src.empty())
      return {};
    assert(s_device != VK_NULL_HANDLE);

    vko::TimelineQueue timelineQueue(*s_device, s_queueFamilyIndex, 0);
    vko::vma::RecyclingStagingPool stagingPool(*s_device, *s_allocator, 3, 5, 1 << 24,
                                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                                   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    vko::StagingStream stream(timelineQueue, std::move(stagingPool));

    vko::cmdMemoryBarrier(*s_device, stream.commandBuffer(),
                          {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT},
                          {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT});

    auto future = vko::download(stream, *s_device, src);
    stream.submit();
    return future.get(*s_device);
  }

private:
  static inline thread_local const vko::Device*   s_device           = nullptr;
  static inline thread_local uint32_t             s_queueFamilyIndex = 0;
  static inline thread_local vko::vma::Allocator* s_allocator        = nullptr;
};

template <class T>
std::ostream& rangeSummaryVk(std::ostream& os,
                             VkBuffer      buffer,
                             size_t        elementCount,
                             size_t        maxItems  = 6,
                             bool          multiline = false)
{
  vko::BufferSpan<const T> span(vko::BufferAddress<const T>(buffer, 0), elementCount);
  auto hostArray = BufferDownloader::download(span);
  return rangeSummary(os, hostArray, maxItems, multiline);
}

template <class T>
std::ostream& rangeSummaryVk(std::ostream&           os,
                             const vkobj::Buffer<T>& array,
                             size_t                  maxItems  = 6,
                             bool                    multiline = false)
{
  auto hostArray          = BufferDownloader::download(vko::BufferSpan(array));
  VkDeviceAddress address = array.address();
  return rangeSummary(os << reinterpret_cast<void*>(address), hostArray, maxItems, multiline);
}

template <class T>
std::ostream& rangeSummaryVk(std::ostream&           os,
                             vkobj::DeviceAddress<T> address,
                             size_t                  elementCount,
                             size_t                  maxItems  = 6,
                             bool                    multiline = false)
{
  vko::DeviceSpan<const T> span(vko::DeviceAddress<const T>(address.address), elementCount);
  auto hostArray = BufferDownloader::download(span);
  return rangeSummary(os << reinterpret_cast<void*>(address.address), hostArray,
                      maxItems, multiline);
}

template <class T>
std::ostream& rangeSummaryVk(std::ostream&   os,
                             VkDeviceAddress address,
                             size_t          elementCount,
                             size_t          maxItems  = 6,
                             bool            multiline = false)
{
  return rangeSummaryVk(os << reinterpret_cast<void*>(address),
                        vkobj::DeviceAddress<T>(address), elementCount, maxItems, multiline);
}
