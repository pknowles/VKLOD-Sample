/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Pyarelal Knowles
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <vko/command_recording.hpp>
#include <vko/handles.hpp>
#include <vko/query_pool.hpp>
#include <vko/timeline_queue.hpp>
#include <vulkan/vulkan_core.h>

// Simple ring buffer for storing recent samples
template <typename T, size_t Capacity>
class RingBuffer
{
public:
  void push(T value)
  {
    m_data[m_pivot] = value;
    m_pivot         = (m_pivot + 1) % Capacity;
    if(m_count < Capacity)
      ++m_count;
  }

  [[nodiscard]] size_t   size() const { return m_count; }
  [[nodiscard]] size_t   capacity() const { return Capacity; }
  [[nodiscard]] bool     empty() const { return m_count == 0; }
  [[nodiscard]] size_t   pivot() const { return m_pivot; }
  [[nodiscard]] const T* data() const { return m_data.data(); }

  [[nodiscard]] const T& operator[](size_t i) const
  {
    assert(i < m_count);
    size_t physicalIndex = (m_pivot + Capacity - m_count + i) % Capacity;
    return m_data[physicalIndex];
  }

  [[nodiscard]] const T& at(size_t i) const
  {
    if(i >= m_count)
      throw std::out_of_range("RingBuffer::at: index out of range");
    return (*this)[i];
  }

  [[nodiscard]] const T& back() const
  {
    assert(!empty());
    return (*this)[m_count - 1];
  }

  [[nodiscard]] auto begin() const { return Iterator{this, 0}; }
  [[nodiscard]] auto end() const { return Iterator{this, m_count}; }

private:
  std::array<T, Capacity> m_data{};
  size_t                  m_pivot = 0;
  size_t                  m_count = 0;

  struct Iterator
  {
    const RingBuffer* buffer;
    size_t            index;

    using value_type        = T;
    using difference_type   = std::ptrdiff_t;
    using reference         = const T&;
    using pointer           = const T*;
    using iterator_category = std::random_access_iterator_tag;

    [[nodiscard]] reference operator*() const { return (*buffer)[index]; }
    [[nodiscard]] pointer   operator->() const { return &(*buffer)[index]; }

    Iterator& operator++()
    {
      ++index;
      return *this;
    }
    Iterator  operator++(int) { return {buffer, index++}; }
    Iterator& operator--()
    {
      --index;
      return *this;
    }
    Iterator  operator--(int) { return {buffer, index--}; }
    Iterator& operator+=(difference_type n)
    {
      index += n;
      return *this;
    }
    Iterator& operator-=(difference_type n)
    {
      index -= n;
      return *this;
    }

    [[nodiscard]] Iterator operator+(difference_type n) const
    {
      return {buffer, index + n};
    }
    [[nodiscard]] Iterator operator-(difference_type n) const
    {
      return {buffer, index - n};
    }
    [[nodiscard]] difference_type operator-(const Iterator& other) const
    {
      return static_cast<difference_type>(index)
             - static_cast<difference_type>(other.index);
    }
    [[nodiscard]] reference operator[](difference_type n) const
    {
      return (*buffer)[index + n];
    }

    [[nodiscard]] friend Iterator operator+(difference_type n, const Iterator& it)
    {
      return it + n;
    }

    [[nodiscard]] bool operator==(const Iterator& other) const
    {
      return index == other.index;
    }
    [[nodiscard]] bool operator!=(const Iterator& other) const
    {
      return index != other.index;
    }
    [[nodiscard]] bool operator<(const Iterator& other) const
    {
      return index < other.index;
    }
    [[nodiscard]] bool operator<=(const Iterator& other) const
    {
      return index <= other.index;
    }
    [[nodiscard]] bool operator>(const Iterator& other) const
    {
      return index > other.index;
    }
    [[nodiscard]] bool operator>=(const Iterator& other) const
    {
      return index >= other.index;
    }
  };
};

// Statistics computed from a range of duration samples
struct DurationStatsView
{
  uint64_t currentNs   = 0;
  double   avgNs       = 0.0;
  uint64_t minNs       = 0;
  uint64_t maxNs       = 0;
  size_t   sampleCount = 0;

  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>, uint64_t>
  [[nodiscard]] static DurationStatsView compute(R&& samples, uint64_t currentNs)
  {
    DurationStatsView stats{};
    stats.currentNs = currentNs;
    stats.minNs     = std::numeric_limits<uint64_t>::max();

    size_t count = 0;
    for(uint64_t sample : samples)
    {
      stats.avgNs += (static_cast<double>(sample) - stats.avgNs)
                     / static_cast<double>(count + 1);
      stats.minNs = std::min(stats.minNs, sample);
      stats.maxNs = std::max(stats.maxNs, sample);
      ++count;
    }
    stats.sampleCount = count;

    if(count == 0)
      stats.minNs = 0;

    return stats;
  }
};

// Holds duration samples with lazy stats computation
class DurationSamples
{
public:
  static constexpr size_t kHistorySize = 128;

  void addSample(uint64_t ns)
  {
    m_samples.push(ns);
    m_currentNs = ns;
  }

  [[nodiscard]] DurationStatsView stats() const
  {
    return DurationStatsView::compute(m_samples, m_currentNs);
  }
  [[nodiscard]] uint64_t currentNs() const { return m_currentNs; }
  [[nodiscard]] const RingBuffer<uint64_t, kHistorySize>& samples() const
  {
    return m_samples;
  }

private:
  RingBuffer<uint64_t, kHistorySize> m_samples;
  uint64_t                           m_currentNs = 0;
};

// Main profiler: manages queries and duration samples
class SampleProfiler
{
public:
  using TimestampQuery = vko::SharedQuery<uint64_t, VK_QUERY_TYPE_TIMESTAMP>;
  using GpuMeasurement = std::pair<TimestampQuery, TimestampQuery>;

  [[nodiscard]] vko::TimestampQueryStream<>& queryStream() {
    std::lock_guard lock(m_mutex);
    return m_threadQueryStreams[std::this_thread::get_id()];
  }

  // Protects everything - timeline ordering, parents
  mutable std::mutex m_mutex;

  struct TaggedSamples : DurationSamples
  {
    std::vector<GpuMeasurement> pendingGpuSamples;
    bool isGPU = false;
    bool inactive = false;

    void collectGpuResults(const vko::Device& device)
    {
      auto it = pendingGpuSamples.begin();
      while(it != pendingGpuSamples.end())
      {
        if(it->first.semaphore().ready(device))
        {
          uint64_t begin = it->first.get(device);
          uint64_t end   = it->second.get(device);
          DurationSamples::addSample(end - begin);
          it = pendingGpuSamples.erase(it);
        }
        else
        {
          ++it;
        }
      }
    }

    void addSample(uint64_t ns)
    {
      assert(!isGPU); // duplicate name for both CPU and GPU sample
      DurationSamples::addSample(ns);
    }

    void addSample(GpuMeasurement&& sample)
    {
      isGPU = true;
      pendingGpuSamples.emplace_back(std::move(sample));
    }
  };

  struct Node
  {
    Node(const std::string& name) : name(name) {}

    std::string       name;
    TaggedSamples     samples;
    std::vector<Node> children;

    template <typename Fn>
    void visit(this auto&& self, Fn&& fn)
    {
      if constexpr(std::same_as<std::invoke_result_t<Fn, Node&>, bool>)
      {
        if(!fn(self))
          return;
      }
      else
        fn(self);
      for(auto& child : self.children)
        child.visit(fn);
    }
  };

  struct NodeSubrange
  {
    NodeSubrange(std::vector<Node>& nodes) : nodes(nodes), m_nextToSample(0) {}
    std::reference_wrapper<std::vector<Node>> nodes;
    std::vector<Node>::iterator begin() const { return nodes.get().begin() + m_nextToSample; }
    std::vector<Node>::iterator end() const { return nodes.get().end(); }
    Node& makeFrontOrInsert(const std::string& name) {
      auto it = std::find_if(begin(), end(), [&](const Node& node){ return node.name == name; });
      if(it == end())
        it = nodes.get().insert(begin(), Node{name});
      else if(it != begin())
      {
        // Move the existing node to the front of the range; inactive nodes are
        // left at the end.
        // TODO: would be nice to preserve the order of inactive nodes
        std::rotate(begin(), it, it + 1);
        it = begin();
      }
      return *it;
    }
    bool empty() const { return begin() == end(); }
    void advance()
    {
      assert(!empty());
      m_nextToSample++;
    }
    size_t m_nextToSample = 0;
  };

  std::map<std::thread::id, std::vector<Node>> threadSamples;
  std::map<std::thread::id, vko::TimestampQueryStream<>> m_threadQueryStreams;

  // Display metadata
  std::map<std::thread::id, std::vector<std::string>> durationOrder;
  std::map<std::string, std::string>                  durationParents;

  // Collect GPU queries that have completed (non-blocking)
  void collectGpuResults(const vko::Device& device)
  {
    std::lock_guard lock(m_mutex);
    for(auto& root : threadSamples[std::this_thread::get_id()])
    {
      root.visit([&](Node& node){
        node.samples.collectGpuResults(device);
      });
    }
  }

  void endQueryBatch(vko::SemaphoreValue semaphore)
  {
    queryStream().endBatch(std::move(semaphore));
  }

  [[nodiscard]] size_t maxPendingGpuSamples() const
  {
    std::lock_guard lock(m_mutex);
    size_t maxPending = 0;
    auto it = threadSamples.find(std::this_thread::get_id());
    if(it == threadSamples.end())
      return maxPending;
    for(const auto& root : it->second)
    {
      root.visit([&](const Node& node){
        maxPending = std::max(maxPending, node.samples.pendingGpuSamples.size());
      });
    }
    return maxPending;
  }

  [[nodiscard]] std::optional<DurationStatsView> stats(const std::string& name) const
  {
    std::lock_guard      lock(m_mutex);
    std::optional<DurationStatsView> result;
    auto it = threadSamples.find(std::this_thread::get_id());
    if(it == threadSamples.end())
      return result;
    for(const auto& root : it->second)
    {
      root.visit([&](const Node& node) {
        if(node.name == name)
          result = node.samples.stats();
      });
    }
    return result;
  }

  void beginSample(const std::string& name)
  {
    std::lock_guard lock(m_mutex);

    // Initialize the stack with the roots the first time for each thread and
    // again any time a duplicate name is found. This handles the frame loop.
    auto& s = stack();
    if(s.empty() || (s.size() == 1 && std::ranges::any_of(s.front().nodes.get(), [&](const Node& node){ return node.name == name; })))
    {
      s.clear();
      s.emplace_back(threadSamples[std::this_thread::get_id()]);
    }
    
    // Search through the remaining items on the top of the stack to find
    // 'name'. If it is not found, we should insert a new node.
    // TODO: add some node removal/cleanup
    Node& node = s.back().makeFrontOrInsert(name);
    node.samples.inactive = false;
    stack().emplace_back(node.children);
  }

  template <class SampleType>
  void endSample(SampleType&& sample)
  {
    std::lock_guard lock(m_mutex);
    auto&           s = stack();
    if(!s.empty())
    {
      for(Node& node : s.back())
        node.samples.inactive = true;
      s.pop_back();
    }
    if(s.empty() || s.back().empty())
      throw std::runtime_error("Mismatched begin/end");
    s.back().begin()->samples.addSample(std::move(sample));
    s.back().advance();
  }

  static std::vector<NodeSubrange>& stack()
  {
    static thread_local std::vector<NodeSubrange> s_stack;
    return s_stack;
  }
};

// RAII helper for scoped GPU timing
template <class CommandBufferType>
class ScopedGpuTimer
{
public:
  ScopedGpuTimer(const vko::Device& device,
                 SampleProfiler&    profiler,
                 CommandBufferType& cmd,
                 std::string        name)
      : m_profiler(profiler)
      , m_name(std::move(name))
      , m_cmd(cmd)
      , m_device(device)
      , m_queryBegin(vko::cmdWriteTimestamp(device, m_cmd.get(), profiler.queryStream(), VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT))
  {
    m_profiler.beginSample(m_name);
  }

  ~ScopedGpuTimer()
  {
    if(m_name.empty())
      return;

    auto queryEnd = vko::cmdWriteTimestamp(m_device, m_cmd.get(), m_profiler.queryStream(),
                                           VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
    m_profiler.endSample(SampleProfiler::GpuMeasurement{std::move(m_queryBegin), std::move(queryEnd)});
  }

  ScopedGpuTimer(const ScopedGpuTimer&)            = delete;
  ScopedGpuTimer& operator=(const ScopedGpuTimer&) = delete;

private:

  SampleProfiler&                           m_profiler;
  std::string                               m_name;
  std::reference_wrapper<CommandBufferType> m_cmd;
  const vko::Device&                        m_device;
  SampleProfiler::TimestampQuery            m_queryBegin;
};

// RAII helper for scoped CPU timing
class ScopedCpuTimer
{
public:
  ScopedCpuTimer(SampleProfiler& profiler, std::string name)
      : m_profiler(profiler)
      , m_name(std::move(name))
      , m_startTime(std::chrono::steady_clock::now())
  {
    m_profiler.beginSample(m_name);
  }

  ~ScopedCpuTimer()
  {
    auto endTime = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - m_startTime);
    m_profiler.endSample(uint64_t(duration.count()));
  }

  ScopedCpuTimer(const ScopedCpuTimer&)            = delete;
  ScopedCpuTimer& operator=(const ScopedCpuTimer&) = delete;

private:
  SampleProfiler&                       m_profiler;
  std::string                           m_name;
  std::chrono::steady_clock::time_point m_startTime;
};

// Render a single profiler's data
void renderProfiler(std::string_view label, SampleProfiler& profiler);

// Render consolidated view of multiple profilers (using reference_wrapper)
void renderProfiler(std::span<std::reference_wrapper<SampleProfiler>> profilers);

// Convenience: render multiple profilers from variadic arguments
template <typename... Profilers>
  requires(std::same_as<std::remove_cvref_t<Profilers>, SampleProfiler> && ...)
void renderProfiler(Profilers&... profilers)
{
  std::array<std::reference_wrapper<SampleProfiler>, sizeof...(Profilers)> refs = {
      std::ref(profilers)...};
  renderProfiler(std::span{refs});
}
