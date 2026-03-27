/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Pyarelal Knowles
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sample_profiler.hpp>
#include <format>
#include <imgui.h>

static std::string formatDurationMs(uint64_t ns)
{
  return std::format("{:.3f}", static_cast<double>(ns) / 1'000'000.0);
}

static void renderNode(const SampleProfiler::Node& node)
{
  // Skip nodes that have not seen a sample the last frame
  // TODO: add some node removal/cleanup
  if(node.samples.inactive)
    return;

  const auto stats = node.samples.stats();

  ImGui::TableNextRow();

  // Column 0: Name (with tree structure)
  ImGui::TableNextColumn();
  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAllColumns;
  if(node.children.empty())
    flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

  bool open = ImGui::TreeNodeEx(node.name.c_str(), flags);

  // Column 1: Type
  ImGui::TableNextColumn();
  if(node.samples.isGPU)
  {
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "GPU");
  }
  else if(stats.sampleCount > 0)
  {
    ImGui::TextColored(ImVec4(0.4f, 0.6f, 1.0f, 1.0f), "CPU");
  }
  else
  {
    ImGui::TextDisabled("--");
  }

  // Column 2: Current
  ImGui::TableNextColumn();
  if(stats.sampleCount > 0)
  {
    if(node.samples.inactive)
    {
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%s",
                         formatDurationMs(stats.currentNs).c_str());
    }
    else
    {
      ImGui::Text("%s", formatDurationMs(stats.currentNs).c_str());
    }
  }
  else
  {
    ImGui::TextDisabled("--");
  }

  // Column 3: Avg
  ImGui::TableNextColumn();
  if(stats.sampleCount > 0)
  {
    ImGui::TextDisabled("%s", formatDurationMs(static_cast<uint64_t>(stats.avgNs)).c_str());
  }
  else
  {
    ImGui::TextDisabled("--");
  }

  // Column 4: Min/Max
  ImGui::TableNextColumn();
  if(stats.sampleCount > 0)
  {
    ImGui::TextDisabled("%s / %s", formatDurationMs(stats.minNs).c_str(),
                        formatDurationMs(stats.maxNs).c_str());
  }
  else
  {
    ImGui::TextDisabled("-- / --");
  }

  // Render children if open
  if(open && !node.children.empty())
  {
    for(const auto& child : node.children)
    {
      renderNode(child);
    }
    ImGui::TreePop();
  }
}

void renderProfiler(std::string_view label, SampleProfiler& profiler)
{
  std::string labelStr(label);
  if(ImGui::CollapsingHeader(labelStr.c_str()))
  {

    // Get max pending GPU samples
    size_t maxPending = profiler.maxPendingGpuSamples();
    if(maxPending > 0)
      ImGui::TextDisabled("Pending GPU queries: %zu", maxPending);

    ImGuiTableFlags tableFlags = ImGuiTableFlags_BordersInnerV
                                 | ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg;

    if(ImGui::BeginTable("profiler_table", 5, tableFlags))
    {
      ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoHide
                                          | ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 35.0f);
      ImGui::TableSetupColumn("Current (ms)", ImGuiTableColumnFlags_WidthFixed, 75.0f);
      ImGui::TableSetupColumn("Avg (ms)", ImGuiTableColumnFlags_WidthFixed, 75.0f);
      ImGui::TableSetupColumn("Min / Max (ms)", ImGuiTableColumnFlags_WidthFixed, 110.0f);
      ImGui::TableHeadersRow();

      // Render all root nodes for all threads
      std::lock_guard lock(profiler.m_mutex);
      bool hasAnySamples = false;
      for(const auto& [threadId, roots] : profiler.threadSamples)
      {
        for(const auto& root : roots)
        {
          renderNode(root);
          hasAnySamples = true;
        }
      }

      ImGui::EndTable();

      if(!hasAnySamples)
        ImGui::TextDisabled("No samples recorded yet");
    }
  }
}

void renderProfiler(std::span<std::reference_wrapper<SampleProfiler>> profilers)
{
  if(ImGui::Begin("Profiler"))
  {
    for(size_t i = 0; i < profilers.size(); ++i)
      renderProfiler("Profiler " + std::to_string(i), profilers[i].get());
  }
  ImGui::End();
}
