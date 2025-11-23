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

// Mostly forked from nvpro_core/imgui/imgui_helper.cpp to provide fonts with
// different sizes for use in the HUD and UI

#include <imgui.h>
#include <nvpro_core_legacy/imgui/imgui_icon.h>
#include <nvvkhl/Roboto-Regular.h>

// Provide declaration from imgui_icon.cpp
const char* getOpenIconicFontCompressedBase85TTF();

// Create a large Roboto font (40px) for HUD overlay text
inline ImFont* createLargeFont(ImGuiIO& io)
{
  ImFontConfig font_config{};
  font_config.SizePixels           = 40.0f;
  font_config.FontDataOwnedByAtlas = false;

  return io.Fonts->AddFontFromMemoryTTF(const_cast<uint8_t*>(g_Roboto_Regular),
                                        sizeof(g_Roboto_Regular),
                                        font_config.SizePixels, &font_config);
}

// Create a large icon font (20px) for icon buttons
inline ImFont* createLargeIconFont(ImGuiIO& io)
{
  ImFontConfig font_config{};
  font_config.SizePixels           = 20.0f;
  font_config.FontDataOwnedByAtlas = false;

  const char*           glyphsData = getOpenIconicFontCompressedBase85TTF();
  static uint16_t const range[]    = {0xE000, 0xE0DF, 0};

  return io.Fonts->AddFontFromMemoryCompressedBase85TTF(
      glyphsData, font_config.SizePixels, &font_config, (const ImWchar*)range);
}

// Setup default font (Roboto) with icons merged in
inline ImFont* createDefaultFontWithIcons(ImGuiIO& io, float fontSize = 14.0f)
{
  // Load Roboto as the default font
  ImFontConfig font_config{};
  font_config.FontDataOwnedByAtlas = false;
  ImFont* defaultFont =
      io.Fonts->AddFontFromMemoryTTF(const_cast<uint8_t*>(g_Roboto_Regular),
                                     sizeof(g_Roboto_Regular), fontSize, &font_config);

  // Merge icon font into the default font
  ImFontConfig icon_config{};
  icon_config.MergeMode              = true;  // Merge into previous font
  icon_config.FontDataOwnedByAtlas   = false;
  static uint16_t const icon_range[] = {0xE000, 0xE0DF, 0};
  io.Fonts->AddFontFromMemoryCompressedBase85TTF(getOpenIconicFontCompressedBase85TTF(),
                                                 fontSize, &icon_config,
                                                 (const ImWchar*)icon_range);

  return defaultFont;
}

// Non-owning pointers to extra fonts (owned by ImGui font atlas)
struct ExtraFonts
{
  ImFont* large     = nullptr;  // 40px Roboto for HUD overlay
  ImFont* largeIcon = nullptr;  // 20px OpenIconic for icon buttons
};

// Initialize all extra fonts - call during ImGui font setup, before atlas is built
inline ExtraFonts initExtraFonts(ImGuiIO& io)
{
  return ExtraFonts{
      .large     = createLargeFont(io),
      .largeIcon = createLargeIconFont(io),
  };
}

// Helper function for creating styled toggle icon buttons
// Returns true if the button was pressed
inline bool iconButton(ImFont*       iconFont,
                       const char*   icon,
                       bool          isActive,
                       const char*   tooltipOn,
                       const char*   tooltipOff,
                       const ImVec4& activeColor =
                           ImVec4(118.0f / 255.0f, 185.0f / 255.0f, 0.0f / 255.0f, 1.0f))
{
  ImGui::PushFont(iconFont);
  if(isActive)
  {
    ImGui::PushStyleColor(ImGuiCol_Button, activeColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          ImVec4(activeColor.x * 1.2f, activeColor.y * 1.2f,
                                 activeColor.z * 1.2f, activeColor.w));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                          ImVec4(activeColor.x * 0.8f, activeColor.y * 0.8f,
                                 activeColor.z * 0.8f, activeColor.w));
  }
  bool pressed = ImGui::Button(icon);
  if(isActive)
  {
    ImGui::PopStyleColor(3);
  }
  ImGui::PopFont();
  if(ImGui::IsItemHovered())
  {
    ImGui::SetTooltip("%s", isActive ? tooltipOn : tooltipOff);
  }
  ImGui::SameLine();
  return pressed;
}
