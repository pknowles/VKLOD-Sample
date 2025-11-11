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

// Mostly forked from nvpro_core/imgui/imgui_helper.cpp to provide a font with a
// larger size to use in the HUD

#include <filesystem>
#include <imgui.h>
#include <imgui/imgui_icon.h>
#include <nvvkhl/Roboto-Regular.h>

namespace fs = std::filesystem;

class LargeFont
{
public:
  static ImFont* instance()
  {
    static LargeFont largeFont;
    return largeFont;
  }

private:
  LargeFont()
  {
    ImGuiIO& io = ImGui::GetIO();

    ImFontConfig font_config{};
    font_config.SizePixels           = 40.0f;
    font_config.FontDataOwnedByAtlas = false;

    // From nvpro_core/nvvkhl/application.cpp and nvpro_core/nvvkhl/Roboto-Regular.h
    m_largeFont = io.Fonts->AddFontFromMemoryTTF(const_cast<uint8_t*>(g_Roboto_Regular), sizeof(g_Roboto_Regular),
                                                 font_config.SizePixels, &font_config);
  }

  operator ImFont*() const { return m_largeFont; }

private:
  ImFont* m_largeFont = nullptr;
};

// Hack: provide declaration from imgui_icon.cpp
const char* getOpenIconicFontCompressedBase85TTF();

class LargeIconFont
{
public:
  static ImFont* instance()
  {
    static LargeIconFont largeIconFont;
    return largeIconFont;
  }

private:
  LargeIconFont()
  {
    ImGuiIO& io = ImGui::GetIO();

    ImFontConfig font_config{};
    font_config.SizePixels           = 20.0f;
    font_config.FontDataOwnedByAtlas = false;

    const char*           glyphsData = getOpenIconicFontCompressedBase85TTF();
    static uint16_t const range[]    = {0xE000, 0xE0DF, 0};

    m_largeIconFont = io.Fonts->AddFontFromMemoryCompressedBase85TTF(glyphsData, font_config.SizePixels, &font_config,
                                                                     (const ImWchar*)range);
  }

  operator ImFont*() const { return m_largeIconFont; }

private:
  ImFont* m_largeIconFont = nullptr;
};
