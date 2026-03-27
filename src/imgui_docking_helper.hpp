/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Pyarelal Knowles
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <imgui.h>
#include <imgui_internal.h>  // For DockBuilder API

// Creates a fullscreen dockspace over the main viewport
// Call this once per frame after ImGui::NewFrame()
// Uses the ID from ImGui::GetID("MainDockSpace")
inline void createDockspace()
{
  ImGui::DockSpaceOverViewport(ImGui::GetID("MainDockSpace"), ImGui::GetMainViewport(),
                               ImGuiDockNodeFlags_NoDockingInCentralNode);
}

// Sets up the initial docking layout
// Call this ONCE on first run (e.g., when no .ini file exists)
// The caller is responsible for calling this only once
inline void setupInitialDockingLayout()
{
  ImGuiID dockspaceId = ImGui::GetID("MainDockSpace");
  ImGui::DockBuilderRemoveNode(dockspaceId);
  ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
  ImGui::DockBuilderSetNodeSize(dockspaceId, ImGui::GetMainViewport()->Size);

  // Split: left panel for Settings/Profiler/Camera Paths, right for Viewport
  ImGuiID viewportId  = dockspaceId;
  ImGuiID leftPanelId = ImGui::DockBuilderSplitNode(viewportId, ImGuiDir_Left,
                                                    0.2F, nullptr, &viewportId);

  // Mark viewport node as central so it expands on resize
  if(ImGuiDockNode* node = ImGui::DockBuilderGetNode(viewportId))
  {
    node->SetLocalFlags(node->LocalFlags | ImGuiDockNodeFlags_CentralNode);
  }

  ImGui::DockBuilderDockWindow("Settings", leftPanelId);

  // Split the left panel vertically for Profiler and Camera Paths
  ImGuiID profilerId = ImGui::DockBuilderSplitNode(leftPanelId, ImGuiDir_Down,
                                                   0.3F, nullptr, &leftPanelId);
  ImGui::DockBuilderDockWindow("Profiler", profilerId);

  ImGuiID cameraPathsId = ImGui::DockBuilderSplitNode(profilerId, ImGuiDir_Down,
                                                      0.3F, nullptr, &profilerId);
  ImGui::DockBuilderDockWindow("Camera Paths", cameraPathsId);

  ImGui::DockBuilderDockWindow("##Viewport", viewportId);
  ImGui::DockBuilderFinish(dockspaceId);
}

// Sets up ImGui dark style to match the original nvpro_core appearance
inline void setupImGuiStyle()
{
  ImGui::StyleColorsDark();

  ImGuiStyle& style                  = ImGui::GetStyle();
  style.WindowRounding               = 0.0f;
  style.WindowBorderSize             = 0.0f;
  style.ColorButtonPosition          = ImGuiDir_Right;
  style.FrameRounding                = 2.0f;
  style.FrameBorderSize              = 1.0f;
  style.GrabRounding                 = 4.0f;
  style.IndentSpacing                = 12.0f;
  style.Colors[ImGuiCol_WindowBg]    = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
  style.Colors[ImGuiCol_MenuBarBg]   = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
  style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
  style.Colors[ImGuiCol_PopupBg]     = ImVec4(0.135f, 0.135f, 0.135f, 1.0f);
  style.Colors[ImGuiCol_Border]      = ImVec4(0.4f, 0.4f, 0.4f, 0.5f);
  style.Colors[ImGuiCol_FrameBg]     = ImVec4(0.05f, 0.05f, 0.05f, 0.5f);

  // Normal state colors
  ImVec4 normal_color                   = ImVec4(0.465f, 0.465f, 0.525f, 1.0f);
  style.Colors[ImGuiCol_Header]         = normal_color;
  style.Colors[ImGuiCol_SliderGrab]     = normal_color;
  style.Colors[ImGuiCol_Button]         = normal_color;
  style.Colors[ImGuiCol_CheckMark]      = normal_color;
  style.Colors[ImGuiCol_ResizeGrip]     = normal_color;
  style.Colors[ImGuiCol_TextSelectedBg] = normal_color;
  style.Colors[ImGuiCol_Separator]      = normal_color;
  style.Colors[ImGuiCol_FrameBgActive]  = normal_color;

  // Active state colors
  ImVec4 active_color                 = ImVec4(0.365f, 0.365f, 0.425f, 1.0f);
  style.Colors[ImGuiCol_HeaderActive] = active_color;
  style.Colors[ImGuiCol_SliderGrabActive] = active_color;
  style.Colors[ImGuiCol_ButtonActive]     = active_color;
  style.Colors[ImGuiCol_ResizeGripActive] = active_color;
  style.Colors[ImGuiCol_SeparatorActive]  = active_color;

  // Hovered state colors
  ImVec4 hovered_color                  = ImVec4(0.565f, 0.565f, 0.625f, 1.0f);
  style.Colors[ImGuiCol_HeaderHovered]  = hovered_color;
  style.Colors[ImGuiCol_ButtonHovered]  = hovered_color;
  style.Colors[ImGuiCol_FrameBgHovered] = hovered_color;
  style.Colors[ImGuiCol_ResizeGripHovered] = hovered_color;
  style.Colors[ImGuiCol_SeparatorHovered]  = hovered_color;

  // Other colors
  style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.465f, 0.465f, 0.465f, 1.0f);
  style.Colors[ImGuiCol_TitleBg]       = ImVec4(0.125f, 0.125f, 0.125f, 1.0f);
  style.Colors[ImGuiCol_Tab]           = ImVec4(0.05f, 0.05f, 0.05f, 0.5f);
  style.Colors[ImGuiCol_TabHovered]    = ImVec4(0.465f, 0.495f, 0.525f, 1.0f);
  style.Colors[ImGuiCol_TabActive]     = ImVec4(0.282f, 0.290f, 0.302f, 1.0f);
  style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.465f, 0.465f, 0.465f, 0.350f);
}
