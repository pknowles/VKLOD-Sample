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
#include <fstream>
#include <glm/gtx/euler_angles.hpp>
#include <misc/cpp/imgui_stdlib.h>
#include <numeric>
#include <ranges>
#include <sample_camera_paths.hpp>

namespace camera_paths {

Keyframe Keyframe::fromJSON(const nlohmann::json& keyframe)
{
  return Keyframe{
      Camera{
          .pivot         = json_get<glm::vec3>(keyframe["pivot"]),
          .distance      = keyframe["distance"].get<float>(),
          .eulerRotation = json_get<glm::vec3>(keyframe["euler_rotation"]),
          .verticalFov   = keyframe["vertical_fov"].get<float>(),
      },
      keyframe["duration_scale"].get<float>(),
  };
}

Keyframe Keyframe::fromCamera(const Camera& camera)
{
  return {camera};
}

void Keyframe::toCamera(Camera& camera) const
{
  camera = *this;
}

nlohmann::json Keyframe::toJSON() const
{
  return {
      {"pivot", {pivot.x, pivot.y, pivot.z}},
      {"euler_rotation", {eulerRotation.x, eulerRotation.y, eulerRotation.z}},
      {"distance", distance},
      {"vertical_fov", verticalFov},
      {"duration_scale", durationScale},
  };
}

CameraPath::CameraPath(const std::string& newName, const Camera& sampleCamera)
    : name(newName)
    , keyframes({Keyframe::fromCamera(sampleCamera)})
    , selectedIndex(0)
{
}

CameraPath::CameraPath(const nlohmann::json& ipath)
{
  name     = ipath["name"].get<std::string>();
  duration = ipath["duration"].get<float>();
  for(const auto& keyframe : ipath["keyframes"])
    keyframes.push_back(Keyframe::fromJSON(keyframe));
}

nlohmann::json CameraPath::toJSON() const
{
  nlohmann::json jkeyframes;
  for(const auto& cameraPosition : keyframes)
  {
    jkeyframes.push_back(cameraPosition.toJSON());
  }
  return nlohmann::json{
      {"name", name},
      {"duration", duration},
      {"keyframes", jkeyframes},
  };
}

Keyframe CameraPath::interpolate(float t)
{
  assert(keyframes.size() >= 2);

  // Find the current set of keyframes
  size_t             keyframeCount = keyframes.size();
  std::vector<float> runningDurationScale(keyframeCount);
  auto durationScales = keyframes | std::views::transform(&Keyframe::durationScale);
  std::exclusive_scan(durationScales.begin(), durationScales.end(),
                      runningDurationScale.begin(), 0.0f);
  float  durationScaleT = t * runningDurationScale.back();
  size_t currentIndex   = size_t(
      std::distance(runningDurationScale.begin(),
                      std::ranges::upper_bound(runningDurationScale, durationScaleT) - 1));
  float localT = (durationScaleT - runningDurationScale[currentIndex])
                 / keyframes[currentIndex].durationScale;

  // Clamp to the array bounds and give them names
  const Keyframe& kBefore = keyframes[currentIndex == 0 ? 0 : currentIndex - 1];
  const Keyframe& kStart  = keyframes[currentIndex];
  const Keyframe& kEnd =
      keyframes[std::min<size_t>(currentIndex + 1, keyframeCount - 1)];
  const Keyframe& kAfter =
      keyframes[std::min<size_t>(currentIndex + 2, keyframeCount - 1)];

  // Interpolate keyframe positions
  glm::vec3 before = kBefore.pivot;
  glm::vec3 start  = kStart.pivot;
  glm::vec3 end    = kEnd.pivot;
  glm::vec3 after  = kAfter.pivot;

  // Max. dist should be at most half way between start and end, at which point
  // the middle control points could overlap
  float cpDist = cpScale * 0.5f;

  // Interpolate position using a piecewise Bezier spline
  // Optionally limit the control point distance
#if 1
  auto makelength = [](glm::vec3 v, float l) {
    return v * (l / std::max(0.0001f, glm::length(v)));
  };
  float cpStartSize = std::min(glm::length(start - before) * cpDist,
                               glm::length(end - start) * cpDist);
  float cpEndSize =
      std::min(glm::length(end - after) * cpDist, glm::length(end - start) * cpDist);
  glm::vec3 cpStart = start + makelength(end - before, cpStartSize);
  glm::vec3 cpEnd   = end + makelength(start - after, cpEndSize);
#else
  glm::vec3 cpStart = start + (end - before) * (cpDist * 0.5f);
  glm::vec3 cpEnd   = end + (start - after) * (cpDist * 0.5f);
#endif
  auto bezierInterp = [](const glm::vec3& p0, const glm::vec3& p1,
                         const glm::vec3& p2, const glm::vec3& p3, float t) {
    float u   = 1.0f - t;
    float tt  = t * t;
    float uu  = u * u;
    float uuu = uu * u;
    float ttt = tt * t;
    return (uuu * p0) + (3.0f * uu * t * p1) + (3.0f * u * tt * p2) + (ttt * p3);
  };
  glm::vec3 interpPosition = bezierInterp(start, cpStart, cpEnd, end, localT);

  // Interpolate keyframe rotations
  glm::quat beforeQ = glm::quat(glm::yawPitchRoll(
      kBefore.eulerRotation.y, kBefore.eulerRotation.x, kBefore.eulerRotation.z));
  glm::quat startQ =
      glm::quat(glm::yawPitchRoll(kStart.eulerRotation.y, kStart.eulerRotation.x,
                                  kStart.eulerRotation.z));
  glm::quat endQ =
      glm::quat(glm::yawPitchRoll(kEnd.eulerRotation.y, kEnd.eulerRotation.x,
                                  kEnd.eulerRotation.z));
  glm::quat afterQ =
      glm::quat(glm::yawPitchRoll(kAfter.eulerRotation.y, kAfter.eulerRotation.x,
                                  kAfter.eulerRotation.z));

  // Interpolate rotations using a piecewise Bezier spline
  // Optionally limit the control point angular distance
#if 1
  // TODO: way too much normalization happening here
  auto angleDiffQ = [](glm::quat q1, glm::quat q2) {
    if(glm::dot(q1, q2) < 0.0f)
      q2 = -q2;
    return std::abs(glm::angle(glm::normalize(q2 * glm::conjugate(q1))));
  };
  float cpStartAngle = std::min(angleDiffQ(startQ, beforeQ) * cpDist,
                                angleDiffQ(endQ, startQ) * cpDist);
  float cpEndAngle =
      std::min(angleDiffQ(endQ, afterQ) * cpDist, angleDiffQ(endQ, startQ) * cpDist);
  float cpStartAngleRatio = cpStartAngle / std::max(0.0001f, angleDiffQ(beforeQ, endQ));
  float cpEndAngleRatio = cpEndAngle / std::max(0.0001f, angleDiffQ(afterQ, startQ));
  glm::quat cpStartQ = glm::normalize(
      glm::slerp(startQ, startQ * endQ * glm::conjugate(beforeQ), cpStartAngleRatio));
  glm::quat cpEndQ = glm::normalize(
      glm::slerp(endQ, endQ * startQ * glm::conjugate(afterQ), cpEndAngleRatio));
#else
  glm::quat cpStartQ = glm::normalize(
      glm::slerp(startQ, startQ * endQ * glm::conjugate(beforeQ), (cpDist * 0.5f)));
  glm::quat cpEndQ = glm::normalize(
      glm::slerp(endQ, endQ * startQ * glm::conjugate(afterQ), (cpDist * 0.5f)));
#endif
  auto bezierInterpRecursive = [](const auto& p0, const auto& p1, const auto& p2,
                                  const auto& p3, float t, auto lerpFunc) {
    auto p01  = glm::normalize(lerpFunc(p0, p1, t));
    auto p12  = glm::normalize(lerpFunc(p1, p2, t));
    auto p23  = glm::normalize(lerpFunc(p2, p3, t));
    auto p012 = glm::normalize(lerpFunc(p01, p12, t));
    auto p123 = glm::normalize(lerpFunc(p12, p23, t));
    return glm::normalize(lerpFunc(p012, p123, t));
  };
  glm::quat interpQuat = bezierInterpRecursive(startQ, cpStartQ, cpEndQ, endQ, localT,
                                               glm::slerp<float, glm::defaultp>);

  // Reuse the keyframe struct to return interpolated values
  // Using glm::extractEulerAngleYXZ() with a wasteful mat4 intermediate rather
  // than glm::eulerAngles() to match yawPitchRoll's YXZ convention.
  glm::mat4 rotMat = glm::mat4_cast(interpQuat);
  Keyframe  result;
  result.pivot    = interpPosition;
  result.distance = glm::mix(kStart.distance, kEnd.distance, localT);
  glm::extractEulerAngleYXZ(rotMat, result.eulerRotation.y,
                            result.eulerRotation.x, result.eulerRotation.z);
  result.verticalFov = glm::mix(kStart.verticalFov, kEnd.verticalFov, localT);
  return result;
}

void CameraPath::onUIRender(Camera& sampleCamera)
{
  // Edit the camera path name and other properties
  auto resize = [](ImGuiInputTextCallbackData* data) -> int {
    if(data->EventFlag == ImGuiInputTextFlags_CallbackResize)
    {
      auto& str = *static_cast<std::string*>(data->UserData);
      str.resize(size_t(data->BufTextLen));
      data->Buf = str.data();
    }
    return 0;
  };
  ImGui::InputText("Name", name.data(), name.capacity() + 1,
                   ImGuiInputTextFlags_CallbackResize, resize, &name);
  ImGui::SliderFloat("Animation Duration", &duration, 0.1f, 60.0f, "%.1f seconds");
  ImGui::BeginDisabled(keyframes.size() < 2);
  if(ImGui::SliderFloat("Jump to position", &m_seekPosition, 0.0f, 1.0f))
  {
    interpolate(m_seekPosition).toCamera(sampleCamera);
  }
  if(ImGui::SliderFloat("Bezier Control Point Distance", &cpScale, 0.0f, 1.0f))
  {
    interpolate(m_seekPosition).toCamera(sampleCamera);
  }
  ImGui::EndDisabled();

  // Display the list of camera positions
  for(size_t i = 0; i < keyframes.size(); ++i)
  {
    ImGui::PushID(static_cast<int>(i));
    ImGui::SetNextItemAllowOverlap();
    if(ImGui::Selectable(("P " + glm::to_string(keyframes[i].pivot)).c_str(),
                         selectedIndex == static_cast<int>(i)))
    {
      selectedIndex = static_cast<int>(i);
    }

    // Allow reordering
    if(i > 0)
    {
      ImGui::SameLine();
      if(ImGui::Button("Up"))
      {
        std::swap(keyframes[i], keyframes[i - 1]);
        if(selectedIndex == static_cast<int>(i))
          selectedIndex--;
        else if(selectedIndex == static_cast<int>(i - 1))
          selectedIndex++;
      }
    }
    if(i < keyframes.size() - 1)
    {
      ImGui::SameLine();
      if(ImGui::Button("Down"))
      {
        std::swap(keyframes[i], keyframes[i + 1]);
        if(selectedIndex == static_cast<int>(i))
          selectedIndex++;
        else if(selectedIndex == static_cast<int>(i + 1))
          selectedIndex--;
      }
    }
    ImGui::PopID();
  }

  // Add a new camera position
  if(ImGui::Button("Add Camera Position"))
  {
    keyframes.insert(keyframes.begin() + selectedIndex + 1,
                     Keyframe::fromCamera(sampleCamera));
    selectedIndex += 1;
  }
  if(ImGui::Button("Delete Camera Position"))
  {
    if(selectedIndex >= 0 && selectedIndex < static_cast<int>(keyframes.size()))
    {
      keyframes.erase(keyframes.begin() + selectedIndex);
      if(selectedIndex >= static_cast<int>(keyframes.size()))
      {
        selectedIndex = static_cast<int>(keyframes.size()) - 1;
      }
    }
  }

  // Load and write the current camera manipulator to the selected camera position
  if(selectedIndex >= 0 && selectedIndex < static_cast<int>(keyframes.size()))
  {
    if(ImGui::Button("Save"))
      keyframes[size_t(selectedIndex)] = Keyframe::fromCamera(sampleCamera);
    ImGui::SameLine();
    if(ImGui::Button("Load to Camera"))
      keyframes[size_t(selectedIndex)].toCamera(sampleCamera);
  }

  // Display details of the selected camera position
  if(selectedIndex >= 0 && selectedIndex < static_cast<int>(keyframes.size()))
  {
    Keyframe& selectedCamera = keyframes[size_t(selectedIndex)];
    ImGui::Text("Selected Camera Position:");
    ImGui::InputFloat3("Pivot", &selectedCamera.pivot[0]);
    ImGui::InputFloat3("Euler Rotation", &selectedCamera.eulerRotation[0]);
    ImGui::InputFloat("Duration Scale", &selectedCamera.durationScale);
  }
}

}  // namespace camera_paths

template <typename T>
std::optional<T> envVar(const char* name)
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv is safe for read-only access
#endif
  const char* value = std::getenv(name);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  return value ? std::make_optional<T>(value) : std::nullopt;
}

inline std::filesystem::path cameraPathsJsonPath()
{
  return envVar<std::filesystem::path>("APPIMAGE")
             .value_or(std::filesystem::current_path())
             .parent_path()
         / CameraPathsElement::PathsFilename;
}

CameraPathsElement::CameraPathsElement(Camera& sampleCamera)
    : m_sampleCamera(&sampleCamera)
{
  if(std::filesystem::exists(cameraPathsJsonPath()))
    for(auto ipath : nlohmann::json::parse(std::ifstream{cameraPathsJsonPath()}))
      m_cameraPaths.emplace_back(ipath);
  m_cameraPathIndex = m_cameraPaths.empty() ? -1 : 0;
}

CameraPathsElement::~CameraPathsElement()
{
  try
  {
    nlohmann::json outputJson;
    for(const auto& path : m_cameraPaths)
      outputJson.push_back(path.toJSON());
    std::ofstream outFile(cameraPathsJsonPath());
    outFile << outputJson.dump(4);  // Write JSON with 4-space indentation
    outFile.close();
  }
  catch(const std::exception& e)
  {
    fprintf(stderr, "Error writing camera paths: %s\n", e.what());
  }
}

void CameraPathsElement::renderUI()
{
  if(!m_showWindow)
    return;

  // Set initial window size (only on first appearance)
  ImGui::SetNextWindowSize(ImVec2(400, 600), ImGuiCond_FirstUseEver);

  if(!ImGui::Begin(WindowName, &m_showWindow))
  {
    ImGui::End();
    return;
  }

  // Current camera widget
  if(ImGui::CollapsingHeader("Current Camera", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::DragFloat3("Pivot", &m_sampleCamera->pivot.x, 0.1f);
    ImGui::DragFloat("Distance", &m_sampleCamera->distance, 0.01f, 0.001f, FLT_MAX, "%.3f");

    // Show euler angles in degrees for readability
    glm::vec3 eulerDegrees = glm::degrees(m_sampleCamera->eulerRotation);
    if(ImGui::DragFloat3("Rotation (deg)", &eulerDegrees.x, 1.0f))
    {
      m_sampleCamera->eulerRotation = glm::radians(eulerDegrees);
      m_sampleCamera->clampEuler();
    }

    float fovDegrees = glm::degrees(m_sampleCamera->verticalFov);
    if(ImGui::DragFloat("FOV (deg)", &fovDegrees, 1.0f, 1.0f, 179.0f))
    {
      m_sampleCamera->verticalFov = glm::radians(fovDegrees);
    }

    ImGui::DragFloat2("Clip Planes", &m_sampleCamera->clipPlanes.x, 0.01f,
                      0.0001f, FLT_MAX, "%.4f");
  }

  ImGui::Separator();

  // Dropdown for selecting path
  if(ImGui::BeginCombo("Path", m_cameraPathIndex == -1 ? "None" :
                                                         currentPath().name.c_str()))
  {
    for(int i = 0; i < static_cast<int>(m_cameraPaths.size()); i++)
    {
      bool isSelected = (i == m_cameraPathIndex);
      if(ImGui::Selectable(m_cameraPaths[size_t(i)].name.c_str(), isSelected))
      {
        m_cameraPathIndex = i;
      }
      if(isSelected)
      {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  // Button to create a new camera path
  if(ImGui::Button("Create New Path"))
  {
    m_cameraPaths.push_back(camera_paths::CameraPath("My Camera Path", *m_sampleCamera));
    m_cameraPathIndex = static_cast<int>(m_cameraPaths.size()) - 1;
  }

  // If a path is selected, display its editor
  if(m_cameraPathIndex != -1
     && m_cameraPathIndex < static_cast<int>(m_cameraPaths.size()))
  {
    ImGui::Separator();

    // Animation controls - frame saving options
    ImGui::Checkbox("Save Frames", &m_saveFrames);
    if(ImGui::IsItemHovered())
      ImGui::SetTooltip("Save each frame as PNG during playback (for video export)");

    if(m_saveFrames)
    {
      auto basePath =
          envVar<std::filesystem::path>("OWD").value_or(std::filesystem::current_path());
      ImGui::Indent();
      ImGui::InputText("Output Dir", &m_frameSavePath);
      if(ImGui::IsItemHovered())
        ImGui::SetTooltip("Relative to: %s", basePath.string().c_str());
      ImGui::InputText("File Prefix", &m_framePrefix);
      if(ImGui::IsItemHovered())
        ImGui::SetTooltip("Output: %s", (basePath / m_frameSavePath / (m_framePrefix + "_0001.png"))
                                            .string()
                                            .c_str());
      ImGui::Unindent();
    }

    ImGui::BeginDisabled(currentPath().keyframes.size() < 2);
    if(ImGui::Button("Play"))
    {
      m_cameraAnimating        = true;
      m_cameraAnimateLastFrame = std::chrono::high_resolution_clock::now();
      m_cameraAnimatePosition  = duration::zero();
      m_cameraAnimationFrame   = 0;
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    if(ImGui::Button("Stop"))
    {
      m_cameraAnimating = false;
    }

    ImGui::Separator();

    // Display the keyframe editor
    currentPath().onUIRender(*m_sampleCamera);
  }

  ImGui::End();
}

void CameraPathsElement::update()
{
  if(!m_cameraAnimating || m_cameraPathIndex == -1
     || m_cameraPathIndex >= static_cast<int>(m_cameraPaths.size()))
    return;

  // Use fixed timestep for deterministic animation (for frame saving)
  // This matches the typical 60fps target
  m_cameraAnimatePosition += std::chrono::microseconds(16666);

  // Interpolate camera position
  m_cameraAnimationFrame++;
  float animationTime = std::chrono::duration<float>(m_cameraAnimatePosition).count();
  float t = glm::clamp(animationTime / currentPath().duration, 0.0f, 1.0f);

  if(animationTime > currentPath().duration)
  {
    m_cameraAnimating = false;
  }

  currentPath().interpolate(t).toCamera(*m_sampleCamera);
}

std::filesystem::path CameraPathsElement::framePath(int frameNum) const
{
  // Use OWD (Original Working Directory) if running inside AppImage, otherwise CWD
  auto basePath =
      envVar<std::filesystem::path>("OWD").value_or(std::filesystem::current_path());
  return basePath / m_frameSavePath / std::format("{}_{:04d}.png", m_framePrefix, frameNum);
}
