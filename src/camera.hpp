// Copyright (c) 2025 Pyarelal Knowles, Apache-2.0
#pragma once

#include <GLFW/glfw3.h>
#include <chrono>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <limits>

struct Camera
{
  glm::vec3 pivot         = {0.0f, 0.0f, 0.0f};
  float     distance      = 2.0f;
  glm::vec3 eulerRotation = {-1.0f, 1.0f, 0.0f};
  float     verticalFov   = glm::radians(80.0f);
  glm::vec2 clipPlanes    = {0.001f, 1000.0f};

  glm::mat4 viewInv() const
  {
    glm::mat4 result = glm::identity<glm::mat4>();
    result = glm::translate(glm::identity<glm::mat4>(), {0.0f, 0.0f, distance}) * result;
    result = glm::yawPitchRoll(eulerRotation.y, eulerRotation.x, eulerRotation.z) * result;
    result = glm::translate(glm::identity<glm::mat4>(), pivot) * result;
    return result;
  }

  glm::mat4 view() const
  {
    glm::mat4 result = glm::identity<glm::mat4>();
    result = glm::translate(glm::identity<glm::mat4>(), -pivot) * result;
    result = glm::transpose(glm::yawPitchRoll(eulerRotation.y, eulerRotation.x,
                                              eulerRotation.z))
             * result;
    result = glm::translate(glm::identity<glm::mat4>(), {0.0f, 0.0f, -distance}) * result;
    return result;
  }

  void setLookat(glm::vec3 eye, glm::vec3 center, glm::vec3 up = {0.0f, 1.0f, 0.0f})
  {
    pivot    = center;
    distance = glm::length(eye - center);
    if(distance > 0.0f)
    {
      glm::mat4 view = glm::lookAt(eye, center, up);
      // Extract and transpose rotation (upper 3x3) to get camera-to-world rotation
      glm::mat3 rotation = glm::transpose(glm::mat3(view));
      glm::extractEulerAngleYXZ(glm::mat4(rotation), eulerRotation.y,
                                eulerRotation.x, eulerRotation.z);
    }
  }

  void clampEuler()
  {
    eulerRotation.x = glm::clamp(eulerRotation.x, -0.4999f * glm::pi<float>(),
                                 0.4999f * glm::pi<float>());
    eulerRotation.y = fmodf(eulerRotation.y, glm::two_pi<float>());
    eulerRotation.z = glm::clamp(eulerRotation.z, -0.4999f * glm::pi<float>(),
                                 0.4999f * glm::pi<float>());
  };

  glm::vec3 rotate(glm::vec3 v) const
  {
    return glm::yawPitchRoll(eulerRotation.y, eulerRotation.x, eulerRotation.z)
           * glm::vec4{v, 1.0f};
  }

  glm::vec3 forward() const { return rotate({0.0f, 0.0f, -1.0f}); }

  glm::vec3 right() const { return rotate({1.0f, 0.0f, 0.0f}); }

  glm::vec3 up() const { return rotate({0.0f, 1.0f, 0.0f}); }
};

// Input state and manipulation for camera with GLFW
struct CameraManipulator
{
  glm::vec2 lastMousePos = {0.0f, 0.0f};
  std::chrono::steady_clock::time_point lastTime = std::chrono::steady_clock::now();

  // Manipulate camera with GLFW mouse/keyboard input
  // Call this each frame to keep internal state updated.
  // wheelDelta is for scroll wheel zoom (positive = zoom in).
  // hasFocus controls whether input actually affects the camera.
  // isHovered controls scroll (separate from focus so drags work outside viewport).
  void manipulateWithGLFW(GLFWwindow* window,
                          Camera&     camera,
                          float       wheelDelta = 0.0f,
                          bool        hasFocus   = true,
                          bool        isHovered  = true)
  {
    // Compute delta time
    auto                         currentTime = std::chrono::steady_clock::now();
    std::chrono::duration<float> delta       = currentTime - lastTime;
    lastTime                                 = currentTime;
    float dt                                 = delta.count();

    // Get current mouse position
    glm::vec2 mousePos;
    double    xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    mousePos = {static_cast<float>(xpos), static_cast<float>(ypos)};

    // Compute mouse delta
    glm::vec2 mouseDelta = mousePos - lastMousePos;
    lastMousePos         = mousePos;

    // Scroll wheel zoom (only requires hover, not focus)
    if(wheelDelta != 0.0f && isHovered)
    {
      const float wheelZoomSpeed = 0.1f;
      camera.distance =
          std::max(std::numeric_limits<float>::min(),
                   std::expf(std::logf(camera.distance) - wheelDelta * wheelZoomSpeed));
    }

    // Mouse/keyboard input requires focus
    if(!hasFocus)
      return;

    // Sensitivity settings
    const float rotateSpeed = 0.005f;
    const float panSpeed    = 0.002f;  // Multiplied by camera.distance
    const float zoomSpeed   = 0.005f;  // Applied in log space for linear feel
    const float moveSpeed   = 2.0f;

    // Mouse button states
    bool leftButton = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    bool middleButton = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    bool rightButton = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    // Orbit: Left mouse button
    if(leftButton)
    {
      camera.eulerRotation.y -= mouseDelta.x * rotateSpeed;
      camera.eulerRotation.x -= mouseDelta.y * rotateSpeed;
      camera.clampEuler();
    }

    // Pan: Middle mouse button (distance-relative)
    if(middleButton)
    {
      glm::vec3 cameraSpacePan = {mouseDelta.x, -mouseDelta.y, 0.0f};
      glm::vec3 worldSpacePan  = camera.rotate(cameraSpacePan);
      camera.pivot -= worldSpacePan * panSpeed * camera.distance;
    }

    // Zoom: Right mouse button (in log space for linear feel, distance-relative)
    if(rightButton)
    {
      camera.distance =
          std::max(std::numeric_limits<float>::min(),
                   std::expf(std::logf(camera.distance) + mouseDelta.y * zoomSpeed));
    }

    // Keyboard movement (WASD + QE for up/down)
    {
      // Build camera-space movement vector
      glm::vec3 cameraSpaceMovement = {0.0f, 0.0f, 0.0f};

      if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
      {
        cameraSpaceMovement.z -= 1.0f;
      }
      if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
      {
        cameraSpaceMovement.z += 1.0f;
      }
      if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
      {
        cameraSpaceMovement.x -= 1.0f;
      }
      if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
      {
        cameraSpaceMovement.x += 1.0f;
      }
      if(glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
      {
        cameraSpaceMovement.y -= 1.0f;
      }
      if(glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
      {
        cameraSpaceMovement.y += 1.0f;
      }

      // Apply single rotation to camera-space movement vector
      if(glm::length(cameraSpaceMovement) > 0.0f)
      {
        glm::vec3 worldSpaceMovement = camera.rotate(cameraSpaceMovement);
        camera.pivot += worldSpaceMovement * moveSpeed * dt * camera.distance * 0.1f;
      }
    }
  }
};
