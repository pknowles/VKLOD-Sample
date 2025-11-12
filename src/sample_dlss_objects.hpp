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

// This file is based on utilities from:
//
//     https://github.com/pknowles/vulkan_objects/blob/main/include/vko/nv_ngx.hpp
//     Copyright (c) 2025 Pyarelal Knowles, MIT License
//
// See https://github.com/nvpro-samples/vk_denoise_dlssrr/ for a more complete
// example.

#pragma once

#include <codecvt>
#include <locale>
#include <memory>
#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd.h>
#include <nvsdk_ngx_helpers_vk.h>

// must be after nvsdk_ngx_helpers_vk.h
#include <nvsdk_ngx_helpers_dlssd_vk.h>

/*
    // Example usage:

    // Create the instance and device with the required extensions
    // Make sure to set NVSDK_NGX_FeatureCommonInfo::PathListInfo as this is
    // where NGX goes looking for DLLs implementing its features.
    ngx::requiredInstanceExtensions(NVSDK_NGX_FeatureDiscoveryInfo{...})
    ...
    ngx::requiredDeviceExtensions(instance, physicalDevice,
                                       NVSDK_NGX_FeatureDiscoveryInfo{...})

    // Global init
    ngx::ScopedInit ngx(ngxApplicationId, ngxApplicationPath, instance, physicalDevice, device,
                             globalCommands.vkGetInstanceProcAddr, instance.vkGetDeviceProcAddr,
                             &ngxCommonInfo)

    // Dependent on output window size/resize
    ngx::CapabilityParameter ngxParameter;
    ngx::OptimalSettings dlssOptimal(ngxParameter, outputSize.x, outputSize.y,
   NVSDK_NGX_PerfQuality_Value_MaxQuality);
    ... resize G-buffer to {dlssOptimal.renderOptimalWidth, dlssOptimal.renderOptimalHeight}
    ngx::RayReconstruction dlssrr(device, cmd, 1u, 1u, ngxParameter,
   NVSDK_NGX_DLSSD_Create_Params{...});

    // Upscale and denoise
    glm::mat4 viewRowMajor       = glm::transpose(viewMatrix);
    glm::mat4 projectionRowMajor = glm::transpose(projectionMatrix);
    // NOTE: Make sure these have VK_IMAGE_USAGE_SAMPLED_BIT!
    NVSDK_NGX_Resource_VK colorOut = NVSDK_NGX_Create_ImageView_Resource_VK(...);
    NVSDK_NGX_Resource_VK color    = NVSDK_NGX_Create_ImageView_Resource_VK(...);
    ...
    NVSDK_NGX_VK_DLSSD_Eval_Params dlssdEvalParams{};
    dlssdEvalParams.pInOutput                        = &colorOut;
    dlssdEvalParams.pInColor                         = &color;
    dlssdEvalParams.pInDiffuseAlbedo                 = &albedo;
    dlssdEvalParams.pInSpecularAlbedo                = &specularAlbedo;
    dlssdEvalParams.pInSpecularHitDistance           = &specularHitDistance;
    dlssdEvalParams.pInNormals                       = &normalRoughness;
    dlssdEvalParams.pInDepth                         = &linearDepth;
    dlssdEvalParams.pInMotionVectors                 = &motionVector;
    dlssdEvalParams.pInRoughness                     = &normalRoughness;
    dlssdEvalParams.InJitterOffsetX                  = -jitter.x;
    dlssdEvalParams.InJitterOffsetY                  = -jitter.y;
    dlssdEvalParams.InMVScaleX                       = 1.0f;
    dlssdEvalParams.InMVScaleY                       = 1.0f;
    dlssdEvalParams.InRenderSubrectDimensions.Width  = inputSize.x;
    dlssdEvalParams.InRenderSubrectDimensions.Height = inputSize.y;
    dlssdEvalParams.pInWorldToViewMatrix             = glm::value_ptr(viewRowMajor);
    dlssdEvalParams.pInViewToClipMatrix              = glm::value_ptr(projectionRowMajor);
    dlssdEvalParams.InReset                          = frameIndex <= 1;
    dlssrr->evaluate(commandBuffer, ngxParameter, dlssdEvalParams);
*/

namespace ngx {

inline std::vector<const wchar_t*> makeWcharPtrs(const std::vector<std::wstring>& strings)
{
  std::vector<const wchar_t*> result(strings.size(), nullptr);
  for(size_t i = 0; i < strings.size(); ++i)
  {
    result[i] = strings[i].c_str();
  }
  return result;
}

struct FeatureDiscovery
{
#if defined(NDEBUG)
  static constexpr NVSDK_NGX_Logging_Level minLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
#else
  static constexpr NVSDK_NGX_Logging_Level minLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;  // NVSDK_NGX_LOGGING_LEVEL_VERBOSE
#endif
  std::wstring                   applicationDataPath;
  std::vector<std::wstring>      ngxSearchPaths;
  std::vector<const wchar_t*>    ngxSearchPathPtrs;
  NVSDK_NGX_FeatureCommonInfo    commonInfo;  // TODO: is this even needed
  NVSDK_NGX_FeatureDiscoveryInfo discoveryInfo;
  FeatureDiscovery(unsigned long long applicationId, const std::wstring& appDataPath,
                    NVSDK_NGX_Feature feature, std::initializer_list<std::wstring_view> searchPaths)
      : applicationDataPath(appDataPath)
      , ngxSearchPaths(searchPaths.begin(), searchPaths.end())
      , ngxSearchPathPtrs(makeWcharPtrs(ngxSearchPaths))
      , commonInfo{
        .PathListInfo = {.Path = ngxSearchPathPtrs.data(),
                         .Length = static_cast<unsigned int>(ngxSearchPathPtrs.size())},
        .InternalData = nullptr,
        .LoggingInfo =
            {
                .LoggingCallback = nullptr, /*
                    [](const char* message, NVSDK_NGX_Logging_Level loggingLevel,
                       NVSDK_NGX_Feature sourceComponent) {
                        fprintf(stderr, "NGX: %s (level %u, feature %u)", message,
                                loggingLevel, sourceComponent);
                    },*/
                .MinimumLoggingLevel      = minLoggingLevel,
                .DisableOtherLoggingSinks = {},
            },
        }
      , discoveryInfo{
            .SDKVersion = NVSDK_NGX_Version_API,
            .FeatureID  = feature,
            .Identifier = {.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id,
                            .v              = {.ApplicationId = applicationId}},
            .ApplicationDataPath = applicationDataPath.c_str(), // Internal raw pointer
            .FeatureInfo         = &commonInfo,                 // Internal raw pointer
        }
  {
  }
  operator const NVSDK_NGX_FeatureDiscoveryInfo&() const { return discoveryInfo; }
  FeatureDiscovery(const FeatureDiscovery& other)            = delete;
  FeatureDiscovery& operator=(const FeatureDiscovery& other) = delete;
};


// because std::wstring_convert is deprecated, utf8 is the future and enums
// won't have non-ascii characters anyway
inline std::string wchar_to_ascii(const wchar_t* wstr)
{
  if(!wstr)
    return {};

  std::string result;
  for(; *wstr; ++wstr)
  {
    if(*wstr > 127)
    {
      throw std::runtime_error("Non-ASCII character encountered");
    }
    result += static_cast<char>(*wstr);
  }
  return result;
}

class ResultException : public std::runtime_error
{
public:
  ResultException(NVSDK_NGX_Result result)
      : std::runtime_error("NGX error: " + wchar_to_ascii(GetNGXResultAsString(result)))
  {
  }
};

inline void check(NVSDK_NGX_Result result)
{
  if(NVSDK_NGX_FAILED(result))
  {
    throw ResultException(result);
  }
}

// Straight init wrapper and error handling
class ScopedInit
{
public:
  ScopedInit(const FeatureDiscovery&   featureDiscovery,
             VkInstance                instance,
             VkPhysicalDevice          physicalDevice,
             VkDevice                  device,
             PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr,
             PFN_vkGetDeviceProcAddr   vkGetDeviceProcAddr,
             NVSDK_NGX_Version         sdkVersion = NVSDK_NGX_Version_API)
      : ScopedInit(featureDiscovery.discoveryInfo.Identifier.v.ApplicationId,
                   featureDiscovery.applicationDataPath,
                   instance,
                   physicalDevice,
                   device,
                   vkGetInstanceProcAddr,
                   vkGetDeviceProcAddr,
                   &featureDiscovery.commonInfo,
                   sdkVersion)
  {
  }
  ScopedInit(unsigned long long                 applicationId,
             const std::wstring&                applicationDataPath,
             VkInstance                         instance,
             VkPhysicalDevice                   physicalDevice,
             VkDevice                           device,
             PFN_vkGetInstanceProcAddr          vkGetInstanceProcAddr,
             PFN_vkGetDeviceProcAddr            vkGetDeviceProcAddr,
             const NVSDK_NGX_FeatureCommonInfo* featureInfo = nullptr,
             NVSDK_NGX_Version                  sdkVersion  = NVSDK_NGX_Version_API)
      : m_device(device)
  {
    check(NVSDK_NGX_VULKAN_Init(applicationId, applicationDataPath.c_str(), instance, physicalDevice, device,
                                vkGetInstanceProcAddr, vkGetDeviceProcAddr, featureInfo, sdkVersion));
  }
  ScopedInit(const std::string&                 projectId,
             NVSDK_NGX_EngineType               engineType,
             const std::string&                 engineVersion,
             const std::wstring&                applicationDataPath,
             VkInstance                         instance,
             VkPhysicalDevice                   physicalDevice,
             VkDevice                           device,
             PFN_vkGetInstanceProcAddr          vkGetInstanceProcAddr,
             PFN_vkGetDeviceProcAddr            vkGetDeviceProcAddr,
             const NVSDK_NGX_FeatureCommonInfo* featureInfo = nullptr,
             NVSDK_NGX_Version                  sdkVersion  = NVSDK_NGX_Version_API)
      : m_device(device)
  {
    check(NVSDK_NGX_VULKAN_Init_with_ProjectID(projectId.c_str(), engineType, engineVersion.c_str(),
                                               applicationDataPath.c_str(), instance, physicalDevice, device,
                                               vkGetInstanceProcAddr, vkGetDeviceProcAddr, featureInfo, sdkVersion));
  }
  ~ScopedInit()
  {
    if(NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Shutdown1(m_device); NVSDK_NGX_FAILED(result))
    {
      // Print and ignore as we're in a destructor
      fprintf(stderr, "NGX error: %s\n", wchar_to_ascii(GetNGXResultAsString(result)).c_str());
    }
  }
  ScopedInit(const ScopedInit& other)           = delete;
  ScopedInit operator=(const ScopedInit& other) = delete;

private:
  VkDevice m_device = VK_NULL_HANDLE;
};

struct ParameterDeleter
{
  void operator()(NVSDK_NGX_Parameter* p) const { NVSDK_NGX_VULKAN_DestroyParameters(p); }
};

inline std::unique_ptr<NVSDK_NGX_Parameter, ParameterDeleter> makeCapabilityParameter()
{
  NVSDK_NGX_Parameter* parameter = nullptr;
  check(NVSDK_NGX_VULKAN_GetCapabilityParameters(&parameter));
  return std::unique_ptr<NVSDK_NGX_Parameter, ParameterDeleter>{parameter};
}

using CapabilityParameter = std::unique_ptr<NVSDK_NGX_Parameter, ParameterDeleter>;

template <class T>
T get(const NVSDK_NGX_Parameter& parameter, const char* name)
{
  T result;
  check(parameter.Get(name, &result));
  return result;
}

inline std::span<const VkExtensionProperties> requiredInstanceExtensions(const NVSDK_NGX_FeatureDiscoveryInfo& featureDiscoveryInfo)
{
  uint32_t               extensionCount = 0;
  VkExtensionProperties* extensions     = nullptr;
  check(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&featureDiscoveryInfo, &extensionCount, &extensions));
  return {extensions, extensionCount};
}

inline std::span<const VkExtensionProperties> requiredDeviceExtensions(VkInstance       instance,
                                                                       VkPhysicalDevice physicalDevice,
                                                                       const NVSDK_NGX_FeatureDiscoveryInfo& featureDiscoveryInfo)
{
  uint32_t               extensionCount = 0;
  VkExtensionProperties* extensions     = nullptr;
  check(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, physicalDevice, &featureDiscoveryInfo,
                                                               &extensionCount, &extensions));
  return {extensions, extensionCount};
}

struct OptimalSettings
{
  OptimalSettings(NVSDK_NGX_Parameter& parameter, unsigned int selectedWidth, unsigned int selectedHeight, NVSDK_NGX_PerfQuality_Value qualityValue)
  {
    check(NGX_DLSSD_GET_OPTIMAL_SETTINGS(&parameter, selectedWidth, selectedHeight, qualityValue, &renderOptimalWidth, &renderOptimalHeight,
                                         &renderMaxWidth, &renderMaxHeight, &renderMinWidth, &renderMinHeight, &sharpness));
  }
  unsigned int renderOptimalWidth;
  unsigned int renderOptimalHeight;
  unsigned int renderMaxWidth;
  unsigned int renderMaxHeight;
  unsigned int renderMinWidth;
  unsigned int renderMinHeight;
  float        sharpness;
};

template <auto CreateFunc>
class Handle
{
public:
  template <class... Args>
  Handle(Args&&... args)
      : m_handle(CreateFunc(std::forward<Args>(args)...))
  {
  }
  ~Handle() { destroy(); }
  Handle(const Handle&)            = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept
      : m_handle(other.m_handle)
  {
    other.m_handle = nullptr;
  }
  Handle& operator=(Handle&& other) noexcept
  {
    destroy();
    m_handle       = other.m_handle;
    other.m_handle = nullptr;
    return *this;
  }
  operator NVSDK_NGX_Handle*() const { return m_handle; }

private:
  void destroy()
  {
    if(m_handle)
      NVSDK_NGX_VULKAN_ReleaseFeature(m_handle);
  }
  NVSDK_NGX_Handle* m_handle = nullptr;
};

inline NVSDK_NGX_Handle* makeRayReconstruction(VkDevice                             device,
                                               VkCommandBuffer                      commandBuffer,
                                               unsigned int                         creationNodeMask,
                                               unsigned int                         visibilityNodeMask,
                                               NVSDK_NGX_Parameter&                 parameter,
                                               const NVSDK_NGX_DLSSD_Create_Params& dlssDCreateParams)
{
  // ideally dlssDCreateParams would be const in the API
  NVSDK_NGX_Handle* handle = nullptr;
  check(NGX_VULKAN_CREATE_DLSSD_EXT1(device, commandBuffer, creationNodeMask, visibilityNodeMask, &handle, &parameter,
                                     const_cast<NVSDK_NGX_DLSSD_Create_Params*>(&dlssDCreateParams)));
  return handle;
}

// Example/demonstration
inline void assertRayReconstructionSupported(NVSDK_NGX_Parameter& parameter)
{
  auto minMajor = get<unsigned>(parameter, NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMajor);
  auto minMinor = get<unsigned>(parameter, NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMinor);
  if(get<int>(parameter, NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver) != 0)
    throw std::runtime_error("NGX Super Sampling Denoising needs a driver update: Min. version "
                             + std::to_string(minMajor) + "." + std::to_string(minMinor));
  if(get<int>(parameter, NVSDK_NGX_Parameter_SuperSamplingDenoising_Available) == 0)
    throw std::runtime_error("NGX Super Sampling Denoising is not available");
  if(get<int>(parameter, NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult) == 0)
    throw std::runtime_error("NGX Super Sampling Denoising FeatureInitResult was 0");
}

class RayReconstruction : public Handle<makeRayReconstruction>
{
public:
  using Handle<makeRayReconstruction>::Handle;
  void evaluate(VkCommandBuffer commandBuffer, NVSDK_NGX_Parameter& parameter, const NVSDK_NGX_VK_DLSSD_Eval_Params& evalParams)
  {
    // ideally evalParams would be const in the API
    check(NGX_VULKAN_EVALUATE_DLSSD_EXT(commandBuffer, *this, &parameter, const_cast<NVSDK_NGX_VK_DLSSD_Eval_Params*>(&evalParams)));
  }
};

}  // namespace ngx
