// Copyright (c) 2026 Pyarelal Knowles, MIT License
#pragma once

#include <array>
#include <map>
#include <vko/gen_structures.hpp>
#include <vko/glfw_objects.hpp>
#include <vko/handles.hpp>
#include <vko/nv_ngx.hpp>
#include <vko/shortcuts.hpp>
#include <vulkan/vulkan_core.h>

static constexpr uint32_t SampleVulkanVersion = VK_API_VERSION_1_4;

// Helper to reduce boilerplate when initializing Vulkan feature structs.
// Sets sType and pNext automatically, then calls the lambda to enable specific fields.
template <class T, class Fn>
  requires std::invocable<Fn, T&>
T makeFeatures(void* pNext, Fn&& fn)
{
  T t{};  // All fields to false/zero
  t.sType = vko::struct_traits<T>::sType;
  t.pNext = pNext;
  fn(t);  // User lambda to switch some back on
  return t;
}

template <typename... Bools>
concept all_bool = (std::same_as<std::remove_cvref_t<Bools>, bool> && ...);

// Shortcut for when there's only a single member
template <class T, all_bool... Bools>
T makeFeatures(void* pNext, Bools... bools)
{
  return T{vko::struct_traits<T>::sType, pNext,
           (bools ? VkBool32{VK_TRUE} : VkBool32{VK_FALSE})...};
}

// Non-copyable struct encapsulating VkInstanceCreateInfo with sample-specific extensions.
// Internal pointers remain valid for the lifetime of the struct.
struct SampleInstanceCreateInfo
{
  vko::glfw::PlatformSupport*           platformSupport;
  const NVSDK_NGX_FeatureDiscoveryInfo* ngxFeatureDiscovery;
  bool                                  enableValidation;

  // Storage for extension/layer names (vectors own the const char* pointers to static strings)
  std::vector<const char*> extensions;
  std::vector<const char*> layers;

  VkApplicationInfo    appInfo;
  VkInstanceCreateInfo createInfo;

  const VkBool32 verboseValidateSync = true;
  const VkLayerSettingEXT layerSetting = {"VK_LAYER_KHRONOS_validation", "validate_sync",
                                          VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1,
                                          &verboseValidateSync};
  VkLayerSettingsCreateInfoEXT layerSettingsCreateInfo = {
      VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT, nullptr, 1, &layerSetting};

  SampleInstanceCreateInfo(vko::glfw::PlatformSupport&           platformSupport_,
                             const NVSDK_NGX_FeatureDiscoveryInfo* ngxFeatureDiscovery_ = nullptr,
                             bool                                  enableValidation_    = false)
        : platformSupport(&platformSupport_)
        , ngxFeatureDiscovery(ngxFeatureDiscovery_)
        , enableValidation(enableValidation_)
        , extensions{
              VK_KHR_SURFACE_EXTENSION_NAME,
              VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME,
              VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME,
              vko::glfw::platformSurfaceExtension(platformSupport_),
              VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
          }
        , layers{}
        , appInfo{
              .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
              .pNext              = nullptr,
              .pApplicationName   = "VKLOD-Sample",
              .applicationVersion = 0,
              .pEngineName        = nullptr,
              .engineVersion      = 0,
              .apiVersion         = SampleVulkanVersion,
          }
        , createInfo{}
  {
    // Add NGX instance extensions if provided
    if(ngxFeatureDiscovery)
    {
      vko::ngx::withRequiredInstanceExtensions(*ngxFeatureDiscovery, [&](auto&& ngxExts) {
        for(const auto& ext : ngxExts)
        {
          extensions.push_back(ext.extensionName);
        }
      });
    }

    // Add validation layer if requested
    if(enableValidation)
    {
      layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    // Build the create info (must be done after extensions/layers are finalized)
    createInfo = VkInstanceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = enableValidation ? &layerSettingsCreateInfo : nullptr,
        .flags = 0,
        .pApplicationInfo        = &appInfo,
        .enabledLayerCount       = uint32_t(layers.size()),
        .ppEnabledLayerNames     = layers.data(),
        .enabledExtensionCount   = uint32_t(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
  }

  operator const VkInstanceCreateInfo&() const { return createInfo; }

  // Non-copyable to protect internal pointers
  SampleInstanceCreateInfo(const SampleInstanceCreateInfo&)            = delete;
  SampleInstanceCreateInfo& operator=(const SampleInstanceCreateInfo&) = delete;
};

// A queue's family and index within that family (self-describing)
struct QueueLocation
{
  uint32_t family;
  uint32_t index;
};

// All queues needed by the sample, with their locations
struct SampleQueues
{
  QueueLocation primary;    // GRAPHICS | COMPUTE | TRANSFER - primary rendering
  QueueLocation asyncLoad;  // Same family as primary, for async scene loading
  QueueLocation compute;    // COMPUTE (prefer dedicated compute queue)
  QueueLocation transfer;   // TRANSFER (prefer dedicated transfer queue)
};

// Encapsulates all Vulkan feature structs required by the sample, with pNext chain wired up.
// Non-copyable so the pNext chain remains valid.
struct SampleDeviceFeatures
{
  VkPhysicalDevicePresentWaitFeaturesKHR presentWait =
      makeFeatures<VkPhysicalDevicePresentWaitFeaturesKHR>(nullptr, [](auto& f) {
        f.presentWait = VK_TRUE;
      });

  VkPhysicalDevicePresentIdFeaturesKHR presentId =
      makeFeatures<VkPhysicalDevicePresentIdFeaturesKHR>(&presentWait, [](auto& f) {
        f.presentId = VK_TRUE;
      });

  VkPhysicalDeviceMaintenance7FeaturesKHR maintenance7{
      .sType = vko::struct_traits<VkPhysicalDeviceMaintenance7FeaturesKHR>::sType,
      .pNext        = &presentId,
      .maintenance7 = VK_TRUE,
  };

  VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swapchainMaintenance1 =
      makeFeatures<VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT>(&maintenance7, true);

  VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clusterAccelerationStructure =
      makeFeatures<VkPhysicalDeviceClusterAccelerationStructureFeaturesNV>(
          &swapchainMaintenance1,
          [](auto& f) { f.clusterAccelerationStructure = VK_TRUE; });

  VkPhysicalDeviceCopyMemoryIndirectFeaturesNV copyMemoryIndirect =
      makeFeatures<VkPhysicalDeviceCopyMemoryIndirectFeaturesNV>(
          &clusterAccelerationStructure,
          [](auto& f) { f.indirectCopy = VK_FALSE; });

  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR fragmentShaderBarycentric =
      makeFeatures<VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR>(
          &copyMemoryIndirect,
          [](auto& f) { f.fragmentShaderBarycentric = VK_TRUE; });

  VkPhysicalDeviceMeshShaderFeaturesNV meshShader =
      makeFeatures<VkPhysicalDeviceMeshShaderFeaturesNV>(&fragmentShaderBarycentric,
                                                         [](auto& f) {
                                                           f.taskShader = VK_FALSE;
                                                           f.meshShader = VK_TRUE;
                                                         });

  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR rayTracingPositionFetch =
      makeFeatures<VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR>(
          &meshShader,
          [](auto& f) { f.rayTracingPositionFetch = VK_TRUE; });

  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipeline =
      makeFeatures<VkPhysicalDeviceRayTracingPipelineFeaturesKHR>(
          &rayTracingPositionFetch,
          [](auto& f) { f.rayTracingPipeline = VK_TRUE; });

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure =
      makeFeatures<VkPhysicalDeviceAccelerationStructureFeaturesKHR>(
          &rayTracingPipeline,
          [](auto& f) { f.accelerationStructure = VK_TRUE; });

  VkPhysicalDeviceVulkan11Features vulkan11 =
      makeFeatures<VkPhysicalDeviceVulkan11Features>(&accelerationStructure, [](auto& f) {
        f.storageBuffer16BitAccess = VK_TRUE;
      });

  VkPhysicalDeviceVulkan12Features vulkan12 =
      makeFeatures<VkPhysicalDeviceVulkan12Features>(&vulkan11, [](auto& f) {
        f.timelineSemaphore   = VK_TRUE;
        f.bufferDeviceAddress = VK_TRUE;
      });

  VkPhysicalDeviceVulkan13Features vulkan13 =
      makeFeatures<VkPhysicalDeviceVulkan13Features>(&vulkan12, [](auto& f) {
        f.dynamicRendering = VK_TRUE;
      });

  // Head of the pNext chain (point VkDeviceCreateInfo::pNext or VkPhysicalDeviceFeatures2::pNext here)
  void* pNext() { return &vulkan13; }

  // Non-copyable to protect pNext chain
  SampleDeviceFeatures()                                       = default;
  SampleDeviceFeatures(const SampleDeviceFeatures&)            = delete;
  SampleDeviceFeatures& operator=(const SampleDeviceFeatures&) = delete;
};

// Non-copyable struct encapsulating VkDeviceCreateInfo with sample-specific extensions and features.
struct SampleDeviceCreateInfo
{
  static constexpr std::initializer_list baseExtensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
      VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME,
      VK_KHR_PRESENT_ID_EXTENSION_NAME,
      VK_KHR_PRESENT_WAIT_EXTENSION_NAME,
      VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
      VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
      VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
      VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME,
      VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME,
      VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
      VK_NV_MESH_SHADER_EXTENSION_NAME,
      VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_NV_COPY_MEMORY_INDIRECT_EXTENSION_NAME,
      VK_KHR_MAINTENANCE_7_EXTENSION_NAME,
  };

  const SampleQueues& queues;

  std::vector<const char*>             extensions;
  std::vector<float>                   queuePriorities;
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  SampleDeviceFeatures                 features;
  VkPhysicalDeviceFeatures2            features2;
  VkDeviceCreateInfo                   createInfo;

  SampleDeviceCreateInfo(const vko::Instance& instance,
                         VkPhysicalDevice     physicalDevice,
                         const SampleQueues&  queues_,
                         const NVSDK_NGX_FeatureDiscoveryInfo* ngxFeatureDiscovery = nullptr,
                         bool enableValidation = false)
      : queues(queues_)
      , extensions(baseExtensions.begin(), baseExtensions.end())
      , queuePriorities{1.0f, 1.0f, 1.0f, 1.0f}
      , queueCreateInfos{}
      , features{}
      , features2{.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                  .pNext    = features.pNext(),
                  .features = {}}
      , createInfo{}
  {
    if(enableValidation)
    {
      extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    }

    if(ngxFeatureDiscovery)
    {
      if(!vko::ngx::withRequiredDeviceExtensions(
             instance, physicalDevice, *ngxFeatureDiscovery,
             [&](std::span<const VkExtensionProperties> ngxExts) {
               for(const auto& ext : ngxExts)
               {
                 extensions.push_back(ext.extensionName);
               }
             }))
      {
        std::cout << "Warning: DLSS device extensions not available." << std::endl;
      }
    }

    instance.vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
    if(features.clusterAccelerationStructure.clusterAccelerationStructure != VK_TRUE)
    {
      throw std::runtime_error("Cluster Acceleration Structure feature is not supported by the physical device");
    }

    // Derive queue counts per family from the queue locations
    std::map<uint32_t, uint32_t> familyCounts;
    familyCounts[queues.primary.family]++;
    familyCounts[queues.asyncLoad.family]++;
    familyCounts[queues.compute.family]++;
    familyCounts[queues.transfer.family]++;

    for(auto [familyIndex, queueCount] : familyCounts)
    {
      queueCreateInfos.push_back(VkDeviceQueueCreateInfo{
          .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .pNext            = nullptr,
          .flags            = 0,
          .queueFamilyIndex = familyIndex,
          .queueCount       = queueCount,
          .pQueuePriorities = queuePriorities.data(),
      });
    }

    createInfo = VkDeviceCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &features2,
        .flags                   = 0,
        .queueCreateInfoCount    = uint32_t(queueCreateInfos.size()),
        .pQueueCreateInfos       = queueCreateInfos.data(),
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = uint32_t(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
        .pEnabledFeatures        = nullptr,
    };
  }

  operator const VkDeviceCreateInfo&() const { return createInfo; }

  // Non-copyable to protect internal pointers
  SampleDeviceCreateInfo(const SampleDeviceCreateInfo&)            = delete;
  SampleDeviceCreateInfo& operator=(const SampleDeviceCreateInfo&) = delete;
};

// Helper to select a suitable physical device
VkPhysicalDevice selectPhysicalDevice(const vko::Instance& instance,
                                      vko::glfw::PlatformSupport& platformSupport,
                                      const NVSDK_NGX_FeatureDiscoveryInfo* ngxFeatureDiscovery = nullptr);

// Helper to select suitable queues (finds best queue families and assigns indices)
SampleQueues selectQueues(const vko::Instance& instance, VkPhysicalDevice physicalDevice);

// Context that owns Vulkan objects (instance, device, queues)
struct SampleVulkanContext
{
  struct Queue
  {
    Queue(const vko::Device& device, QueueLocation location_)
        : location(location_)
        , queue(vko::get(device.vkGetDeviceQueue,
                         device,
                         location_.family,
                         location_.index))
    {
    }
    QueueLocation location;
    VkQueue       queue;
  };

  SampleVulkanContext(vko::GlobalCommands&          globalCommands_,
                      vko::Instance&&               instance_,
                      VkPhysicalDevice              physicalDevice_,
                      const SampleDeviceCreateInfo& deviceCreateInfo);

  ~SampleVulkanContext()
  {
    if(device.engaged())
    {
      vko::check(device.vkQueueWaitIdle(primary.queue));
      vko::check(device.vkQueueWaitIdle(compute.queue));
      vko::check(device.vkQueueWaitIdle(transfer.queue));
      vko::check(device.vkQueueWaitIdle(asyncLoad.queue));
    }
  }

  vko::GlobalCommands& globalCommands;
  vko::Instance        instance;
  VkPhysicalDevice     physicalDevice;
  vko::Device          device;
  vko::vma::Allocator  allocator;
  Queue                primary;
  Queue                compute;
  Queue                transfer;
  Queue                asyncLoad;

private:
  // Non-copyable
  SampleVulkanContext(const SampleVulkanContext&)            = delete;
  SampleVulkanContext& operator=(const SampleVulkanContext&) = delete;
};
