#include "sample_vulkan_context.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vko/functions.hpp>

VkPhysicalDevice selectPhysicalDevice(const vko::Instance& instance,
                                      vko::glfw::PlatformSupport& platformSupport,
                                      const NVSDK_NGX_FeatureDiscoveryInfo* ngxFeatureDiscovery)
{
  auto physicalDevices = vko::toVector(instance.vkEnumeratePhysicalDevices, instance);

  if(physicalDevices.size() > 1)
  {
    std::cout << "Found " << physicalDevices.size() << " physical devices:" << std::endl;
    for(size_t i = 0; i < physicalDevices.size(); ++i)
    {
      VkPhysicalDeviceProperties props;
      instance.vkGetPhysicalDeviceProperties(physicalDevices[i], &props);
      std::cout << "  [" << i << "] " << props.deviceName << std::endl;
    }
  }

  // Helper to check if DLSS device extensions are supported
  auto supportsDLSS = [&](VkPhysicalDevice physicalDevice) -> bool {
    if(!ngxFeatureDiscovery)
      return false;

    bool hasDLSS = false;
    vko::ngx::withRequiredDeviceExtensions(instance, physicalDevice, *ngxFeatureDiscovery,
                                           [&](std::span<const VkExtensionProperties> exts) {
                                             hasDLSS = !exts.empty();
                                           });
    return hasDLSS;
  };

  // Rank devices based on their capabilities
  struct DeviceScore
  {
    VkPhysicalDevice device;
    int              score;
    std::string      name;
  };

  std::vector<DeviceScore> scoredDevices;

  for(VkPhysicalDevice physicalDevice : physicalDevices)
  {
    VkPhysicalDeviceProperties props;
    instance.vkGetPhysicalDeviceProperties(physicalDevice, &props);

    int score = 0;

    // Score based on device type
    if(props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      score += 1000;
    else if(props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
      score += 100;
    else
      score += 10;

    // Score based on DLSS support
    if(supportsDLSS(physicalDevice))
      score += 500;

    // Check presentation support (required)
    uint32_t queueFamilyIndex = 0;
    if(!vko::glfw::physicalDevicePresentationSupport(instance, platformSupport,
                                                     physicalDevice, queueFamilyIndex))
    {
      score = 0;  // Skip devices without presentation support
    }

    scoredDevices.push_back({physicalDevice, score, props.deviceName});
  }

  if(scoredDevices.empty())
  {
    throw std::runtime_error("No suitable physical device found with presentation support");
  }

  // Sort by score (highest first)
  std::ranges::sort(scoredDevices, [](const DeviceScore& a, const DeviceScore& b) {
    return a.score > b.score;
  });

  VkPhysicalDevice selectedDevice = scoredDevices[0].device;

  if(physicalDevices.size() > 1)
  {
    std::cout << "Selected: " << scoredDevices[0].name << std::endl;
  }

  return selectedDevice;
}

SampleQueues selectQueues(const vko::Instance& instance, VkPhysicalDevice physicalDevice)
{
  auto queueProperties =
      vko::toVector(instance.vkGetPhysicalDeviceQueueFamilyProperties, physicalDevice);

  // 1. Find graphics queue family (GRAPHICS | COMPUTE | TRANSFER)
  auto graphicsIt =
      std::ranges::find_if(queueProperties, [](const VkQueueFamilyProperties& properties) {
        VkQueueFlags requiredFlags =
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        return (properties.queueFlags & requiredFlags) == requiredFlags;
      });

  if(graphicsIt == queueProperties.end())
  {
    throw std::runtime_error("No graphics queue family found");
  }

  uint32_t graphicsFamily = uint32_t(std::distance(queueProperties.begin(), graphicsIt));

  // 2. Try to find a dedicated compute queue family (COMPUTE but not GRAPHICS)
  auto computeIt =
      std::ranges::find_if(queueProperties, [](const VkQueueFamilyProperties& properties) {
        return (properties.queueFlags & VK_QUEUE_COMPUTE_BIT)
               && !(properties.queueFlags & VK_QUEUE_GRAPHICS_BIT);
      });

  uint32_t computeFamily =
      (computeIt != queueProperties.end()) ?
          uint32_t(std::distance(queueProperties.begin(), computeIt)) :
          graphicsFamily;

  // 3. Try to find a dedicated transfer queue family (TRANSFER but not GRAPHICS or COMPUTE)
  auto transferIt =
      std::ranges::find_if(queueProperties, [](const VkQueueFamilyProperties& properties) {
        return (properties.queueFlags & VK_QUEUE_TRANSFER_BIT)
               && !(properties.queueFlags & VK_QUEUE_GRAPHICS_BIT)
               && !(properties.queueFlags & VK_QUEUE_COMPUTE_BIT);
      });

  uint32_t transferFamily =
      (transferIt != queueProperties.end()) ?
          uint32_t(std::distance(queueProperties.begin(), transferIt)) :
          graphicsFamily;

  // Assign queue indices while counting queues per family
  std::map<uint32_t, uint32_t> familyCounts;
  return SampleQueues{
      .primary   = {graphicsFamily, familyCounts[graphicsFamily]++},
      .asyncLoad = {graphicsFamily, familyCounts[graphicsFamily]++},
      .compute   = {computeFamily, familyCounts[computeFamily]++},
      .transfer  = {transferFamily, familyCounts[transferFamily]++},
  };
}

SampleVulkanContext::SampleVulkanContext(vko::GlobalCommands& globalCommands_,
                                         vko::Instance&&      instance_,
                                         VkPhysicalDevice     physicalDevice_,
                                         const SampleDeviceCreateInfo& deviceCreateInfo)
    : globalCommands(globalCommands_)
    , instance(std::move(instance_))
    , physicalDevice(physicalDevice_)
    , device(instance, physicalDevice, deviceCreateInfo)
    , allocator(globalCommands, instance, physicalDevice, device, SampleVulkanVersion, VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT)
    , primary(device, deviceCreateInfo.queues.primary)
    , compute(device, deviceCreateInfo.queues.compute)
    , transfer(device, deviceCreateInfo.queues.transfer)
    , asyncLoad(device, deviceCreateInfo.queues.asyncLoad)
{
}
