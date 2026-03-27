/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fileformats/nv_dds.h>
#include <fileformats/nv_ktx.h>
#include <fileformats/texture_formats.h>
#include <filesystem>
#include <optional>
#include <print>
#include <span>
#include <stb_image.h>
#include <vko/allocator.hpp>
#include <vko/bound_image.hpp>
#include <vko/formats.hpp>
#include <vko/staging_memory.hpp>
#include <vulkan/vulkan_core.h>

// Unique pointers to automatically free stb_image data
using UniqueStbiImage8  = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;
using UniqueStbiImage16 = std::unique_ptr<stbi_us, decltype(&stbi_image_free)>;

// Non-owning image data descriptor and pointer
struct ImageBase
{
  VkFormat                   format;
  VkExtent2D                 extent;
  std::span<const std::byte> data;
};

// Image descriptor and data
struct Image : ImageBase
{
  UniqueStbiImage8  image8  = {nullptr, &stbi_image_free};
  UniqueStbiImage16 image16 = {nullptr, &stbi_image_free};
  nv_dds::Image     dds;
  nv_ktx::KTXImage  ktx;
};

// RAII Vulkan image object
class ImageVk
{
public:
  template <vko::device_and_commands DeviceAndCommands>
  ImageVk(const DeviceAndCommands& device,
          vko::vma::Allocator&     allocator,
          const VkImageCreateInfo& createInfo,
          VkMemoryPropertyFlags memoryPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
      : m_image(device, createInfo, memoryPropertyFlags, allocator)
  {
  }

  ImageVk(const ImageVk&)                      = delete;
  ImageVk& operator=(const ImageVk&)           = delete;
  ImageVk(ImageVk&& other) noexcept            = default;
  ImageVk& operator=(ImageVk&& other) noexcept = default;

  operator VkImage() const { return m_image; }

private:
  vko::BoundImage<vko::vma::Allocator> m_image;
};

inline std::optional<Image> createImageDDS(const std::filesystem::path& path, bool srgb)
{
  Image result;

  nv_dds::ReadSettings  settings{};
  nv_dds::ErrorWithText readResult =
      result.dds.readFromFile(path.string().c_str(), settings);
  if(readResult.has_value())
  {
    std::println(stderr, "Failed to read {} using nv_dds: {}", path.string(),
                 readResult.value());
    return std::nullopt;
  }

  // Check for unsupported features
  if(result.dds.getDepth(0) > 1)
  {
    std::println(stderr, "This DDS image had a depth of {}, but createImageDDS() cannot handle volume textures.",
                 result.dds.getDepth(0));
    return std::nullopt;
  }
  if(result.dds.getNumFaces() > 1)
  {
    std::println(stderr, "This DDS image had {} faces, but createImageDDS() cannot handle cubemaps.",
                 result.dds.getNumFaces());
    return std::nullopt;
  }
  if(result.dds.getNumLayers() > 1)
  {
    std::println(stderr, "This DDS image had {} array elements, but createImageDDS() cannot handle array textures.",
                 result.dds.getNumLayers());
    return std::nullopt;
  }

  // Set extent and format
  result.extent.width  = result.dds.getWidth(0);
  result.extent.height = result.dds.getHeight(0);
  result.format =
      static_cast<VkFormat>(texture_formats::dxgiToVulkan(result.dds.dxgiFormat));
  result.format = static_cast<VkFormat>(
      texture_formats::tryForceDXGIFormatTransferFunction(result.dds.dxgiFormat, srgb));

  if(VK_FORMAT_UNDEFINED == result.format)
  {
    std::println(stderr, "Could not determine a VkFormat for DXGI format {} ({}).",
                 result.dds.dxgiFormat,
                 texture_formats::getDXGIFormatName(result.dds.dxgiFormat));
    return std::nullopt;
  }

  // Use the first mip level's data
  const std::vector<char>& mipData = result.dds.subresource(0, 0, 0).data;
  result.data =
      std::span{reinterpret_cast<const std::byte*>(mipData.data()), mipData.size()};

  return result;
}

inline std::optional<Image> createImageKTX(const std::filesystem::path& path, bool srgb)
{
  Image result;

  const nv_ktx::ReadSettings ktxReadSettings;
  nv_ktx::ErrorWithText      maybeError =
      result.ktx.readFromFile(path.string().c_str(), ktxReadSettings);
  if(maybeError.has_value())
  {
    std::println(stderr, "Failed to read {} using nv_ktx: {}", path.string(), *maybeError);
    return std::nullopt;
  }

  // Check for unsupported features
  if(result.ktx.mip_0_depth > 1)
  {
    std::println(stderr, "This KTX image had a depth of {}, but createImageKTX() cannot handle volume textures.",
                 result.ktx.mip_0_depth);
    return std::nullopt;
  }
  if(result.ktx.num_faces > 1)
  {
    std::println(stderr, "This KTX image had {} faces, but createImageKTX() cannot handle cubemaps.",
                 result.ktx.num_faces);
    return std::nullopt;
  }
  if(result.ktx.num_layers_possibly_0 > 1)
  {
    std::println(stderr, "This KTX image had {} array elements, but createImageKTX() cannot handle array textures.",
                 result.ktx.num_layers_possibly_0);
    return std::nullopt;
  }

  // Set extent and format
  result.extent.width  = result.ktx.mip_0_width;
  result.extent.height = result.ktx.mip_0_height;
  result.format =
      texture_formats::tryForceVkFormatTransferFunction(result.ktx.format, srgb);

  // Use the first mip level's data
  const std::vector<char>& mipData = result.ktx.subresource(0, 0, 0);
  result.data =
      std::span{reinterpret_cast<const std::byte*>(mipData.data()), mipData.size()};

  return result;
}

// Loads an decompresses an image from disk into system memory using stb_image
inline std::optional<Image> createImageSTB(const std::filesystem::path& path, bool srgb)
{
  // Read the header once to check how many channels it has. We can't trivially use RGB/VK_FORMAT_R8G8B8_UNORM and
  // need to set req_comp=4 in such cases.
  int w = 0, h = 0, comp = 0;
  if(!stbi_info(path.string().c_str(), &w, &h, &comp))
  {
    std::println(stderr, "Failed to read {}", path.string());
    return std::nullopt;
  }

  // Read the header again to check if it has 16 bit data, e.g. for a heightmap.
  bool   is_16Bit = stbi_is_16_bit(path.string().c_str());
  int    req_comp = comp == 1 ? 1 : 4;
  size_t bytes_per_pixel =
      (is_16Bit ? sizeof(stbi_us) : sizeof(stbi_uc)) * size_t(req_comp);

  Image result;
  result.extent = {uint32_t(w), uint32_t(h)};
  if(is_16Bit)
  {
    result.image16 =
        UniqueStbiImage16(stbi_load_16(path.string().c_str(), &w, &h, &comp, req_comp),
                          &stbi_image_free);
    result.data = std::span{reinterpret_cast<const std::byte*>(result.image16.get()),
                            size_t(bytes_per_pixel * size_t(w) * size_t(h))};
    result.format = req_comp == 1 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R16G16B16A16_UNORM;
  }
  else
  {
    result.image8 =
        UniqueStbiImage8(stbi_load(path.string().c_str(), &w, &h, &comp, req_comp),
                         &stbi_image_free);
    result.data = std::span{reinterpret_cast<const std::byte*>(result.image8.get()),
                            size_t(bytes_per_pixel * size_t(w) * size_t(h))};
    result.format = req_comp == 1 ?
                        VK_FORMAT_R8_UNORM :
                        (srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM);
  }
  return result;
}

inline std::optional<Image> createImage(const std::filesystem::path& path, bool srgb)
{
  if(path.extension() == ".dds")
  {
    // Special case for DDS images
    return createImageDDS(path, srgb);
  }
  else if(path.extension() == ".ktx" || path.extension() == ".ktx2")
  {
    // Special case for KTX images
    return createImageKTX(path, srgb);
  }
  else
  {
    // stb for everything else
    return createImageSTB(path, srgb);
  }
}

// Texture wrapper combining image, view, and sampler
class TextureVk
{
public:
  template <vko::device_and_commands DeviceAndCommands>
  TextureVk(const DeviceAndCommands&               device,
            vko::BoundImage<vko::vma::Allocator>&& image,
            const VkImageViewCreateInfo&           imageViewCreateInfo,
            const VkSamplerCreateInfo&             samplerCreateInfo)
      : m_image(std::move(image))
      , m_view(device, imageViewCreateInfo)
      , m_sampler(device, samplerCreateInfo)
  {
  }

  VkDescriptorImageInfo descriptor() const
  {
    return {.sampler = m_sampler, .imageView = m_view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  }

  VkImage     image() const { return m_image; }
  VkImageView view() const { return m_view; }
  VkSampler   sampler() const { return m_sampler; }

private:
  vko::BoundImage<vko::vma::Allocator> m_image;
  vko::ImageView                       m_view;
  vko::Sampler                         m_sampler;
};

// Creates a texture from an ImageBase, uploading data via staging buffer
// Note: generateMipmaps is ignored for now - all images have single mip level
template <vko::device_and_commands DeviceAndCommands, vko::staging_stream StagingStream>
inline TextureVk createTextureVk(const DeviceAndCommands& device,
                                 vko::vma::Allocator&     allocator,
                                 StagingStream&           staging,
                                 const ImageBase&         image,
                                 bool /*generateMipmaps*/)
{
  uint32_t mipLevels = 1;  // TODO: Add mipmap generation support

  VkImageCreateInfo imageCreateInfo{
      .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext       = nullptr,
      .flags       = 0,
      .imageType   = VK_IMAGE_TYPE_2D,
      .format      = image.format,
      .extent      = {image.extent.width, image.extent.height, 1},
      .mipLevels   = mipLevels,
      .arrayLayers = 1,
      .samples     = VK_SAMPLE_COUNT_1_BIT,
      .tiling      = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices   = nullptr,
      .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  vko::BoundImage boundImage(device, imageCreateInfo,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, allocator);

  // Transition to transfer dst layout
  VkImageMemoryBarrier toTransfer{
      .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .pNext               = nullptr,
      .srcAccessMask       = 0,
      .dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image               = boundImage,
      .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  device.vkCmdPipelineBarrier(staging.commandBuffer(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                              nullptr, 1, &toTransfer);

  // Upload via staging buffer
  vko::upload(staging, device, image.data,
              vko::ImageRegion(boundImage, VK_IMAGE_ASPECT_COLOR_BIT),
              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, image.format);

  // Transition to shader read layout
  VkImageMemoryBarrier toShader{
      .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .pNext               = nullptr,
      .srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask       = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image               = boundImage,
      .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  device.vkCmdPipelineBarrier(staging.commandBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                              nullptr, 0, nullptr, 1, &toShader);

  VkImageViewCreateInfo imageViewCreateInfo{
      .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .pNext            = nullptr,
      .flags            = 0,
      .image            = boundImage,
      .viewType         = VK_IMAGE_VIEW_TYPE_2D,
      .format           = image.format,
      .components       = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                           VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A},
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1},
  };

  VkSamplerCreateInfo samplerCreateInfo{
      .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .pNext                   = nullptr,
      .flags                   = 0,
      .magFilter               = VK_FILTER_LINEAR,
      .minFilter               = VK_FILTER_LINEAR,
      .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .mipLodBias              = 0.0f,
      .anisotropyEnable        = VK_FALSE,
      .maxAnisotropy           = 0.0f,
      .compareEnable           = VK_FALSE,
      .compareOp               = VK_COMPARE_OP_NEVER,
      .minLod                  = 0.0f,
      .maxLod                  = float(mipLevels),
      .borderColor             = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
      .unnormalizedCoordinates = VK_FALSE,
  };

  return TextureVk(device, std::move(boundImage), imageViewCreateInfo, samplerCreateInfo);
}
