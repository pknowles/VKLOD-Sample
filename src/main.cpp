/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// See README.md for docs

#include <GLFW/glfw3.h>
#include <ImGuiFileDialog.h>
#include <acceleration_structures.hpp>
#include <algorithm>
#include <args.hxx>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <camera.hpp>
#include <cstddef>
#include <cstdlib>
#include <debug_range_summary.hpp>
#include <debugbreak.h>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <imgui.h>
#include <imgui_docking_helper.hpp>
#include <implot.h>
#include <iostream>
#include <lod_streaming_scene.hpp>
#include <memory>
#include <nvclusterlod/nvclusterlod_hierarchy.h>
#include <nvpro_core_legacy/imgui/imgui_icon.h>
#include <platform_utils.hpp>
#include <renderer_common.hpp>
#include <renderer_rasterize.hpp>
#include <renderer_raytrace.hpp>
#include <sample_camera_paths.hpp>
#include <sample_font_extra.hpp>
#include <sample_glsl_compiler.hpp>
#include <sample_profiler.hpp>
#include <sample_progress.hpp>
#include <sample_raytracing_objects.hpp>
#include <sample_vulkan_context.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <stb_image_write.h>
#include <variant>
#include <vector>
#include <vko/functions.hpp>
#include <vko/glfw_objects.hpp>
#include <vko/handles.hpp>
#include <vko/imgui_objects.hpp>
#include <vko/implot_objects.hpp>
#include <vko/pnext_chain.hpp>
#include <vko/shortcuts.hpp>
#include <vko/swapchain.hpp>
#include <vulkan/vulkan_core.h>

#ifdef __linux__
#include <csignal>
#endif

namespace fs = std::filesystem;

#ifdef __linux__
static bool g_shouldShutdown = false;

void signalHandler(int signal)
{
  if(signal == SIGTERM)
  {
    g_shouldShutdown = true;
  }
}
#endif

// The renderer owns the streaming pool, which is big. GPUs without much VRAM
// might run out of memory with two of these alive at once, so this defaults to
// off. The result is a stall, waiting for all GPU commands to complete before
// recreating the renderer.
static constexpr bool ASYNC_RECREATE_RENDERER = false;

static std::optional<std::filesystem::path> g_droppedFile;
static bool                                 m_needsResize             = false;
static VkExtent2D                           g_glfwLastFramebufferSize = {0, 0};

void dropCallback(GLFWwindow* /*window*/, int count, const char** paths)
{
  for(int i = 0; i < count; ++i)
  {
    std::filesystem::path path(paths[i]);
    auto                  ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if(ext == ".gltf" || ext == ".glb")
    {
      g_droppedFile = path;
      std::print("Dropped file: {}\n", path.string());
      break;  // Only handle the first gltf/glb file
    }
  }
}

void resizeRequested(GLFWwindow* /*window*/, int width, int height)
{
  g_glfwLastFramebufferSize = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  m_needsResize = true;
}

fs::path getRendercachePath(const fs::path& cacheDir, const fs::path& gltfPath)
{
  return cacheDir / ("rendercache_" + gltfPath.filename().string() + ".dat");
}

template <class T, size_t N>
class PlotSamples : public std::array<T, N>
{
public:
  PlotSamples()
  {
    push(T(0));
  }  // implot only renders if there are at least two samples
  T&   first() { return (*this)[pivot()]; }
  T&   last() { return (*this)[(m_size + N - 1) % N]; }
  void push(T v)
  {
    (*this)[m_size % N] = v;
    ++m_size;
  }
  size_t pivot() const { return m_size < N ? 0 : m_size % N; }
  size_t size() const { return std::min(m_size, N); }

private:
  size_t m_size = 0;
};

struct Config
{
  int  rendererIndex              = 0;
  int  streamingBufferSizeMB      = 1024;  // 1GB
  int  streamingMaxResidentGroups = 1 << 13;
  int  maxFramesInFlight          = 1;  // Limit frames queued for presentation
  bool invalidateNextLoadRendercache = false;
  bool streamingStress = false;  // random camera position each frame
  bool greedyUnload = false;  // evict immediately, instead of waiting for memory pressure
  bool                            vSync = true;  // V-sync (FIFO present mode)
  float                           generatedSceneDetailScale = 1.0f;
  SceneLodConfig                  sceneLodConfig;
  vko::shared_obj<RendererConfig> common;
  vko::shared_obj<RaytraceConfig> raytracing;  // shared to persist between scene changes
};

// Variant for renderer types - each owns its streaming
using RendererVariant = std::variant<RaytraceRenderer, RasterizeRenderer>;

// Renderer names for UI
constexpr std::array<const char*, 2> RENDERER_NAMES = {{"Raytrace Clusters", "Rasterize Clusters"}};

SceneFile loadCachedScene(const fs::path       gltfPath,
                          const fs::path&      cachePath,
                          const Config&        config,
                          bool                 invalidateCache,
                          const vko::Device&   device,
                          vko::vma::Allocator& allocator,
                          vkobj::Staging&      staging,
                          SampleGlslCompiler&  glslCompiler,
                          TaskProgress*        progress = nullptr)
{
  // The initial scene load is synchronous. Easier to make a dummy object than
  // handle null everywhere.
  std::optional<TaskProgress>    progressOptional;
  std::optional<ProgressPrinter> progressPrinter;
  if(!progress)
  {
    progressOptional.emplace();
    progressPrinter.emplace(*progressOptional);
    progress = &progressOptional.value();
  }

  // Try to load from cache
  std::optional<SceneFile> result;

  if(!gltfPath.empty())
  {
    // Load from the cache or a gltf file
    if(!invalidateCache)
      result = makeSceneFromCache(gltfPath, cachePath);
    if(!result)
    {
      progress->defineSubtasks({{"generate LOD", 1.0f}});
      result = makeSceneFromGltf(gltfPath, cachePath, config.sceneLodConfig, *progress);
    }
  }
  else
  {
    // Load from the cache or procedurally generate a scene
    fs::path generatedCachePath = cachePath.parent_path() / ("rendercache_generated.dat");
    if(!invalidateCache)
      result = makeSceneFromCache(gltfPath, generatedCachePath);
    if(!result)
    {
      fprintf(stderr, "Procedurally generating some high poly meshes.\nThis could take ~60 seconds.\nAlternatively, run with --mesh path/to/scene.gltf\n");
      progress->defineSubtasks({{"generate scene", 1.25f}, {"generate LOD", 0.75f}});
      GeneratedScene generatedScene =
          makeTerrainAndRocksScene(device, glslCompiler, allocator, staging,
                                   config.generatedSceneDetailScale, *progress);
      result = makeSceneFromGenerated(std::move(generatedScene), generatedCachePath,
                                      config.sceneLodConfig, *progress);
    }
  }
  return std::move(*result);
}

// Scene container to guarantee lifetime and destruction order
struct RenderableScene
{
  RenderableScene(const fs::path&       gltfPath,
                  const fs::path&       cachePath,
                  const Config&         config,
                  bool                  invalidateCache,
                  const vko::Device&    device,
                  vko::vma::Allocator&  allocator,
                  vkobj::Staging&       staging,
                  vkobj::TimelineQueue& initQueue,
                  SampleGlslCompiler&   glslCompiler)
      : mapping{loadCachedScene(gltfPath, cachePath, config, invalidateCache, device, allocator, staging, glslCompiler)}
      , view{*mapping.data}
      , vk{staging, device, allocator, initQueue, view}
  {
  }
  RenderableScene(SceneFile&&           mapping,
                  const vko::Device&    device,
                  vko::vma::Allocator&  allocator,
                  vkobj::Staging&       staging,
                  vkobj::TimelineQueue& initQueue)
      : mapping{std::move(mapping)}
      , view{*mapping.data}
      , vk{staging, device, allocator, initQueue, view}
  {
  }
  SceneFile mapping;  // file mapping objects
  Scene     view;     // pointers into the file mapping
  SceneVK   vk;       // GPU data, mostly uploaded from SceneMapping
};

// Helper to get streaming from any renderer variant
template <class Renderer>
decltype(auto) getStreaming(Renderer&& renderer)
{
  return std::visit(
      [](auto&& r) -> decltype(auto) {
        return std::forward<decltype(r)>(r).streaming();
      },
      std::forward<Renderer>(renderer));
}

SampleGlslCompiler makeGlslCompiler()
{
  return SampleGlslCompiler{INSTALL_SUBDIRECTORY,  // search after cmake --install
                            "shaders",  // this project's shaders (CWD = source dir)
                            "nvpro_core_legacy",  // nvpro_core shaders (CWD = source dir)
                            SHADER_DIR_RELPATH_FROM_BINARY,  // this project's shaders (CWD = build dir)
                            NVPRO_LEGACY_RELPATH_FROM_BINARY};  // nvpro_core shaders (CWD = build dir)
}

class Sample
{
public:
  auto& stagingCommandBuffer() { return m_context.staging.commandBuffer(); }

  Sample(SampleVulkanContext&         vkContext,
         vko::shared_obj<Config>      config,
         glm::uvec2                   vpSize,
         const std::filesystem::path& gltfPath,
         const std::filesystem::path& cacheDir,
         bool                         dlssAvailable)
      : m_config(config)
      , m_dlssAvailable(dlssAvailable)
      , m_queueStates(vkContext)
      , m_context(vkContext.instance,
                  vkContext.device,
                  vkContext.physicalDevice,
                  vkContext.allocator,
                  m_queueStates.primary)
      , m_glslCompiler(makeGlslCompiler())
      , m_framebuffer(m_context.device.get(),
                      m_context.allocator.get(),
                      m_context.commandPool,
                      m_queueStates.primary,
                      m_glslCompiler,
                      vpSize)
      , m_scene{gltfPath,
                getRendercachePath(cacheDir, gltfPath),
                *m_config,
                false,
                m_context.device.get(),
                m_context.allocator.get(),
                m_context.staging,
                m_queueStates.primary,
                m_glslCompiler}
      , m_rendererCommon{m_context.device.get(), m_context.allocator.get(),
                         m_config->common}
      , m_cameraPaths(m_camera)
      , m_cacheDir(cacheDir)
  {
    m_renderer = createRenderer(m_config->rendererIndex);
    onSceneLoaded();
  }

  ~Sample()
  {
    // Force-stop the background thread by throwing an exception the next time
    // it makes progress
    if(m_sceneLoadProgress)
      m_sceneLoadProgress->cancel();
  }

  void load(const std::filesystem::path& filename)
  {
    if(m_sceneFuture.valid())
    {
      std::print(stderr, "A scene is already loading");
      return;
    }

    // Load the new scene. This may take a while
    m_sceneLoadError.clear();  // clear any previous error
    m_sceneLoadProgress = std::make_unique<TaskProgress>();
    m_sceneFuture = std::async(std::launch::async, [this, filename, config = m_config,
                                                    invalidateRendercache =
                                                        m_config->invalidateNextLoadRendercache]() {
      // Need a context with its own staging memory for async loading
      vkobj::Context     asyncLoadContext(m_context.instance.get(),
                                          m_context.device.get(), m_context.physicalDevice,
                                          m_context.allocator.get(), m_queueStates.asyncLoad);
      SampleGlslCompiler compiler = makeGlslCompiler();
      SceneFile          file =
          loadCachedScene(filename, getRendercachePath(m_cacheDir, filename),
                          *config, invalidateRendercache, m_context.device.get(),
                          m_context.allocator.get(), asyncLoadContext.staging,
                          compiler, m_sceneLoadProgress.get());
      return std::make_optional(std::move(file));
    });
    m_config->invalidateNextLoadRendercache = false;
  }

  void onSceneLoaded()
  {
    // Initial camera view for the new scene
    float sceneSize =
        glm::length(m_scene.view.worldAABB.max - m_scene.view.worldAABB.min);
    float sceneSizeDec  = powf(10.0f, ceilf(log10f(sceneSize)));
    m_camera.clipPlanes = {sceneSizeDec * 0.001f, sceneSizeDec * 20.0f};
    m_camera.setLookat(m_scene.view.worldAABB.center() + glm::vec3{-0.2F, 0.4F, 0.8F} * sceneSize,
                       m_scene.view.worldAABB.center(), {0.0F, 1.0F, 0.0F});
    m_camera.verticalFov = glm::radians(80.0f);

    // Initialize fog parameters based on the scene size
    AABB  sceneAABB = m_scene.view.worldAABB;
    float sceneAspectRatio =
        length(glm::vec2(sceneAABB.size())) / sceneAABB.size().y;
    m_config->raytracing->shaders.fogHeightOffset =
        sceneAABB.min.y + sceneAABB.size().y * 0.7f;
    m_config->raytracing->shaders.fogDensity =
        (2.0f / sceneAABB.size().y) * glm::smoothstep(4.0f, 8.0f, sceneAspectRatio);  // dense for high aspect scenes
  }

  std::unique_ptr<RendererVariant> createRenderer(int index)
  {
    RenderInitParams initParams{
        .context       = m_context,
        .glslCompiler  = m_glslCompiler,
        .common        = m_rendererCommon,
        .scene         = m_scene.view,
        .sceneVk       = m_scene.vk,
        .framebuffer   = *m_framebuffer,
        .dlssAvailable = m_dlssAvailable,
        .queue         = m_queueStates.primary,
        .initPool      = m_context.commandPool,
        .initQueue     = m_queueStates.primary,
        .transferQueue = m_queueStates.transfer,
        .streamingBufferSize = VkDeviceSize(m_config->streamingBufferSizeMB) << 20,
        .streamingMaxResidentGroups = uint32_t(m_config->streamingMaxResidentGroups),
        .streamingGreedyUnload = m_config->greedyUnload,
        .profiler      = m_profiler,
    };
    if(index == 0)
    {
      return std::make_unique<RendererVariant>(std::in_place_type<RaytraceRenderer>,
                                               initParams, m_config->raytracing);
    }
    else
    {
      return std::make_unique<RendererVariant>(std::in_place_type<RasterizeRenderer>, initParams);
    }
  }

  void renderUI(GLFWwindow*           window,
                vkobj::SemaphoreValue lastRenderFinishedSemaphore,
                const ExtraFonts&     extraFonts)
  {

    // Minimal transparent overlay with triangle count and control icons,
    // position at the top-right corner
    ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(displaySize.x - 10, 10), ImGuiCond_Always,
                            ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.3f);
    vko::imgui::window("Cluster Count", nullptr,
                       ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize
                           | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar
                           | ImGuiWindowFlags_NoCollapse)([&] {
      {
        auto styleGuard = vko::imgui::styleColor(
            ImGuiCol_Text, ImVec4(118.0f / 255.0f, 185.0f / 255.0f, 0.0f / 255.0f, 1.0f));
        auto fontGuard = vko::imgui::font(extraFonts.large);
        std::visit([](auto& r) { r.uiOverlay(); }, *m_renderer);
        auto memoryUsed =
            std::visit([](auto& r) { return r.deviceMemoryUsage(); }, *m_renderer)
            + m_scene.vk.deviceMemoryUsage()
            + getStreaming(*m_renderer).pool().bytesAllocated();  // Excludes remaining reserved streaming memory, DLSS-RR and GBuffer
        ImGui::Text("Memory: %s", formatBytes(memoryUsed).c_str());
        if(auto frameTime = m_profiler.stats("Frame"))
        {
          ImGui::Text("Time: %.1fms", frameTime->avgNs / 1e6);
        }
      }

      ImGui::Spacing();

      // Display progress bar and the log if a scene is loading
      if(m_sceneLoadProgress)
      {
        // Create a top-center overlay window for loading progress
        ImGui::SetNextWindowPos(ImVec2(displaySize.x * 0.5f, 20),
                                ImGuiCond_Always, ImVec2(0.5f, 0.0f));
        ImGui::SetNextWindowBgAlpha(0.8f);
        vko::imgui::window("Scene Loading", nullptr,
                           ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize
                               | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar
                               | ImGuiWindowFlags_NoCollapse)([&] {
          ImGui::Text("Loading scene...");
          ImGui::Spacing();
          auto progress = m_sceneLoadProgress->progress();
          ImGui::ProgressBar(progress.ratio, ImVec2(800, 0));
          if(progress.remaining)
          {
            ImGui::Text("%s -> %s",
                        std::format("{:%M:%S}", std::chrono::floor<std::chrono::seconds>(
                                                    progress.elapsed))
                            .c_str(),
                        std::format("{:%M:%S}", std::chrono::floor<std::chrono::seconds>(
                                                    progress.remaining.value()))
                            .c_str());
          }
          ImGui::Spacing();
          ImGui::Text("%s", progress.currentWork().c_str());
          if(ImGui::Button("Cancel"))
          {
            // Throw an exception on the scene loading thread the next time it
            // updates progress.
            m_sceneLoadProgress->cancel();
          }
        });
      }

      // Settings window toggle
      if(iconButton(extraFonts.largeIcon, OpenIconic::icon_cog, m_showSettings,
                    "Settings: ON", "Settings: OFF"))
      {
        m_showSettings = !m_showSettings;
      }

      // Profiler window toggle
      if(iconButton(extraFonts.largeIcon, OpenIconic::icon_bar_chart,
                    m_showProfiler, "Profiler: ON", "Profiler: OFF"))
      {
        m_showProfiler = !m_showProfiler;
      }

      // V-Sync toggle (main loop applies the change)
      if(iconButton(extraFonts.largeIcon, OpenIconic::icon_monitor, m_config->vSync,
                    "V-Sync: ON (Ctrl+Shift+V)", "V-Sync: OFF (Ctrl+Shift+V)"))
      {
        m_config->vSync = !m_config->vSync;
      }

      // Camera paths toggle
      if(iconButton(extraFonts.largeIcon, OpenIconic::icon_video,
                    m_cameraPaths->isWindowVisible(), "Camera Paths: ON", "Camera Paths: OFF"))
      {
        m_cameraPaths->toggleWindow();
      }

      // Handle keyboard shortcut to exit
      if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
      {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }
      // V-Sync keyboard shortcut (main loop applies the change)
      if(ImGui::IsKeyPressed(ImGuiKey_V) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl)
         && ImGui::IsKeyDown(ImGuiKey_LeftShift))
      {
        m_config->vSync = !m_config->vSync;
      }
    });

    // Settings menu with a few common settings at the top and the rest hidden
    // in collapsible sections
    vko::imgui::window("Settings", &m_showSettings)([&] {
      //  Max Pixel Error
      if(ImGui::SliderFloat("Max. Pixel Error",
                            &m_config->common->lodTargetPixelError, 0.5f, 1000.0f))
      {
        m_rendererCommon.m_frameAccumIndex = 0;
      }
      if(ImGui::IsItemHovered())
        ImGui::SetTooltip("Targets LOD selection where the screen space geometric error is below this value.");

      // Button to disable LOD changes
      ImGui::Checkbox("Lock LOD", &m_config->common->lockLodCamera);
      if(ImGui::IsItemHovered())
        ImGui::SetTooltip("Keeps the LOD at the same detail, ignoring further camera movement.");

      // Renderer-specific UI
      {
        bool needRecreate = false, resetFrameAccumulation = false;
        std::visit(
            [&](auto& r) { r.uiInline(needRecreate, resetFrameAccumulation); }, *m_renderer);
        if(needRecreate)
        {
          m_garbage.push(Garbage{moveAny(std::move(m_renderer)), lastRenderFinishedSemaphore});
          if constexpr(!ASYNC_RECREATE_RENDERER)
          {
            waitAndEmptyGarbage(m_garbage, m_context.device.get());
          }
          m_renderer = createRenderer(m_config->rendererIndex);
        }
        if(resetFrameAccumulation)
          m_rendererCommon.m_frameAccumIndex = 0;
      }

      ImGui::Spacing();
      vko::imgui::disabled(m_sceneFuture.valid())([&] {  // disable loading buttons when already loading
        // Provide a procedural scene for a quick demo without large downloads
        if(m_scene.mapping.path.filename() == "bunny.gltf")
        {
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.82f, 1.0f, 0.36f, 1.0f));
          ImGui::TextWrapped("The bunny mesh is a simple placeholder. You can generate a more complex scene to better demonstrate LOD and streaming. It takes about a minute the first time.");
          ImGui::PopStyleColor();

          ImGui::SliderFloat("Detail Scale",
                             &m_config->generatedSceneDetailScale, 0.1f, 4.0f);
          if(ImGui::IsItemHovered())
            ImGui::SetTooltip("Scales the triangle count of all meshes. Ignored if a render cache exists - rebuild in 'LOD Processing'.");
          if(ImGui::Button("Generate Procedural Scene"))
          {
            load(fs::path{});  // empty path implies generated scene
          }
        }

        if(ImGui::Button("Load Scene"))
        {
          IGFD::FileDialogConfig config;
          config.path  = ".";
          config.flags = ImGuiFileDialogFlags_Modal;
          ImGuiFileDialog::Instance()->OpenDialog("LoadSceneDialog", "Choose glTF Scene",
                                                  ".gltf,.glb", config);
        }
      });  // loading buttons section

      // File dialog display (must be outside disabled section)
      if(ImGuiFileDialog::Instance()->Display("LoadSceneDialog", ImGuiWindowFlags_None,
                                              ImVec2(800, 400)))
      {
        if(ImGuiFileDialog::Instance()->IsOk())
        {
          load(ImGuiFileDialog::Instance()->GetFilePathName());
        }
        ImGuiFileDialog::Instance()->Close();
      }

      ImGui::Spacing();

      // Rendering settings
      if(ImGui::CollapsingHeader("Rendering"))
      {
        bool needRecreate = false, resetFrameAccumulation = false;
        if(ImGui::Combo("Renderer", &m_config->rendererIndex,
                        RENDERER_NAMES.data(), int(RENDERER_NAMES.size())))
        {
          needRecreate = true;
        }

        // Renderer-specific UI
        std::visit(
            [&](auto& r) { r.uiSection(needRecreate, resetFrameAccumulation); }, *m_renderer);
        if(needRecreate)
        {
          m_garbage.push(Garbage{moveAny(std::move(m_renderer)), lastRenderFinishedSemaphore});
          if constexpr(!ASYNC_RECREATE_RENDERER)
          {
            waitAndEmptyGarbage(m_garbage, m_context.device.get());
          }
          m_renderer = createRenderer(m_config->rendererIndex);
        }
        if(resetFrameAccumulation)
          m_rendererCommon.m_frameAccumIndex = 0;

        ImGui::SliderInt("Max Frames In Flight", &m_config->maxFramesInFlight, 1, 4);

        m_framebuffer->renderUI();

        if(m_config->rendererIndex == 1)
        {
          ImGui::Checkbox("Use Occlusion", &m_rendererCommon.m_config->useOcclusion);
        }

        if(ImGui::Button("Reload Shaders (R)"))
        {
          m_reloadShadersRequested = true;
        }

        if(m_rendererCommon.uiSky())
          m_rendererCommon.m_frameAccumIndex = 0;  // reset framebuffer accumulation
      }

      // Streaming stats and configuration
      if(ImGui::CollapsingHeader("Streaming"))
      {
        bool  recreateRenderer = false;
        auto& streaming = getStreaming(*m_renderer);

        // Memory usage plot
        float streamingMemoryMB =
            float(streaming.pool().bytesAllocated()) / 1024.0f / 1024.0f;
        if(streamingMemoryMB != m_memoryUsageHistory.last()
           || streaming.residentGroups() != m_residentGroupsHistory.last())
        {
          m_memoryUsageHistory.push(streamingMemoryMB);
          m_residentGroupsHistory.push(streaming.residentGroups());
        }
        if(ImPlot::BeginPlot("Streaming Memory Pool Usage",
                             ImVec2(-1, 150 * ImGui::GetIO().FontGlobalScale)))
        {
          ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
          ImPlot::SetupAxisLimits(ImAxis_X1, 0,
                                  static_cast<double>(m_memoryUsageHistory.size()),
                                  ImGuiCond_Always);
          ImPlot::SetupAxis(ImAxis_Y1, "MiB", ImPlotAxisFlags_LockMin);
          ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0,
                                  static_cast<double>(m_config->streamingBufferSizeMB),
                                  ImGuiCond_Always);
          ImPlot::SetupAxis(ImAxis_Y2, "Groups",
                            ImPlotAxisFlags_LockMin | ImPlotAxisFlags_Opposite);
          ImPlot::SetupAxisLimits(ImAxis_Y2, 0.0,
                                  static_cast<double>(m_config->streamingMaxResidentGroups),
                                  ImGuiCond_Always);

          // Plot memory usage on left Y axis
          ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
          auto plotFloat = [](auto name, auto& series) {
            ImPlot::PlotLine(name, series.data(), static_cast<int>(series.size()),
                             1.0f, 0.0f, ImPlotShadedFlags_None,
                             static_cast<int>(series.pivot()), sizeof(*series.data()));
          };
          plotFloat("Memory", m_memoryUsageHistory);

          // Plot resident groups on right Y axis
          ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
          auto plotUint = [](auto name, auto& series) {
            ImPlot::PlotLine(name, series.data(), static_cast<int>(series.size()),
                             1.0f, 0.0f, ImPlotShadedFlags_None,
                             static_cast<int>(series.pivot()), sizeof(*series.data()));
          };
          plotUint("Groups", m_residentGroupsHistory);

          ImPlot::EndPlot();
        }

        // Streaming configuration
        ImGui::Separator();
        if(ImGui::SliderInt("Streaming Pool Size (MiB)", (int*)&m_config->streamingBufferSizeMB,
                            100, 4000, "%d", ImGuiSliderFlags_AlwaysClamp))
        {
          // Make cluster groups broadly fit the streaming buffer size.
          // TODO: measure this - it's implementation and HW dependent!
          m_config->streamingMaxResidentGroups =
              (m_config->streamingBufferSizeMB * 33000) / 2048;
          recreateRenderer = true;
        }
        if(ImGui::SliderInt("Max Cluster Groups", &m_config->streamingMaxResidentGroups,
                            int(m_scene.view.counts.totalMeshes * 64u), 1 << 16,
                            "%d", ImGuiSliderFlags_AlwaysClamp))
        {
          recreateRenderer = true;
        }
        ImGui::Checkbox("Stress Test", &m_config->streamingStress);
        if(ImGui::Checkbox("Greedy Unload", &m_config->greedyUnload))
        {
          streaming.setGreedyUnload(m_config->greedyUnload);
        }

        // Streaming stats
        ImGui::Text("Pending Requests: %u", streaming.pendingRequests());
        if(ImGui::IsItemHovered())
          ImGui::SetTooltip("The number of loads/unloads waiting to be processed. This includes unload requests that are ignored until memory needs to be reclaimed and loads blocked by the memory or group limit.");
        ImGui::Text("Memory Usage: %s",
                    formatBytes(streaming.pool().bytesAllocated()).c_str());
        ImGui::Text("Internal Fragmentation: %s",
                    formatBytes(streaming.pool().internalFragmentation()).c_str());
        ImGui::Text("External Fragmentation: %s",
                    formatBytes(streaming.pool().externalFragmentation()).c_str());
        ImGui::Text("Staging Memory: %s",
                    formatBytes(streaming.stagingMemoryUsage()).c_str());

        // Cluster information
        ImGui::Text("Resident Clusters: %zu / %u", (size_t)m_scene.vk.totalResidentClusters,
                    m_scene.view.counts.totalClusters);
        ImGui::Text("Resident Clusters x Instances: %zu / %u",
                    (size_t)m_scene.vk.totalResidentInstanceClusters,
                    m_scene.view.counts.maxTotalInstanceClusters);

        if(recreateRenderer)
        {
          m_garbage.push(Garbage{moveAny(std::move(m_renderer)), lastRenderFinishedSemaphore});

          if constexpr(!ASYNC_RECREATE_RENDERER)
          {
            waitAndEmptyGarbage(m_garbage, m_context.device.get());
          }

          m_renderer.reset();  // Destroy the renderer first
          m_renderer = createRenderer(m_config->rendererIndex);
          m_rendererCommon.m_frameAccumIndex = 0;
        }
      }

      // Mesh Processing Configuration
      if(ImGui::CollapsingHeader("LOD Processing"))
      {
        ImGui::TextWrapped("These settings control how meshes are preprocessed into LOD clusters. Changes require rebuilding the scene cache.");
        ImGui::Spacing();

        ImGui::SliderInt("Cluster Size",
                         (int*)&m_config->sceneLodConfig.clusterSize, 8, 256);
        ImGui::SliderInt("Cluster Group Size",
                         (int*)&m_config->sceneLodConfig.clusterGroupSize, 2, 32);
        ImGui::SliderFloat("Decimation Factor",
                           &m_config->sceneLodConfig.lodLevelDecimationFactor, 0.1f, 0.9f);
        if(m_scene.mapping.path.empty())
        {
          // Allow adjusting the generated scene detail if it's already loaded
          ImGui::SliderFloat("Generated Detail Scale",
                             &m_config->generatedSceneDetailScale, 0.1f, 4.0f);
        }

        if(ImGui::Button("Rebuild LOD"))
        {
          m_config->invalidateNextLoadRendercache = true;
          load(m_scene.mapping.path.string().c_str());
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(This may take time)");
      }
    });

    // Profiler window
    vko::imgui::window("Profiler", &m_showProfiler)(
        [&] { renderProfiler("Frame Timing", m_profiler); });

    // Camera paths window
    m_cameraPaths->renderUI();

    // Viewport window - locked to the central dockspace, defines the render region.
    // Note: It must be a window for ImGui docking to give it a stable "remaining space" rect.
    ImGuiWindowFlags viewportFlags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar
        | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoNavFocus
        | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoBackground;

    const ImGuiID dockspaceId    = ImGui::GetID("MainDockSpace");
    ImGuiID       viewportDockId = dockspaceId;
    if(ImGuiDockNode* centralNode = ImGui::DockBuilderGetCentralNode(dockspaceId))
    {
      viewportDockId = centralNode->ID;
    }

    // Keep the viewport docked in the central node and hide its tab bar.
    ImGuiWindowClass viewportClass{};
    viewportClass.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoTabBar;
    ImGui::SetNextWindowClass(&viewportClass);
    ImGui::SetNextWindowDockID(viewportDockId, ImGuiCond_FirstUseEver);

    vko::imgui::styleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F))([&] {
      vko::imgui::window("##Viewport", nullptr, viewportFlags)([&] {
        ImVec2     size = ImGui::GetContentRegionAvail();
        VkExtent2D desiredExtent{uint32_t(std::max(1.0f, size.x)),
                                 uint32_t(std::max(1.0f, size.y))};

        if(desiredExtent.width != m_framebuffer->size().width
           || desiredExtent.height != m_framebuffer->size().height)
        {
          resize(glm::uvec2(desiredExtent.width, desiredExtent.height),
                 lastRenderFinishedSemaphore);
        }

        // Display the rendered scene as an ImGui image
        ImGui::ImageWithBg(m_framebuffer->displayImGuiTexture(), size, {0, 0},
                           {1, 1}, {0, 0, 0, 1});

        // Switch focus on right and middle click for initial drag to zoom/pan
        if(ImGui::IsWindowHovered()
           && (ImGui::IsMouseClicked(ImGuiMouseButton_Right)
               || ImGui::IsMouseClicked(ImGuiMouseButton_Middle)))
          ImGui::SetWindowFocus();

        // Camera input
        // Note: GLFW scroll is callback-only (no polling), so we use ImGui's captured value
        // Always call to record deltas, pass focus/hover state to control manipulation
        m_cameraManipulator.manipulateWithGLFW(
            window, m_camera, ImGui::GetIO().MouseWheel,
            ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows),
            ImGui::IsItemHovered());

        // Update camera animation (overrides manual camera control when playing)
        m_cameraPaths->update();
      });
    });

    // Show error popup if scene loading failed
    if(!m_sceneLoadError.empty())
    {
      ImGui::OpenPopup("Scene Load Error");
      vko::imgui::popupModal("Scene Load Error", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)([&] {
        ImGui::Text("Failed to load scene:");
        ImGui::TextWrapped("%s", m_sceneLoadError.c_str());
        ImGui::Spacing();
        if(ImGui::Button("OK", ImVec2(120, 0)))
        {
          m_sceneLoadError.clear();
          ImGui::CloseCurrentPopup();
        }
      });
    }
  }

  void renderToFramebuffer(vko::CyclingCommandBuffer<>& cmd)
  {
    // NOTE: cmd is m_context.staging.commandBuffer() passed back to us. A
    // bit of an ownership smell.

    // Scope for the frame timer before m_profiler.endQueryBatch()
    // TODO: less indentation :P
    {
      ScopedGpuTimer frameTimer(m_context.device.get(), m_profiler, cmd, "Frame");

      // Free unused allocations after all work that references them has finished.
      // E.g. check if the submit in GroupModsList::modifyGroups() has finished
      // before freeing streaming memory pool allocations, calling the
      // streaming::ClusterGroupVk destructor
      {
        vkobj::NvtxRange garbageRange("Free Garbage");
        ScopedCpuTimer   timer(m_profiler, "Free Garbage");
        emptyUnusedGarbage(m_garbage, m_context.device.get());
      }

      // If a new scene has finished loading, replace the current one
      if(m_sceneFuture.valid()
         && m_sceneFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
      {
        vkobj::NvtxRange sceneSwapRange("Scene Swap");
        // The RenderableScene is created on the main thread because the streaming
        // object always uses the same transfer queue in the background, which is
        // not yet synchronized. For the same reason we must stop the streaming
        // thread now, which in turn means we must empty the garbage, which
        // necessitates a wait-for-idle.
        // TODO: add a std::mutex to SerialTimelineQueue?
        vko::check(m_context.device.get().vkQueueWaitIdle(m_queueStates.primary));

        // Swap out the scene. Renderer (which owns streaming) is scene-dependent and will be recreated.
        waitAndEmptyGarbage(m_garbage, m_context.device.get());
        std::optional<SceneFile> sceneFile;
        try
        {
          // The std::async's future may forward an exception, e.g. if loading an
          // incompatible scene.
          sceneFile = m_sceneFuture.get();
        }
        catch(const std::exception& e)
        {
          std::print(stderr, "Error loading scene: {}", e.what());
          m_sceneLoadError = e.what();
        }
        if(sceneFile.has_value())
        {
          // Destroy the objects first (not exception safe). TODO: move to garbage instead?
          std::move(m_renderer).reset();
          RendererCommon(std::move(m_rendererCommon));
          RenderableScene(std::move(m_scene));
          m_scene = RenderableScene{
              std::move(*sceneFile),     m_context.device.get(),
              m_context.allocator.get(), m_context.staging,
              m_queueStates.primary,
          };
          // Recreate renderer (and streaming) for the new scene
          m_rendererCommon = RendererCommon{m_context.device.get(),
                                            m_context.allocator.get(), m_config->common};
          m_renderer       = createRenderer(m_config->rendererIndex);
          onSceneLoaded();
        }

        m_sceneFuture = {};
        m_sceneLoadProgress.reset();

        m_rendererCommon.m_frameAccumIndex = 0;  // reset framebuffer accumulation
      }

      // Streaming stress test with a random camera position each frame
      if(m_config->streamingStress)
      {
        float sceneSize =
            glm::length(m_scene.view.worldAABB.max - m_scene.view.worldAABB.min);
        m_camera.setLookat(m_scene.view.worldAABB.center()
                               + (glm::vec3{-0.2F, 0.4F, 0.8F} + glm::ballRand(1.0f)) * sceneSize,
                           m_scene.view.worldAABB.center(), {0.0F, 1.0F, 0.0F});
      }

      if(ImGui::IsKeyPressed(ImGuiKey_R))
      {
        m_reloadShadersRequested = true;
      }

      bool reloadShaders       = m_reloadShadersRequested;
      m_reloadShadersRequested = false;

      // Press "R" to reload traversal and rendering shaders
      if(reloadShaders)
      {
        // Make sure no GPU objects are in use
        // TODO: handle ASYNC_RECREATE_RENDERER here instead
        vko::check(m_context.device.get().vkQueueWaitIdle(m_queueStates.primary));

        // Garbage may reference the memory pool owned by streaming and
        // must be cleared now. Could make its allocator a weak_ptr.
        m_garbage = {};

        // Clear the current renderer first to avoid 2x streaming memory usage
        m_renderer.reset();
        RendererCommon(std::move(m_rendererCommon));

        // Reload traversal shaders by re-creating the common renderer
        // TODO: recreate only shader modules and pipelines?
        m_rendererCommon = RendererCommon{m_context.device.get(),
                                          m_context.allocator.get(), m_config->common};

        // This happens anyway, but for completeness, reset framebuffer
        // accumulation
        m_rendererCommon.m_frameAccumIndex = 0;

        m_renderer = createRenderer(m_config->rendererIndex);
      }

      auto& streaming = getStreaming(*m_renderer);

      // Build one batch of cluster acceleration structures for the streaming
      // pipeline.
      {
        ScopedGpuTimer timer(m_context.device.get(), m_profiler, cmd, "Build CLAS");
        streaming.buildClasBatch(m_context.device.get(), m_context.staging, false);
      }

      // Insert scene geometry streaming from the streaming thread before
      // traversal to compute LOD. Keep any unloaded pages around for another
      // frame since commands referencing them may still be in flight.
      {
        ScopedGpuTimer timer(m_context.device.get(), m_profiler, cmd, "Modify Groups");
        streaming.modifyGroups(m_context.device.get(), m_context.staging,
                               m_scene.view, m_scene.vk.meshPointers, false,
                               m_scene.vk.totalResidentClusters,
                               m_scene.vk.totalResidentInstanceClusters);
      }

      // Render the scene. This includes traversing LOD hierarchies to create
      // per-instance LODs. If the renderer is a raytracer, acceleration
      // structures are built during this process.
      {
        vkobj::NvtxRange sceneRenderRange("Scene Render");
        m_rendererCommon.cmdUpdateParams(m_context.device.get(), *m_framebuffer, m_camera,
                                         m_scene.view.maxWorldDiagonalInObjectSpace, cmd);

        RenderParams renderParams{m_context,      m_rendererCommon, m_camera,
                                  *m_framebuffer, m_profiler,       m_garbage,
                                  m_queueStates};
        std::visit(
            [&](auto& r) {
              r.render(renderParams, m_scene.vk, m_context.staging);
            },
            *m_renderer);
      }

      {
        vkobj::NvtxRange tonemapRange("Tonemap");
        m_framebuffer->cmdTonemap(cmd);
      }

      // Gather any new geometry requests from traversal and send them to the
      // streaming thread. Results will be picked up next frame or when next
      // available.
      {
        vkobj::NvtxRange makeRequestsRange("Make Requests");
        ScopedGpuTimer timer(m_context.device.get(), m_profiler, cmd, "Make Requests");
        streaming.makeRequests(m_context.device.get(), m_scene.vk.allGroupNeededFlags,
                               m_context.staging.commandBuffer().nextSubmitSemaphore(), cmd);
      }

      // Limit parallel frame saves to 12, join oldest if full
      while(m_frameSaveQueue.size() >= 12)
      {
        m_frameSaveQueue.front().join();
        m_frameSaveQueue.pop();
      }

      // Save frame if camera animation recording is enabled
      if(m_cameraPaths->saveFrames())
      {
        VkExtent2D extent   = m_framebuffer->size();
        int        frameNum = m_cameraPaths->animationFrame();

        // Transition LDR image to TRANSFER_SRC
        vko::cmdImageBarrier(
            m_context.device.get(), cmd, m_framebuffer->displayLdrImage(),
            vko::ImageAccess{.stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             .access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                             .layout = VK_IMAGE_LAYOUT_GENERAL},
            vko::ImageAccess{.stage  = VK_PIPELINE_STAGE_TRANSFER_BIT,
                             .access = VK_ACCESS_TRANSFER_READ_BIT,
                             .layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});

        // Record download (returns future to pixel data)
        auto pixelsFuture = vko::download(
            m_context.staging, m_context.device.get(),
            m_framebuffer->displayLdrImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VkImageSubresourceLayers{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                     .mipLevel   = 0,
                                     .baseArrayLayer = 0,
                                     .layerCount     = 1},
            VkExtent3D{extent.width, extent.height, 1}, Framebuffer::c_colorLDRFormat);

        // Transition back to GENERAL for next frame
        vko::cmdImageBarrier(
            m_context.device.get(), cmd, m_framebuffer->displayLdrImage(),
            vko::ImageAccess{.stage  = VK_PIPELINE_STAGE_TRANSFER_BIT,
                             .access = VK_ACCESS_TRANSFER_READ_BIT,
                             .layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
            vko::ImageAccess{.stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             .access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                             .layout = VK_IMAGE_LAYOUT_GENERAL});

        // Save in background thread (waits for GPU, then writes PNG)
        auto framePath = m_cameraPaths->framePath(frameNum);
        m_frameSaveQueue.push(std::jthread([extent, framePath, future = std::move(pixelsFuture),
                                            device = &m_context.device.get()]() mutable {
          auto& pixels = future.get(*device);  // Wait and get pixel data
          // Create directory if it doesn't exist
          if(framePath.has_parent_path())
            std::filesystem::create_directories(framePath.parent_path());
          stbi_write_png(framePath.string().c_str(), int(extent.width),
                         int(extent.height), 4, pixels.data(), int(extent.width) * 4);
        }));
      }
    }

    // End this frame's query batch so the pool can recycle queries after submission completes
    m_profiler.endQueryBatch(m_context.staging.commandBuffer().nextSubmitSemaphore());

    // Collect any GPU profiler results that have completed
    m_profiler.collectGpuResults(m_context.device.get());
  }

  void resize(const glm::uvec2& vpSize, vkobj::SemaphoreValue lastRenderFinishedSemaphore)
  {
    // Recreate the framebuffer, keeping the old one alive until the next frame is complete.
    // The old framebuffer contains the ImGui texture descriptor, so this prevents dangling refs.
    m_garbage.push(Garbage{moveAny(std::move(m_framebuffer)), lastRenderFinishedSemaphore});

    m_framebuffer.emplace(m_context.device.get(), m_context.allocator.get(),
                          m_context.commandPool, m_queueStates.primary,
                          m_glslCompiler, vpSize);

    // DANGER: update renderer references to avoid them becoming dangling
    RenderParams renderParams{m_context,      m_rendererCommon, m_camera,
                              *m_framebuffer, m_profiler,       m_garbage,
                              m_queueStates};
    std::visit([&](auto& r) { r.updatedFrambuffer(renderParams); }, *m_renderer);

    // gbuffer is recreated so reset the accumulation count
    m_rendererCommon.m_frameAccumIndex = 0;
  }

  // Due to inter-member dependencies, default move assignment ordering would break things
  Sample(Sample&& other) noexcept   = default;
  Sample& operator=(Sample&& other) = delete;

private:
  vko::shared_obj<Config> m_config;
  bool                    m_dlssAvailable;
  TimelineQueueContainer  m_queueStates;
  vkobj::Context m_context;  // non-owning vulkan context, plus staging memory for one thread
  SampleGlslCompiler            m_glslCompiler;
  vko::unique_obj<Framebuffer>  m_framebuffer;
  RenderableScene               m_scene;
  std::unique_ptr<TaskProgress> m_sceneLoadProgress;
  std::string m_sceneLoadError;  // error message for failed scene loads
  PlotSamples<float, 20>              m_memoryUsageHistory;
  PlotSamples<uint32_t, 20>           m_residentGroupsHistory;
  RendererCommon                      m_rendererCommon;
  SampleProfiler                      m_profiler;
  std::unique_ptr<RendererVariant>    m_renderer;
  vko::shared_obj<CameraPathsElement> m_cameraPaths;
  Camera                              m_camera;
  CameraManipulator                   m_cameraManipulator;
  std::queue<Garbage>                 m_garbage;
  fs::path                            m_cacheDir;
  std::queue<std::jthread> m_frameSaveQueue;  // Parallel frame saves (max 6)
  std::future<std::optional<SceneFile>> m_sceneFuture;  // for async loading, optional due to msvc bug fixed in v19.32
  bool m_reloadShadersRequested = false;
  bool m_showSettings           = true;
  bool m_showProfiler           = false;
};

// Custom surface creation function that respects windowing system preference
template <vko::instance_commands InstanceCommands>
vko::SurfaceKHR makeSurfaceWithPreference(const InstanceCommands& vk,
                                          VkInstance              instance,
                                          const vko::glfw::PlatformSupport& support,
                                          GLFWwindow*        window,
                                          const std::string& preference)
{
  // If x11 is preferred, try X11 platforms first, then fall back to default
  if(preference == "x11")
  {
    std::optional<vko::SurfaceVariant> result;
    std::string                        exceptionStrings;

#if VK_KHR_xcb_surface
    if(!result && support.xcb && vko::glfw::glfwGetXCBConnection())
    {
      try
      {
        result = vko::glfw::makeXcbSurfaceKHR(vk, instance, window);
      }
      catch(const vko::Exception& e)
      {
        exceptionStrings += std::string(e.what()) + "\n";
      }
    }
#endif
#if VK_KHR_xlib_surface
    if(!result && support.xlib && vko::glfw::glfwGetX11Display())
    {
      try
      {
        result = vko::glfw::makeXlibSurfaceKHR(vk, instance, window);
      }
      catch(const vko::Exception& e)
      {
        exceptionStrings += std::string(e.what()) + "\n";
      }
    }
#endif

    if(result)
    {
      return vko::SurfaceKHR{std::move(*result)};
    }
    // X11 failed, fall through to default order
  }

  // Default order (or fallback if x11 preference failed)
  return vko::glfw::makeSurface(vk, instance, support, window);
}

// Overload for instance_and_commands (matches original makeSurface signature)
template <vko::instance_and_commands InstanceAndCommands>
vko::SurfaceKHR makeSurfaceWithPreference(const InstanceAndCommands& vk,
                                          const vko::glfw::PlatformSupport& support,
                                          GLFWwindow*        window,
                                          const std::string& preference)
{
  return makeSurfaceWithPreference(vk, vk, support, window, preference);
}

int main(int argc, char** argv)
{
  try
  {
    bool enableValidation = false;
    bool enableDlss       = true;
#ifndef NDEBUG
    enableValidation = true;
#endif

    // Paths outside the AppImage, if packaged
    // See: https://docs.appimage.org/packaging-guide/environment-variables.html
    auto outsideExePath =
        envVar<fs::path>("APPIMAGE").value_or(executablePath()).parent_path();
    auto currentPath = envVar<fs::path>("OWD").value_or(fs::current_path());

    // Parse command line arguments
    fs::path insideExePath = executablePath().parent_path();  // inside the AppImage, if packaged
    auto downloadPath = (executablePath().parent_path() / DOWNLOAD_RELPATH_FROM_BINARY)
                            .lexically_normal()
                            .string();
    std::vector<std::string> defaultSearchPaths = {
        currentPath.string(),  // Current working directory
        (currentPath / "downloaded_resources").string(),  // Downloaded resources (e.g., bunny.gltf)
        downloadPath,  // Path from binary location
        fs::absolute(insideExePath / "media").string(),  // Default media directory (for installed/packaged versions)
        fs::absolute(insideExePath / "..").string(),  // Parent of executable (for build directory)
    };
    std::print("Search paths:\n");
    for(auto& path : defaultSearchPaths)
    {
      std::print("  '{}'\n", path);
    }

    // Find gltf file in search paths
    auto findFile = [](const std::string& filename,
                       const std::vector<std::string>& searchPaths) -> std::string {
      for(const auto& dir : searchPaths)
      {
        auto candidate = fs::path(dir) / filename;
        if(fs::exists(candidate))
          return candidate.string();
      }
      return {};
    };

    std::string gltfPath = findFile("bunny_v2/bunny.gltf", defaultSearchPaths);
    std::string cacheDir = currentPath.string();  // or fs::temp_directory_path()?

    // Parse command line arguments
    args::ArgumentParser parser("vk_continuous_lod_clusters - a vulkan sample to demo continuous level of detail with ray tracing");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> meshArg(parser, "mesh", "Mesh filename (*.gltf) or 'generated'",
                                         {'m', "mesh"}, gltfPath);
    args::ValueFlag<std::string> cacheDirArg(parser, "cache-dir",
                                             "Directory to keep render cache files. Default is CWD",
                                             {'c', "cache-dir"}, cacheDir);
    args::ValueFlag<int> rendererIndexArg(parser, "renderer", "Renderer index (0=raytrace, 1=rasterize)",
                                          {'r', "renderer"}, 0);
    args::ValueFlag<bool> validateArg(parser, "validate", "Enable validation layers (may break depending on VK_NV_cluster_acceleration_structure support)",
                                      {"validate"}, enableValidation);
    args::ValueFlag<bool> enableDlssArg(parser, "dlss", "Enables DLSS, if supported",
                                        {"dlss"}, enableDlss);
#ifdef __linux__
    args::ValueFlag<std::string> wsiArg(parser, "wsi", "Window system integration preference (x11 or wayland)",
                                        {"wsi"}, "");
#endif

    vko::shared_obj<Config> config;
    try
    {
      parser.ParseCLI(argc, argv);
      if(meshArg)
        gltfPath = args::get(meshArg);
      if(cacheDirArg)
        cacheDir = args::get(cacheDirArg);
      if(validateArg)
        enableValidation = args::get(validateArg);
      if(enableDlssArg)
        enableDlss = args::get(enableDlssArg);
      if(rendererIndexArg)
        config->rendererIndex = args::get(rendererIndexArg);
    }
    catch(const args::Help&)
    {
      std::cout << parser;
      return EXIT_SUCCESS;
    }
    catch(const args::ParseError& e)
    {
      std::cerr << e.what() << std::endl;
      std::cerr << parser;
      return EXIT_FAILURE;
    }

    // Parse windowing system preference
    std::string windowingPreference;
#ifdef __linux__
    if(wsiArg)
      windowingPreference = args::get(wsiArg);
#endif

    if(gltfPath.empty())
    {
      std::print("Default bunny not found. Falling back to generated scene.\n");
    }
    if(gltfPath == "generated")
    {
      gltfPath = {};  // empty path implies generated scene
    }

    // Initialize GLFW and create window
    glm::ivec2            defaultWindowSize{1920, 1080};
    vko::glfw::ScopedInit glfwInit;
    vko::glfw::Window     window =
        vko::glfw::makeWindow(defaultWindowSize.x, defaultWindowSize.y, "VKLOD-Sample");

#ifdef __linux__
    // Register signal handlers for graceful shutdown on Linux
    std::signal(SIGTERM, signalHandler);
#endif

    // Set up drag and drop callback for GLTF files
    glfwSetDropCallback(window.get(), dropCallback);

    // Mostly needed for Wayland (Linux), which does not trigger
    // VK_ERROR_OUT_OF_DATE_KHR but provides a callback and expects the user to
    // resize the swapchain themselves. Note this should only be done if the
    // surface capability current extents are the special 0xFFFFFFFF value.
    {
      int width, height;
      glfwGetFramebufferSize(window.get(), &width, &height);
      g_glfwLastFramebufferSize = {static_cast<uint32_t>(width),
                                   static_cast<uint32_t>(height)};
    }
    glfwSetFramebufferSizeCallback(window.get(), resizeRequested);

    // Create Vulkan instance and device context
    vko::VulkanLibrary  library;
    vko::GlobalCommands globalCommands(library.loader());

    // Get GLFW platform support
    auto instanceExtensions =
        vko::toVector(globalCommands.vkEnumerateInstanceExtensionProperties, nullptr);
    vko::glfw::PlatformSupport platformSupport(instanceExtensions);

    // #DLSS_RR: Initialize NGX discovery info (optional - gracefully degrade if unavailable)
    auto appDataPath = outsideExePath.wstring();

    // AppImage-aware DLSS search paths. A current quirk of DLSS is it requires an absolute path.
    // 1. INSTALL: relative to exe location (inside AppImage if packaged)
    // 2. SOURCE: relative to CWD (uses OWD env var for AppImage)
    // 3. BINARY: relative path computed at build time, applied from build output directory
    auto dlssPath1 =
        (insideExePath / DLSS_RELPATH_FROM_INSTALL).lexically_normal().wstring();
    auto dlssPath2 =
        (currentPath / DLSS_RELPATH_FROM_SOURCE).lexically_normal().wstring();
    auto dlssPath3 = (executablePath().parent_path() / DLSS_RELPATH_FROM_BINARY)
                         .lexically_normal()
                         .wstring();

    // Print DLSS search paths
    if(enableDlss)
    {
      std::wcout << L"DLSS search paths:" << std::endl;
      std::wcout << L"  Path 1 (INSTALL): " << dlssPath1 << std::endl;
      std::wcout << L"  Path 2 (SOURCE):  " << dlssPath2 << std::endl;
      std::wcout << L"  Path 3 (BINARY):  " << dlssPath3 << std::endl;
      std::wcout << L"  AppDataPath:      " << appDataPath << std::endl;
    }

    vko::ngx::FeatureDiscovery dlssDiscoveryInfo(0xdeadbeefull, appDataPath,
                                                 NVSDK_NGX_Feature_RayReconstruction,
                                                 {
                                                     dlssPath1,
                                                     dlssPath2,
                                                     dlssPath3,
                                                 },
                                                 NVSDK_NGX_LOGGING_LEVEL_OFF);

    // Create Vulkan context (instance, device, queues, allocator)
    // Lambda scope hides moved-from instance from the rest of the function
    SampleVulkanContext vkContext = [&]() {
      SampleInstanceCreateInfo instanceCreateInfo(
          platformSupport, enableDlss ? &dlssDiscoveryInfo.discoveryInfo : nullptr,
          enableValidation);
      vko::Instance    instance(globalCommands, instanceCreateInfo);
      VkPhysicalDevice physicalDevice =
          selectPhysicalDevice(instance, platformSupport,
                               enableDlss ? &dlssDiscoveryInfo.discoveryInfo : nullptr);
      SampleQueues           queues = selectQueues(instance, physicalDevice);
      SampleDeviceCreateInfo deviceCreateInfo(
          instance, physicalDevice, queues,
          enableDlss ? &dlssDiscoveryInfo.discoveryInfo : nullptr, enableValidation);
      return SampleVulkanContext(globalCommands, std::move(instance),
                                 physicalDevice, deviceCreateInfo);
    }();


    // Create debug messenger AFTER vkContext so it destructs BEFORE vkContext.instance
    // (vkContext owns the actual VkInstance handle)
    std::optional<vko::GlobalDebugMessenger> debugMessenger;
    if(enableValidation)
    {
      debugMessenger.emplace(
          vkContext.instance,
          [](VkDebugUtilsMessageSeverityFlagBitsEXT severityBits, VkDebugUtilsMessageTypeFlagsEXT,
             const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) -> VkBool32 {
            VkFlags breakOnSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                      | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

            // Validation Error: [ VUID-VkClusterAccelerationStructureClustersBottomLevelInputNV-sType-sType ] | MessageID = 0x368e9b3 | vkGetClusterAccelerationStructureBuildSizesNV(): pInfo->pClustersBottomLevel->sType must be VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV.
            if(uint32_t(pCallbackData->messageIdNumber) == 0x368e9b3)
              return VK_FALSE;
            // Validation Error: [ VUID-VkClusterAccelerationStructureMoveObjectsInputNV-sType-sType ] | MessageID = 0xb8678e92 | vkGetClusterAccelerationStructureBuildSizesNV(): pInfo->pMoveObjects->sType must be VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV.
            if(uint32_t(pCallbackData->messageIdNumber) == 0xb8678e92)
              return VK_FALSE;
            // Validation Error: [ VUID-VkClusterAccelerationStructureMoveObjectsInputNV-type-parameter ] | MessageID = 0xfbebe16e | vkGetClusterAccelerationStructureBuildSizesNV(): pInfo->pMoveObjects->type (106) does not fall within the begin..end range of the VkClusterAccelerationStructureTypeNV enumeration tokens and is not an extension added token.
            if(uint32_t(pCallbackData->messageIdNumber) == 0xfbebe16e)
              return VK_FALSE;
            // Validation Error: [ VUID-VkClusterAccelerationStructureTriangleClusterInputNV-sType-sType ] | MessageID = 0x9af6b82f | vkGetClusterAccelerationStructureBuildSizesNV
            if(uint32_t(pCallbackData->messageIdNumber) == 0x9af6b82f)
              return VK_FALSE;
            // Validation Error: [ VUID-VkClusterAccelerationStructureTriangleClusterInputNV-vertexFormat-parameter ] | MessageID = 0x4e253d9b | vkGetClusterAccelerationStructureBuildSizesNV(): pInfo->pTriangleClusters->vertexFormat (524288) does not fall within the begin..end range of the VkFormat enumeration tokens and is not an extension added token.
            if(uint32_t(pCallbackData->messageIdNumber) == 0x4e253d9b)
              return VK_FALSE;
            // Validation Error: [ UNASSIGNED-GeneralParameterError-UnrecognizedBool32 ] | MessageID = 0xa320b052 | vkGetClusterAccelerationStructureBuildSizesNV(): pInfo->pMoveObjects->noMoveOverlap (1197) is neither VK_TRUE nor VK_FALSE. Applications MUST not pass any other values than VK_TRUE or VK_FALSE into a Vulkan implementation where a VkBool32 is expected.
            if(uint32_t(pCallbackData->messageIdNumber) == 0xa320b052)
              return VK_FALSE;
            // Validation Error: [ VUID-VkShaderModuleCreateInfo-pCode-08737 ] | MessageID = 0xa5625282 | vkCreateShaderModule(): pCreateInfo->pCode (spirv-val produced an error):
            // Invalid capability operand: 5437.
            if(uint32_t(pCallbackData->messageIdNumber) == 0xa5625282)
              return VK_FALSE;
            // Validation Error: [ VUID-VkShaderModuleCreateInfo-pCode-08739 ] | MessageID = 0x605314fa | vkCreateShaderModule(): SPIR-V has Capability (Unhandled OpCapability) declared, but this is not supported by Vulkan.
            if(uint32_t(pCallbackData->messageIdNumber) == 0x605314fa)
              return VK_FALSE;

            // Filter known false positives in cluster acceleration structure validation (SDK 1.4.328.1 bugs)
            // VUID-VkClusterAccelerationStructureCommandsInfoNV-srcInfosCount-parameter - srcInfosCount is actually optional
            if(std::string_view(pCallbackData->pMessageIdName ? pCallbackData->pMessageIdName : "")
                   .find("srcInfosCount-parameter")
               != std::string_view::npos)
              return VK_FALSE;
            // VUID-vkCmdBuildClusterAccelerationStructureIndirectNV-pCommandInfos-10459 - dstAddressesArray doesn't need ACCELERATION_STRUCTURE_STORAGE_BIT
            if(std::string_view(pCallbackData->pMessageIdName ? pCallbackData->pMessageIdName : "")
                   .find("pCommandInfos-10459")
               != std::string_view::npos)
              return VK_FALSE;
            // VUID-VkClusterAccelerationStructureCommandsInfoNV-dstImplicitData-parameter - dstImplicitData is optional when using dstAddressesArray
            if(std::string_view(pCallbackData->pMessageIdName ? pCallbackData->pMessageIdName : "")
                   .find("dstImplicitData-parameter")
               != std::string_view::npos)
              return VK_FALSE;

            std::cout << pCallbackData->pMessage << std::endl;

            if((severityBits & breakOnSeverity) != 0)
            {
              debug_break();
            }
            return VK_FALSE;  // never skip calls, even when there are errors
          });
    }

    // Create surface (after vkContext for correct destruction order)
    vko::SurfaceKHR surface =
        makeSurfaceWithPreference(vkContext.instance, platformSupport,
                                  window.get(), windowingPreference);

    // Initialize DLSS-RR (optional - gracefully degrade if unavailable)
    std::optional<vko::ngx::ScopedInit> ngx;
    try
    {
      if(enableDlss)
      {
        ngx.emplace(0xdeadbeefull, dlssDiscoveryInfo.applicationDataPath,
                    vkContext.instance, vkContext.physicalDevice, vkContext.device,
                    library.loader(), vkContext.instance.vkGetDeviceProcAddr,
                    &dlssDiscoveryInfo.commonInfo);
        std::cout << "DLSS initialized successfully" << std::endl;
      }
    }
    catch(const std::exception& e)
    {
      enableDlss = false;
      std::cout << "DLSS initialization failed (will continue without DLSS): "
                << e.what() << std::endl;
    }

    // Initialize ImGui and ImPlot contexts
    vko::imgui::Context  imguiContext;
    vko::implot::Context implotContext;

    // Set dark color scheme to match original nvpro_core appearance
    // The tonemapper outputs sRGB-encoded data to UNORM, so ImGui should use sRGB colors directly (no conversion)
    setupImGuiStyle();

    ImGuiIO& ioMain = ImGui::GetIO();
    ioMain.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ioMain.ConfigFlags |= ImGuiConfigFlags_DockingEnable;  // Enable docking support

    // Set ImGui ini file to write outside AppImage (if packaged)
    static std::string s_imguiIniFilename =
        (outsideExePath / (executablePath().stem().string() + ".ini")).string();
    ioMain.IniFilename = s_imguiIniFilename.c_str();

    // Load existing settings or flag that we need to set up initial docking layout
    bool needInitialDockingLayout = !std::filesystem::exists(s_imguiIniFilename);
    ImGui::LoadIniSettingsFromDisk(s_imguiIniFilename.c_str());

    // Initialize ImGui for GLFW
    vko::imgui::ScopedGlfwInit imguiGlfw(window.get(), true);

    // Set window icon
    std::optional<Image> icon;
    auto                 iconPath = findFile("icon.png", defaultSearchPaths);
    if(!iconPath.empty())
      icon = createImage(iconPath, false);
    if(icon)
    {
      GLFWimage image{int(icon->extent.width), int(icon->extent.height),
                      const_cast<unsigned char*>(
                          reinterpret_cast<const unsigned char*>(icon->data.data()))};
      glfwSetWindowIcon(window.get(), 1, &image);
    }

    // Setup fonts (must be done before ImGui atlas is built)
    createDefaultFontWithIcons(ImGui::GetIO(), 14.0f);
    ExtraFonts extraFonts = initExtraFonts(ImGui::GetIO());

    // Get surface formats and present modes
    // Prefer UNORM (not SRGB) to avoid double gamma correction with ImGui
    // The tonemapping already outputs gamma-corrected values to the LDR UNORM buffer
    auto surfaceFormats = vko::toVector(vkContext.instance.vkGetPhysicalDeviceSurfaceFormatsKHR,
                                        vkContext.physicalDevice, surface);

    // First try to find BGRA8 UNORM with standard color space
    auto surfaceFormatIt =
        std::ranges::find_if(surfaceFormats, [](const VkSurfaceFormatKHR& format) {
          return format.format == VK_FORMAT_B8G8R8A8_UNORM
                 && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });

    // Fall back to anything
    if(surfaceFormatIt == surfaceFormats.end())
    {
      std::print(stderr, "No BGRA8 UNORM. Falling back to the first supported format. Expect the unexpected.\n");
      surfaceFormatIt = surfaceFormats.begin();
    }

    VkSurfaceFormatKHR surfaceFormat = *surfaceFormatIt;
    std::print("Selected swapchain format: {} (color space: {})\n",
               vko::to_string(surfaceFormat.format),
               static_cast<int>(surfaceFormat.colorSpace));

    // FIFO = present every frame at vsync; IMMEDIATE = no vsync; MAILBOX =
    // triple buffer, allow skipping frames
    VkPresentModeKHR presentModeVsyncOn  = VK_PRESENT_MODE_FIFO_KHR;
    VkPresentModeKHR presentModeVsyncOff = VK_PRESENT_MODE_IMMEDIATE_KHR;
    auto             surfacePresentModes =
        vko::toVector(vkContext.instance.vkGetPhysicalDeviceSurfacePresentModesKHR,
                      vkContext.physicalDevice, surface);
    if(std::ranges::find(surfacePresentModes, VK_PRESENT_MODE_IMMEDIATE_KHR)
       == surfacePresentModes.end())
    {
      std::print("Warning: VSync off with VK_PRESENT_MODE_IMMEDIATE_KHR not supported\n");
      presentModeVsyncOff = VK_PRESENT_MODE_FIFO_KHR;
    }
#if 0
    // Could use mailbox if available - lower latency vsync when framerate is high
    if(std::ranges::find(surfacePresentModes, VK_PRESENT_MODE_MAILBOX_KHR) != surfacePresentModes.end())
      presentModeVsyncOn = VK_PRESENT_MODE_MAILBOX_KHR;
#endif

    // Create swapchain with automatic recreation on surface changes
    vko::RecreatingSwapchain<>::Config swapchainConfig{
        surface,
        surfaceFormat,
        vkContext.primary.location.family,
        config->vSync ? presentModeVsyncOn : presentModeVsyncOff,
        g_glfwLastFramebufferSize,
        0u};
    vko::RecreatingSwapchain<> swapchain(vkContext.instance, vkContext.device,
                                         vkContext.physicalDevice, swapchainConfig);

    // ImGui descriptor pool - let ImGui manage its own pool by setting DescriptorPoolSize
    // We don't create a VkDescriptorPool ourselves; ImGui will create one internally
    // This avoids hardcoded magic numbers and lets ImGui manage what it needs

    // Initialize ImGui for Vulkan - use dynamic rendering
    ImGui_ImplVulkan_PipelineInfo pipelineInfo{
        .RenderPass  = VK_NULL_HANDLE,
        .Subpass     = 0,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
                                        .pNext                = nullptr,
                                        .viewMask             = 0,
                                        .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats =
                                            &surfaceFormat.format,
                                        .depthAttachmentFormat = VK_FORMAT_UNDEFINED,
                                        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED},
        .SwapChainImageUsage = 0};

    ImGui_ImplVulkan_InitInfo imguiVulkanInitInfo{
        .ApiVersion     = SampleVulkanVersion,
        .Instance       = vkContext.instance,
        .PhysicalDevice = vkContext.physicalDevice,
        .Device         = vkContext.device,
        .QueueFamily    = vkContext.primary.location.family,
        .Queue          = vkContext.primary.queue,
        .DescriptorPool = VK_NULL_HANDLE,  // Let ImGui create its own pool
        .DescriptorPoolSize = 1000,  // ImGui will create a pool with this many combined image samplers
        .MinImageCount = 2u,  // Unused - we manage swapchain ourselves
        .ImageCount = 2u,  // Unused - we manage swapchain ourselves (just needs >= MinImageCount)
        .PipelineCache              = VK_NULL_HANDLE,
        .PipelineInfoMain           = pipelineInfo,
        .PipelineInfoForViewports   = {},  // Not using secondary viewports
        .UseDynamicRendering        = true,
        .Allocator                  = nullptr,
        .CheckVkResultFn            = nullptr,
        .MinAllocationSize          = 1024 * 1024,
        .CustomShaderVertCreateInfo = {},
        .CustomShaderFragCreateInfo = {}};
    vko::imgui::ScopedVulkanInit imguiVulkan(library.loader(), imguiVulkanInitInfo);

    // Create Sample after ImGui is initialized since Framebuffer also creates
    // an ImGui texture with ImGui descriptor sets
    Sample sample(vkContext, config,
                  glm::uvec2(swapchain.extent().width, swapchain.extent().height),
                  gltfPath, cacheDir, enableDlss);

    std::print("Starting main loop\n");
    vkobj::SemaphoreValue lastRenderFinishedSemaphore =
        vko::SemaphoreValue::makeSignalled();

    // Main render loop
    while(!glfwWindowShouldClose(window.get()))
    {
      // Handle dropped GLTF files
      if(g_droppedFile)
      {
        sample.load(*g_droppedFile);
        g_droppedFile.reset();
      }

      // Check for DPI scaling and adjust the font size
      float xscale, yscale;
      glfwGetWindowContentScale(window.get(), &xscale, &yscale);
      ImGui::GetIO().FontGlobalScale = xscale;

      glfwPollEvents();

      // Start ImGui frame
      ImGui_ImplVulkan_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      // Set up initial docking layout on first run (when no .ini file exists)
      if(needInitialDockingLayout)
      {
        setupInitialDockingLayout();
        needInitialDockingLayout = false;
      }

      // Create fullscreen dockspace over the main viewport
      createDockspace();

      // Limit frames in flight using VK_KHR_present_wait (before renderUI so we
      // have accurate counts)
      swapchain.swapchain().waitForPresentIds(vkContext.device,
                                              size_t(config->maxFramesInFlight));

      // Render UI (may resize framebuffer)
      sample.renderUI(window.get(), lastRenderFinishedSemaphore, extraFonts);

      // Finalize ImGui
      ImGui::Render();

      auto maybeSemaphore = vko::tryPresentFrame(
          swapchain.swapchain(), sample.stagingCommandBuffer(),
          [&](auto& cmd, VkImage image, VkImageView imageView, VkImageLayout initialImageLayout) {
            // Transition swapchain image to COLOR_ATTACHMENT for ImGui
            vko::cmdImageBarrier(
                vkContext.device, cmd, image,
                vko::ImageAccess{.stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 .access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                                 .layout = initialImageLayout},
                vko::ImageAccess{.stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                          | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 .access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                 .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

            // Render the scene to the framebuffer
            sample.renderToFramebuffer(cmd);

            // Render ImGui (composites the scene texture via ImGui::Image())
            VkRenderingAttachmentInfo colorAttachment{
                .sType            = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext            = nullptr,
                .imageView        = imageView,
                .imageLayout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .resolveMode      = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp             = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp            = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}},
            };
            VkRenderingInfo renderingInfo{
                .sType      = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext      = nullptr,
                .flags      = 0,
                .renderArea = {.offset = {0, 0}, .extent = swapchain.extent()},
                .layerCount = 1,
                .viewMask   = 0,
                .colorAttachmentCount = 1,
                .pColorAttachments    = &colorAttachment,
                .pDepthAttachment     = nullptr,
                .pStencilAttachment   = nullptr,
            };
            vkContext.device.vkCmdBeginRendering(cmd, &renderingInfo);
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
            vkContext.device.vkCmdEndRendering(cmd);

            // Transition for present: COLOR_ATTACHMENT -> PRESENT_SRC
            vko::cmdImageBarrier(
                vkContext.device, cmd, image,
                vko::ImageAccess{.stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 .access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                 .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
                vko::ImageAccess{.stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 .access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                                 .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR});
          });

      if(maybeSemaphore)
      {
        lastRenderFinishedSemaphore = *maybeSemaphore;
      }

      // Handle all swapchain recreation
      VkPresentModeKHR desiredPresentMode =
          config->vSync ? presentModeVsyncOn : presentModeVsyncOff;
      if(!maybeSemaphore || desiredPresentMode != swapchainConfig.presentMode || m_needsResize)
      {
        swapchainConfig.presentMode    = desiredPresentMode;
        swapchainConfig.fallbackExtent = g_glfwLastFramebufferSize;
        swapchain.recreate(vkContext.instance, vkContext.device,
                           vkContext.physicalDevice, swapchainConfig);
        m_needsResize = false;
      }

#ifdef __linux__
      if(g_shouldShutdown)
      {
        std::print(stderr, "Received SIGTERM. Shutting down...\n");
        glfwSetWindowShouldClose(window.get(), GLFW_TRUE);
      }
#endif
    }
    std::println("Exiting main loop");

    // Wait for all queue operations to complete before destroying swapchain/ImGui
    vko::check(vkContext.device.vkQueueWaitIdle(vkContext.primary.queue));
  }
  catch(const std::exception& e)
  {
    // Catch-all case. Anything is fatal.
    std::print(stderr, "Exception thrown: {}\n", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
