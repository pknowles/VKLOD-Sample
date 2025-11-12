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

#include <acceleration_structures.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <debug_range_summary.hpp>
#include <filesystem>
#include <fmt/chrono.h>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <imgui.h>
#include <imgui/imgui_camera_widget.h>
#include <imgui/imgui_helper.h>
#include <imgui/imgui_icon.h>
#include <implot.h>
#include <lod_streaming_scene.hpp>
#include <memory>
#include <nvclusterlod/nvclusterlod_hierarchy.h>
#include <nvh/commandlineparser.hpp>
#include <nvh/fileoperations.hpp>
#include <nvh/nsightevents.h>
#include <nvh/primitives.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <nvvk/dynamicrendering_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/memallocator_vma_vk.hpp>
#include <nvvkhl/alloc_vma.hpp>
#include <nvvkhl/application.hpp>
#include <nvvkhl/element_camera.hpp>
#include <nvvkhl/element_gui.hpp>
#include <nvvkhl/element_nvml.hpp>
#include <nvvkhl/gbuffer.hpp>
#include <renderer_common.hpp>
#include <renderer_rasterize.hpp>
#include <renderer_raytrace.hpp>
#include <sample_app_element.hpp>
#include <sample_camera_paths.hpp>
#include <sample_element_profiler_copy.hpp>
#include <sample_font_extra.hpp>
#include <sample_glsl_compiler.hpp>
#include <sample_progress.hpp>
#include <sample_raytracing_objects.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <third_party/imgui/backends/imgui_impl_vulkan.h>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace fs = std::filesystem;

std::mutex  g_lastLogLineMutex;
std::string g_lastLogLine;

constexpr const char* INSTALL_SUBDIRECTORY = "GLSL_" PROJECT_NAME;

fs::path getRendercachePath(const fs::path& cacheDir, const fs::path& gltfPath)
{
  return cacheDir / ("rendercache_" + gltfPath.filename().string() + ".dat");
}

template <class T, size_t N>
class PlotSamples : public std::array<T, N>
{
public:
  PlotSamples() { push(T(0)); }  // implot only renders if there are at least two samples
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
  int            rendererIndex                 = 0;
  int            streamingBufferSizeMB         = 2048;  // 2GB
  int            streamingMaxResidentGroups    = 1 << 14;
  bool           invalidateNextLoadRendercache = false;
  bool           streamingStress               = false;  // random camera position each frame
  bool           greedyUnload                  = false;  // evict immediately, instead of waiting for memory pressure
  float          generatedSceneDetailScale     = 1.0f;
  SceneLodConfig sceneLodConfig;
};

std::unique_ptr<streaming::StreamingSceneVk> makeStreaming(ResourceAllocator*    allocator,
                                                           SampleGlslCompiler&   glslCompiler,
                                                           nvvkhl::Application*  app,
                                                           Scene&                scene,
                                                           SceneVK&              sceneVk,
                                                           vkobj::TimelineQueue& initQueueState,
                                                           const Config&         config,
                                                           bool                  requiresClas)
{
  return std::make_unique<streaming::StreamingSceneVk>(
      allocator, glslCompiler, VkDeviceSize(config.streamingBufferSizeMB) << 20,
      uint32_t(config.streamingMaxResidentGroups), app->getCommandPool(), initQueueState, scene, sceneVk,
      config.greedyUnload, requiresClas, app->getQueue(2).familyIndex /* transfer */, app->getQueue(2).queue /* transfer */
  );
}

SceneFile loadCachedScene(const fs::path        gltfPath,
                          const fs::path&       cachePath,
                          const Config&         config,
                          bool                  invalidateCache,
                          ResourceAllocator*    allocator,
                          VkCommandPool         initPool,
                          vkobj::TimelineQueue& initQueue,
                          SampleGlslCompiler&   glslCompiler,
                          TaskProgress*         progress = nullptr)
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
      GeneratedScene generatedScene = makeTerrainAndRocksScene(allocator->getDevice(), glslCompiler, allocator, initPool,
                                                               initQueue, config.generatedSceneDetailScale, *progress);
      result = makeSceneFromGenerated(std::move(generatedScene), generatedCachePath, config.sceneLodConfig, *progress);
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
                  ResourceAllocator*    allocator,
                  VkCommandPool         initCommandPool,
                  vkobj::TimelineQueue& initQueue,
                  SampleGlslCompiler&   glslCompiler,
                  nvvkhl::Application*  app,
                  bool                  requiresClas)
      : mapping{loadCachedScene(gltfPath, cachePath, config, invalidateCache, allocator, initCommandPool, initQueue, glslCompiler)}
      , view{*mapping.data}
      , vk{allocator, view, initCommandPool, initQueue.queue}
      , streaming(makeStreaming(allocator, glslCompiler, app, view, vk, initQueue, config, requiresClas))
  {
  }
  RenderableScene(SceneFile&&           mapping,
                  const Config&         config,
                  ResourceAllocator*    allocator,
                  VkCommandPool         initCommandPool,
                  vkobj::TimelineQueue& initQueue,
                  SampleGlslCompiler&   glslCompiler,
                  nvvkhl::Application*  app,
                  bool                  requiresClas)
      : mapping{std::move(mapping)}
      , view{*mapping.data}
      , vk{allocator, view, initCommandPool, initQueue.queue}
      , streaming(makeStreaming(allocator, glslCompiler, app, view, vk, initQueue, config, requiresClas))
  {
  }
  SceneFile mapping;  // file mapping objects
  Scene     view;     // pointers into the file mapping
  SceneVK   vk;       // GPU data, mostly uploaded from SceneMapping

  // Background threads and "groups" of geometry and acceleration structures
  std::unique_ptr<streaming::StreamingSceneVk> streaming;
};

// Lazy renderer container with a name and factory function. This saves inlining
// create() function whenever we recreate the renderer.
struct Renderer
{
  using CreateFunc = std::function<std::unique_ptr<RendererInterface>()>;
  RendererInterface* get()
  {
    if(!instance)
      instance = create();
    return instance.get();
  }
  std::string                        name;
  bool                               requiresClas;  // HACK: flag to skip building CLAS for rasterization
  CreateFunc                         create;
  std::unique_ptr<RendererInterface> instance;
};

SynchronizedMemAllocator makeMemAllocator(nvvkhl::Application* app)
{
  return SynchronizedMemAllocator(std::make_unique<VMAMemAllocator>(VmaAllocatorCreateInfo{
      .flags                       = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice              = app->getPhysicalDevice(),
      .device                      = app->getDevice(),
      .preferredLargeHeapBlockSize = 0,
      .pAllocationCallbacks        = nullptr,
      .pDeviceMemoryCallbacks      = nullptr,
      .pHeapSizeLimit              = nullptr,
      .pVulkanFunctions            = nullptr,
      .instance                    = app->getInstance(),
      .vulkanApiVersion = VK_API_VERSION_1_2,  // Not 1.3. See "# vma" in nvpro_core/third_party/CMakeLists.txt
      .pTypeExternalMemoryHandleTypes = nullptr,
  }));
}

SampleGlslCompiler makeGlslCompiler()
{
  return SampleGlslCompiler{INSTALL_SUBDIRECTORY,  // search after cmake --install
                            "shaders",       // this project's shaders, debugging from the source directory (canonical)
                            NVPRO_CORE_DIR,  // nvpro_core shaders
                            PROJECT_RELDIRECTORY "shaders",  // and again if CWD is not the source directory
                            PROJECT_NVPRO_CORE_RELDIRECTORY};
}

class Sample
{
public:
  Sample(nvvkhl::Application* app, glm::uvec2 vpSize, const std::filesystem::path& gltfPath, const std::filesystem::path& cacheDir)
      : m_app(app)  // held for making temp command buffers
      , m_memAlloc(makeMemAllocator(app))
      , m_context(m_app->getDevice(),
                  m_app->getPhysicalDevice(),
                  &m_memAlloc,
                  m_app->getQueue(0).familyIndex,
                  m_app->getQueue(0).queue)
      , m_queueStates(m_app)
      , m_glslCompiler(makeGlslCompiler())
      , m_tonemap(m_context.device, m_context.allocator)
      , m_framebuffer(std::make_unique<Framebuffer>(m_context.allocator, m_glslCompiler, m_tonemap, vpSize))
      , m_scene{gltfPath,
                getRendercachePath(cacheDir, gltfPath),
                m_config,
                false,
                m_context.allocator,
                m_context.commandPool,
                m_queueStates.primary,
                m_glslCompiler,
                m_app,
                true}
      , m_rendererCommon{m_context.allocator, m_glslCompiler, m_context.commandPool, app->getQueue(0).queue, m_scene.vk}
      , m_profiler(std::make_shared<nvvkhl_copy::ElementProfiler>(false))
      , m_cameraPaths(std::make_shared<CameraPathsElement>())
      , m_cacheDir(cacheDir)
  {
    // Inject VK_EXT_debug_utils begin/end labels too
    m_profiler->setLabelUsage(true);

    // The profiler is special because we need access to it from within the
    // 'Sample' class, so it's created and added here.
    // Similarly, frames can be recorded when playing camera animation
    m_app->addElement(m_profiler);
    m_app->addElement(m_cameraPaths);

    // Used to show status during scene loading. Could also use nvvkhl::ElementLogger.
    nvprintSetCallback([](int /* level */, const char* msg) {
      std::lock_guard lock(g_lastLogLineMutex);
      g_lastLogLine = msg;
    });

    onSceneLoaded();

    // Populate the lazily-created list of renderers
    m_renderers.push_back(Renderer{
        .name         = "Raytrace Clusters",
        .requiresClas = true,
        .create =
            [this]() {
              m_rendererRaytraceConfig.maxTotalClusterGroups = m_config.streamingMaxResidentGroups;
              return std::make_unique<RaytraceRenderer>(
                  RenderInitParams{
                      .instance     = m_app->getInstance(),
                      .context      = m_context,
                      .glslCompiler = m_glslCompiler,
                      .common       = m_rendererCommon,
                      .scene        = m_scene.view,
                      .sceneVk      = m_scene.vk,
                      .framebuffer  = *m_framebuffer,
                  },
                  m_rendererRaytraceConfig);
            },
        .instance = {},
    });
    m_renderers.push_back(Renderer{
        .name         = "Rasterize Clusters",
        .requiresClas = false,
        .create =
            [this]() {
              return std::make_unique<RasterizeRenderer>(m_context.allocator, m_glslCompiler, m_context.commandPool,
                                                         m_app->getQueue(0).familyIndex, m_context.queue,
                                                         m_rendererCommon, m_scene.view, m_scene.vk, *m_framebuffer);
            },
        .instance = {},
    });

    // Keep staging memory around, otherwise continuous allocations will block the
    // render thread and cause a lot of stutter.
    m_context.allocator->getStaging()->setFreeUnusedOnRelease(false);
    m_context.allocator->getStaging()->freeUnused();
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
      LOGW("A scene is already loading");
      return;
    }

    // Load the new scene. This may take a while
    LOGI("\n");                // clear g_lastLogLine
    m_sceneLoadError.clear();  // clear any previous error
    m_sceneLoadProgress.emplace();
    m_sceneFuture = std::async(std::launch::async, [this, filename, config = m_config,
                                                    invalidateRendercache = m_config.invalidateNextLoadRendercache]() {
      // Need a context with its own staging memory and a separate queue for
      // async loading, e.g. in case the scene is generated using a compute
      // shader.
      // TODO: make Context::queue a TimelineQueue?
      vkobj::Context       asyncLoadContext(m_app->getDevice(), m_app->getPhysicalDevice(), &m_memAlloc,
                                            m_app->getQueue(3).familyIndex, m_app->getQueue(3).queue);
      SampleGlslCompiler   compiler = makeGlslCompiler();
      vkobj::TimelineQueue asyncLoadQueue(m_app->getDevice(), m_app->getQueue(3).familyIndex, m_app->getQueue(3).queue);
      SceneFile            file = loadCachedScene(filename, getRendercachePath(m_cacheDir, filename), config,
                                                                           invalidateRendercache, asyncLoadContext.allocator, asyncLoadContext.commandPool,
                                                                           asyncLoadQueue, compiler, &m_sceneLoadProgress.value());
      return std::make_optional(std::move(file));
    });
    m_config.invalidateNextLoadRendercache = false;
  }

  void onSceneLoaded()
  {
    // Initial camera view for the new scene
    float sceneSize    = glm::length(m_scene.view.worldAABB.max - m_scene.view.worldAABB.min);
    float sceneSizeDec = powf(10.0f, ceilf(log10f(sceneSize)));
    CameraManip.setClipPlanes({sceneSizeDec * 0.001f, sceneSizeDec * 20.0f});
    CameraManip.setLookat(m_scene.view.worldAABB.center() + glm::vec3{-0.2F, 0.4F, 0.8F} * sceneSize,
                          m_scene.view.worldAABB.center(), {0.0F, 1.0F, 0.0F});
    CameraManip.setFov(80.0f);

    // Initialize fog parameters based on the scene size
    AABB  sceneAABB                                  = m_scene.view.worldAABB;
    float sceneAspectRatio                           = length(glm::vec2(sceneAABB.size())) / sceneAABB.size().y;
    m_rendererRaytraceConfig.shaders.fogHeightOffset = sceneAABB.min.y + sceneAABB.size().y * 0.7f;
    m_rendererRaytraceConfig.shaders.fogDensity =
        (2.0f / sceneAABB.size().y) * glm::smoothstep(4.0f, 8.0f, sceneAspectRatio);  // dense for high aspect scenes
  }

  void renderUI()
  {
    namespace PropertyEditor = ImGuiH::PropertyEditor;

    // Minimal transparent overlay with triangle count and control icons,
    // position at the top-right corner
    ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(displaySize.x - 10, 10), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.3f);
    if(ImGui::Begin("Cluster Count", nullptr,
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize
                        | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse))
    {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(118.0f / 255.0f, 185.0f / 255.0f, 0.0f / 255.0f, 1.0f));
      ImGui::PushFont(LargeFont::instance());
      m_renderers[m_config.rendererIndex].get()->uiOverlay();
      auto memoryUsed = m_renderers[m_config.rendererIndex].get()->deviceMemoryUsage() + m_scene.vk.deviceMemoryUsage()
                        + m_scene.streaming->pool().bytesAllocated();  // Excludes remaining reserved streaming memory, DLSS-RR and GBuffer
      ImGui::Text("Memory: %s", formatBytes(memoryUsed).c_str());
      nvh::Profiler::TimerInfo frameTime;
      m_profiler->getTimerInfo("Sample Frame", frameTime);  // time in microseconds, not seconds
      ImGui::Text("Time: %.1fms", frameTime.gpu.average / 1000.0);
      ImGui::PopFont();
      ImGui::PopStyleColor();

      ImGui::Spacing();

      // Green for active icons
      ImVec4 activeColor = ImVec4(118.0f / 255.0f, 185.0f / 255.0f, 0.0f / 255.0f, 1.0f);

      // Lambda for creating styled toggle icon buttons
      auto iconButton = [&activeColor](const char* icon, bool isActive, const char* tooltipOn, const char* tooltipOff) -> bool {
        ImGui::PushFont(LargeIconFont::instance());
        if(isActive)
        {
          ImGui::PushStyleColor(ImGuiCol_Button, activeColor);
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                ImVec4(activeColor.x * 1.2f, activeColor.y * 1.2f, activeColor.z * 1.2f, activeColor.w));
          ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                ImVec4(activeColor.x * 0.8f, activeColor.y * 0.8f, activeColor.z * 0.8f, activeColor.w));
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
      };

      // Display progress bar and the log if a scene is loading
      if(m_sceneLoadProgress)
      {
        // Create a top-center overlay window for loading progress
        ImGui::SetNextWindowPos(ImVec2(displaySize.x * 0.5f, 20), ImGuiCond_Always, ImVec2(0.5f, 0.0f));
        ImGui::SetNextWindowBgAlpha(0.8f);
        if(ImGui::Begin("Scene Loading", nullptr,
                        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize
                            | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse))
        {
          ImGui::Text("Loading scene...");
          ImGui::Spacing();
          auto progress = m_sceneLoadProgress->progress();
          ImGui::ProgressBar(progress.ratio, ImVec2(800 / ImGuiH::getDPIScale(), 0));
          if(progress.remaining)
          {
            ImGui::Text("%s -> %s", fmt::format("{:%M:%S}", std::chrono::floor<std::chrono::seconds>(progress.elapsed)).c_str(),
                        fmt::format("{:%M:%S}", std::chrono::floor<std::chrono::seconds>(progress.remaining.value())).c_str());
          }
          ImGui::Spacing();
#if 1
          ImGui::Text("%s", progress.currentWork().c_str());
#else
          if(!g_lastLogLine.empty())
          {
            // HACK: show the last line printed with LOG*(). It's probably the
            // status and useful to display, but far from future proof.
            std::lock_guard lock(g_lastLogLineMutex);
            ImGui::Text("%s", g_lastLogLine.c_str());
          }
#endif
        }
        ImGui::End();
      }

      // Settings window toggle
      if(iconButton(ImGuiH::icon_cog, m_showSettings, "Settings: ON", "Settings: OFF"))
      {
        m_showSettings = !m_showSettings;
      }

      // Profiler window toggle
      if(iconButton(ImGuiH::icon_bar_chart, m_profiler->isWindowShown(), "Profiler: ON", "Profiler: OFF"))
      {
        m_profiler->toggleWindowShown();
      }

      // V-Sync toggle
      bool vSync = m_app->isVsync();
      if(iconButton(ImGuiH::icon_reload, vSync, "V-Sync: ON (Ctrl+Shift+V)", "V-Sync: OFF (Ctrl+Shift+V)"))
      {
        vSync = !vSync;
        m_app->setVsync(vSync);
      }

      // Camera paths toggle
      if(iconButton(ImGuiH::icon_video, m_cameraPaths->isWindowVisible(), "Camera Paths: ON", "Camera Paths: OFF"))
      {
        m_cameraPaths->toggleWindow();
      }

      // Handle keyboard shortcuts normally done in nvvkhl::ElementDefaultMenu
      // since we don't want the menu
      if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
      {
        m_app->close();
      }
      if(ImGui::IsKeyPressed(ImGuiKey_V) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyDown(ImGuiKey_LeftShift))
      {
        vSync = !vSync;
        m_app->setVsync(vSync);
      }
    }
    ImGui::End();

    // Settings menu with a few common settings at the top and the rest hidden
    // in collapsible sections
    if(m_showSettings && ImGui::Begin("Settings", &m_showSettings))
    {
      //  Max Pixel Error
      if(ImGui::SliderFloat("Max. Pixel Error", &m_rendererCommon.m_config.lodTargetPixelError, 0.5f, 1000.0f))
      {
        m_rendererCommon.m_frameAccumIndex = 0;
      }
      ImGuiH::tooltip("Targets LOD selection where the screen space geometric error is below this value.");

      // Button to disable LOD changes
      ImGui::Checkbox("Lock LOD", &m_rendererCommon.m_config.lockLodCamera);
      ImGuiH::tooltip("Keeps the LOD at the same detail, ignoring further camera movement.");

      // Renderer-specific UI
      {
        bool needRecreate = false, resetFrameAccumulation = false;
        m_renderers[m_config.rendererIndex].get()->uiInline(needRecreate, resetFrameAccumulation);
        if(needRecreate)
        {
          m_garbage.push(Garbage{moveAny(std::move(m_renderers[m_config.rendererIndex].instance)),
                                 m_queueStates.primary.nextSubmitValue()});
          m_renderers[m_config.rendererIndex].instance.reset();
        }
        if(resetFrameAccumulation)
          m_rendererCommon.m_frameAccumIndex = 0;
      }

      ImGui::Spacing();
      ImGui::BeginDisabled(m_sceneFuture.valid());  // disable loading buttons when already loading

      // Provide a procedural scene for a quick demo without large downloads
      if(m_scene.mapping.path.filename() == "bunny.gltf")
      {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.82f, 1.0f, 0.36f, 1.0f));
        ImGui::TextWrapped("The bunny mesh is a simple placeholder. You can generate a more complex scene to better demonstrate LOD and streaming. It takes about a minute the first time.");
        ImGui::PopStyleColor();

        ImGui::SliderFloat("Detail Scale", &m_config.generatedSceneDetailScale, 0.1f, 4.0f);
        ImGuiH::tooltip("Scales the triangle count of all meshes. Ignored if a render cache exists - rebuild in 'LOD Processing'.");
        if(ImGui::Button("Generate Procedural Scene"))
        {
          load(fs::path{});  // empty path implies generated scene
        }
      }

      if(ImGui::Button("Load Scene"))
      {
        std::string filename = NVPSystem::windowOpenFileDialog(m_app->getWindowHandle(), "Pick scene file",
                                                               "Supported (glTF 2.0)|*.gltf;*.glb;"
                                                               "|All|*.*");
        if(!filename.empty())
          load(filename);
      }

      ImGui::EndDisabled();  // loading buttons section
      ImGui::Spacing();

      // Rendering settings
      if(ImGui::CollapsingHeader("Rendering"))
      {
        std::vector<const char*> rendererNames;
        for(auto& r : m_renderers)
          rendererNames.push_back(r.name.c_str());
        ImGui::Combo("Renderer", &m_config.rendererIndex, rendererNames.data(), int(rendererNames.size()));

        // Renderer-specific UI
        bool needRecreate = false, resetFrameAccumulation = false;
        m_renderers[m_config.rendererIndex].get()->uiSection(needRecreate, resetFrameAccumulation);
        if(needRecreate)
        {
          m_garbage.push(Garbage{moveAny(std::move(m_renderers[m_config.rendererIndex].instance)),
                                 m_queueStates.primary.nextSubmitValue()});
          m_renderers[m_config.rendererIndex].instance.reset();
        }
        if(resetFrameAccumulation)
          m_rendererCommon.m_frameAccumIndex = 0;

        m_framebuffer->tonemapUI();

        if(m_config.rendererIndex == 1)
        {
          ImGui::Checkbox("Use Occlusion", &m_rendererCommon.m_config.useOcclusion);
        }

        if(ImGui::Button("Reload Shaders (R)"))
        {
          m_reloadShadersRequested = true;
        }

        ImGuiH::CameraWidget();

        if(m_rendererCommon.uiSky())
          m_rendererCommon.m_frameAccumIndex = 0;  // reset framebuffer accumulation
      }

      // Streaming stats and configuration
      if(ImGui::CollapsingHeader("Streaming"))
      {
        // Memory usage plot
        float streamingMemoryMB = float(m_scene.streaming->pool().bytesAllocated()) / 1024.0f / 1024.0f;
        if(streamingMemoryMB != m_memoryUsageHistory.last()
           || m_scene.streaming->residentGroups() != m_residentGroupsHistory.last())
        {
          m_memoryUsageHistory.push(streamingMemoryMB);
          m_residentGroupsHistory.push(m_scene.streaming->residentGroups());
        }
        if(ImPlot::BeginPlot("Streaming Memory Pool Usage", ImVec2(-1, 150 * ImGui::GetIO().FontGlobalScale)))
        {
          ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
          ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(m_memoryUsageHistory.size()), ImGuiCond_Always);
          ImPlot::SetupAxis(ImAxis_Y1, "MiB", ImPlotAxisFlags_LockMin);
          ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, static_cast<double>(m_config.streamingBufferSizeMB), ImGuiCond_Always);
          ImPlot::SetupAxis(ImAxis_Y2, "Groups", ImPlotAxisFlags_LockMin | ImPlotAxisFlags_Opposite);
          ImPlot::SetupAxisLimits(ImAxis_Y2, 0.0, static_cast<double>(m_config.streamingMaxResidentGroups), ImGuiCond_Always);

          // Plot memory usage on left Y axis
          ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
          auto plotFloat = [](auto name, auto& series) {
            ImPlot::PlotLine(name, series.data(), static_cast<int>(series.size()), 1.0f, 0.0f, ImPlotShadedFlags_None,
                             static_cast<int>(series.pivot()), sizeof(*series.data()));
          };
          plotFloat("Memory", m_memoryUsageHistory);

          // Plot resident groups on right Y axis
          ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
          auto plotUint = [](auto name, auto& series) {
            ImPlot::PlotLine(name, series.data(), static_cast<int>(series.size()), 1.0f, 0.0f, ImPlotShadedFlags_None,
                             static_cast<int>(series.pivot()), sizeof(*series.data()));
          };
          plotUint("Groups", m_residentGroupsHistory);

          ImPlot::EndPlot();
        }

        // Streaming configuration
        ImGui::Separator();
        bool recreateStreamingAndRenderer = false;
        PropertyEditor::begin();
        auto clusterSizeSlider = [&] {
          return ImGui::SliderInt("Streaming Pool Size (MiB)", (int*)&m_config.streamingBufferSizeMB, 100, 4000, "%d",
                                  ImGuiSliderFlags_AlwaysClamp);
        };
        if(PropertyEditor::entry("Streaming Pool Size (MiB)", clusterSizeSlider))
        {
          // Make cluster groups broadly fit the streaming buffer size.
          // TODO: measure this - it's implementation and HW dependent!
          m_config.streamingMaxResidentGroups = (m_config.streamingBufferSizeMB * 33000) / 2048;
          recreateStreamingAndRenderer        = true;
        }
        auto maxClusterGroupsSlider = [&] {
          return ImGui::SliderInt("Max Cluster Groups", &m_config.streamingMaxResidentGroups,
                                  m_scene.view.counts.totalMeshes * 64, 1 << 16, "%d", ImGuiSliderFlags_AlwaysClamp);
        };
        recreateStreamingAndRenderer =
            PropertyEditor::entry("Max Cluster Groups", maxClusterGroupsSlider,
                                  "Streaming is limited by both memory and cluster groups. This avoids having to reallocate BLAS memory and traversal queues during streaming.")
            || recreateStreamingAndRenderer;
        PropertyEditor::entry("Stress Test", [&] { return ImGui::Checkbox("Stress Test", &m_config.streamingStress); });
        if(PropertyEditor::entry(
               "Greedy Unload", [&] { return ImGui::Checkbox("Greedy Unload", &m_config.greedyUnload); },
               "Evict immediately, instead of waiting for memory pressure."))
        {
          m_scene.streaming->setGreedyUnload(m_config.greedyUnload);
        }
        PropertyEditor::end();

        if(recreateStreamingAndRenderer || m_scene.streaming->requiresClas() != m_renderers[m_config.rendererIndex].requiresClas)
        {
          NVVK_CHECK(vkQueueWaitIdle(m_context.queue));

          // Garbage may reference the memory pool owned by m_scene.streaming and
          // must be cleared now. Could make its allocator a weak_ptr.
          m_garbage = {};

          m_scene.streaming.reset();
          m_scene.streaming = makeStreaming(m_context.allocator, m_glslCompiler, m_app, m_scene.view, m_scene.vk,
                                            m_queueStates.primary, m_config, m_renderers[m_config.rendererIndex].requiresClas);
          m_rendererCommon.m_frameAccumIndex = 0;
        }
        // TODO: Manual dependencies between renderer and streaming is asking
        // for trouble. Move streaming into the renderer to be safer.
        if(recreateStreamingAndRenderer)
        {
          m_garbage.push(Garbage{moveAny(std::move(m_renderers[m_config.rendererIndex].instance)),
                                 m_queueStates.primary.nextSubmitValue()});
          m_renderers[m_config.rendererIndex].instance.reset();
        }

        // Streaming stats
        ImGui::Text("Pending Requests: %u", m_scene.streaming->pendingRequests());
        ImGuiH::tooltip("The number of loads/unloads waiting to be processed. This includes unload requests that are ignored until memory needs to be reclaimed and loads blocked by the memory or group limit.",
                        true);
        ImGui::Text("Memory Usage: %s", formatBytes(m_scene.streaming->pool().bytesAllocated()).c_str());
        ImGui::Text("Internal Fragmentation: %s", formatBytes(m_scene.streaming->pool().internalFragmentation()).c_str());
        ImGui::Text("External Fragmentation: %s", formatBytes(m_scene.streaming->pool().externalFragmentation()).c_str());
        VkDeviceSize stagingAllocatedSize, stagingUsedSize;
        m_context.allocator->getStaging()->getUtilization(stagingAllocatedSize, stagingUsedSize);
        ImGui::Text("Staging Memory: %s", formatBytes(stagingAllocatedSize + m_scene.streaming->stagingMemoryUsage()).c_str());

        // Cluster information
        ImGui::Text("Resident Clusters: %zu / %u", (size_t)m_scene.vk.totalResidentClusters, m_scene.view.counts.totalClusters);
        ImGui::Text("Resident Clusters x Instances: %zu / %u", (size_t)m_scene.vk.totalResidentInstanceClusters,
                    m_scene.view.counts.maxTotalInstanceClusters);
      }

      // Mesh Processing Configuration
      if(ImGui::CollapsingHeader("LOD Processing"))
      {
        ImGui::TextWrapped("These settings control how meshes are preprocessed into LOD clusters. Changes require rebuilding the scene cache.");
        ImGui::Spacing();

        PropertyEditor::begin();
        PropertyEditor::entry("Cluster Size", [&] {
          return ImGui::SliderInt("Cluster Size", (int*)&m_config.sceneLodConfig.clusterSize, 8, 256);
        });
        PropertyEditor::entry("Cluster Group Size", [&] {
          return ImGui::SliderInt("Cluster Group Size", (int*)&m_config.sceneLodConfig.clusterGroupSize, 2, 32);
        });
        PropertyEditor::entry("Decimation Factor", [&] {
          return ImGui::SliderFloat("Decimation Factor", &m_config.sceneLodConfig.lodLevelDecimationFactor, 0.1f, 0.9f);
        });
        if(m_scene.mapping.path.empty())
        {
          // Allow adjusting the generated scene detail if it's already loaded
          PropertyEditor::entry("Generated Detail Scale", [&] {
            return ImGui::SliderFloat("Generated Detail Scale", &m_config.generatedSceneDetailScale, 0.1f, 4.0f);
          });
        }
        PropertyEditor::end();

        if(ImGui::Button("Rebuild LOD"))
        {
          m_config.invalidateNextLoadRendercache = true;
          load(m_scene.mapping.path.string().c_str());
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(This may take time)");
      }

      ImGui::End();
    }

    // Rendering Viewport
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    if(ImGui::Begin("Viewport"))
    {
      // Display the G-Buffer image
      ImGui::Image(m_framebuffer->displayLdrDescriptorSet(), ImGui::GetContentRegionAvail());
      ImGui::End();
    }
    ImGui::PopStyleVar();

    // Show error popup if scene loading failed
    if(!m_sceneLoadError.empty())
    {
      ImGui::OpenPopup("Scene Load Error");
      if(ImGui::BeginPopupModal("Scene Load Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
      {
        ImGui::Text("Failed to load scene:");
        ImGui::TextWrapped("%s", m_sceneLoadError.c_str());
        ImGui::Spacing();
        if(ImGui::Button("OK", ImVec2(120, 0)))
        {
          m_sceneLoadError.clear();
          ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
      }
    }
  }

  void render(VkCommandBuffer cmd)
  {
    nvvk::ProfilerVK::Section outerTimer(*m_profiler, "Sample Frame", cmd);

    // Free unused allocations after all work that references them has finished.
    // E.g. check if the submit in GroupModsList::modifyGroups() has finished
    // before freeing streaming memory pool allocations, calling the
    // streaming::ClusterGroupVk destructor
    {
      vkobj::NvtxRange          garbageRange("Free Garbage");
      nvvk::ProfilerVK::Section timer(*m_profiler, "Free Garbage", cmd);
      while(!m_garbage.empty() && m_garbage.front().semaphoreState.wait(m_context.device, 0 /* no wait */))
      {
        m_garbage.pop();
      }
    }

    // If a new scene has finished loading, replace the current one
    if(m_sceneFuture.valid() && m_sceneFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
    {
      vkobj::NvtxRange sceneSwapRange("Scene Swap");
      // The RenderableScene is created on the main thread because the streaming
      // object always uses the same transfer queue in the background, which is
      // not yet synchronized. For the same reason we must stop the streaming
      // thread now, which in turn means we must empty the garbage, which
      // necessitates a wait-for-idle.
      // TODO: add a std::mutex to TimelineQueue?
      vkQueueWaitIdle(m_queueStates.primary.queue);

      // Renderers are all scene dependent and need to be reset. This avoids some
      // complexity at the cost of recreating shaders and pipeline objects
      // unnecessarily.
      for(auto& r : m_renderers)
      {
        m_garbage.push(Garbage{moveAny(std::move(r.instance)), m_queueStates.primary.nextSubmitValue()});
        r.instance = {};
      }

      // Swap out the scene
      m_garbage = {};
      std::optional<SceneFile> sceneFile;
      try
      {
        // The std::async's future may forward an exception, e.g. if loading an
        // incompatible scene.
        sceneFile = std::move(m_sceneFuture.get());
      }
      catch(const std::exception& e)
      {
        LOGE("Error loading scene: %s", e.what());
        m_sceneLoadError = e.what();
      }
      if(sceneFile.has_value())
      {
        RenderableScene(std::move(m_scene));  // Destroy the old scene first (not exception safe). TODO: move to garbage instead
        m_scene = RenderableScene{
            std::move(*sceneFile),
            m_config,
            m_context.allocator,
            m_context.commandPool,
            m_queueStates.primary,
            m_glslCompiler,
            m_app,
            m_renderers[m_config.rendererIndex].requiresClas,
        };
        onSceneLoaded();
      }

      m_sceneFuture       = {};
      m_sceneLoadProgress = std::nullopt;

      m_rendererCommon.m_frameAccumIndex = 0;  // reset framebuffer accumulation
    }

    // Streaming stress test with a random camera position each frame
    if(m_config.streamingStress)
    {
      float sceneSize = glm::length(m_scene.view.worldAABB.max - m_scene.view.worldAABB.min);
      CameraManip.setLookat(m_scene.view.worldAABB.center() + (glm::vec3{-0.2F, 0.4F, 0.8F} + glm::ballRand(1.0f)) * sceneSize,
                            m_scene.view.worldAABB.center(), {0.0F, 1.0F, 0.0F});
    }

    if(ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_R)))
    {
      m_reloadShadersRequested = true;
    }

    bool reloadShaders       = m_reloadShadersRequested;
    m_reloadShadersRequested = false;

    // Press "R" to reload traversal and rendering shaders
    if(reloadShaders)
    {
      // Make sure no GPU objects are in use
      NVVK_CHECK(vkQueueWaitIdle(m_context.queue));

      // Reload traversal shaders by re-creating the common renderer
      // TODO: recreate only shader modules and pipelines?
      m_rendererCommon =
          RendererCommon{m_context.allocator, m_glslCompiler, m_context.commandPool, m_context.queue, m_scene.vk};

      // This happens anyway, but for completeness, reset framebuffer
      // accumulation
      m_rendererCommon.m_frameAccumIndex = 0;

      // Reload renderer shaders
      for(auto& r : m_renderers)
        r.instance.reset();
    }

    if(reloadShaders || m_scene.streaming->requiresClas() != m_renderers[m_config.rendererIndex].requiresClas)
    {
      NVVK_CHECK(vkQueueWaitIdle(m_context.queue));

      // Garbage may reference the memory pool owned by m_scene.streaming and
      // must be cleared now. Could make its allocator a weak_ptr.
      m_garbage = {};

      m_scene.streaming.reset();
      m_scene.streaming = makeStreaming(m_context.allocator, m_glslCompiler, m_app, m_scene.view, m_scene.vk,
                                        m_queueStates.primary, m_config, m_renderers[m_config.rendererIndex].requiresClas);
    }

    // Build one batch of cluster acceleration structures for the streaming
    // pipeline.
    {
      nvvk::ProfilerVK::Section timer(*m_profiler, "Build CLAS", cmd);
      m_scene.streaming->buildClasBatch(m_context.allocator, m_queueStates.primary.nextSubmitValue(), false, cmd);
    }

    // Insert scene geometry streaming from the streaming thread before
    // traversal to compute LOD. Keep any unloaded pages around for another
    // frame since commands referencing them may still be in flight.
    {
      nvvk::ProfilerVK::Section              timer(*m_profiler, "Modify Groups", cmd);
      m_scene.streaming->modifyGroups(m_context.allocator, m_scene.view, m_scene.vk.meshPointers, m_garbage,
                                      m_queueStates.primary.nextSubmitValue(), false, cmd,
                                      m_scene.vk.totalResidentClusters, m_scene.vk.totalResidentInstanceClusters);
    }

    // Render the scene. This includes traversing LOD hierarchies to create
    // per-instance LODs. If the renderer is a raytracer, acceleration
    // structures are built during this process. TODO: some rendering commands
    // get recorded into 'cmd' but submitted by the nvvkhl Application so the
    // order here is counterintuitive!
    {
      vkobj::NvtxRange sceneRenderRange("Scene Render");
      RenderParams     renderParams{m_app->getInstance(), m_context, m_rendererCommon, *m_framebuffer,
                                *m_profiler,          m_garbage, m_queueStates};
      m_rendererCommon.cmdUpdateParams(*m_framebuffer, nvh::CameraManipulator::Singleton(),
                                       m_scene.view.maxWorldDiagonalInObjectSpace, cmd);
      m_renderers[m_config.rendererIndex].get()->render(renderParams, m_scene.vk, cmd);
    }

    {
      vkobj::NvtxRange tonemapRange("Tonemap");
      m_framebuffer->cmdTonemap(cmd);
    }

    // Gather any new geometry requests from traversal and send them to the
    // streaming thread. Results will be picked up next frame or when next
    // available.
    {
      vkobj::NvtxRange          makeRequestsRange("Make Requests");
      nvvk::ProfilerVK::Section timer(*m_profiler, "Make Requests", cmd);
      m_scene.streaming->makeRequests(m_scene.vk.allGroupNeededFlags, m_queueStates.primary.nextSubmitValue(), cmd);
    }

    // Record the current batch of staging buffers to reuse once transfer
    // commands have finished. This uses a custom shared_ptr deleter to call the
    // lambda when deleted.
    nvvk::StagingMemoryManager* staging = m_context.allocator->getStaging();
    m_garbage.push(Garbage{.object         = std::shared_ptr<void>(nullptr,
                                                                   [staging, stagingSetId = staging->finalizeResourceSet()](void*) {
                                                             // Release the allocator's staging memory for reuse now that command
                                                             // buffers referencing it have finished.
                                                             vkobj::NvtxRange nvtxRange("releaseResourceSet()");
                                                             staging->releaseResourceSet(stagingSetId);
                                                           }),
                           .semaphoreState = m_queueStates.primary.nextSubmitValue()});

    // Finish the last save before starting the next
    if(m_lastFrameSave.joinable())
    {
      m_lastFrameSave.join();
    }
    if(m_cameraPaths->saveFrames())
    {
      auto saveThread = std::jthread([this, after = m_queueStates.primary.nextSubmitValue()]() {
        after.wait(m_context.device);
        // From nvvkhl::Application::saveScreenShot()
        VkImage        dstImage;
        VkDeviceMemory dstImageMemory;
        {
          vkobj::ImmediateCommandBuffer cmd(m_context.device, m_context.commandPool, m_context.queue);
#if 1
          m_app->imageToRgba8Linear(cmd, m_context.device, m_context.physicalDevice, m_framebuffer->displayLdrImage(),
                                    m_framebuffer->gbuffer().getSize(), dstImage, dstImageMemory);
#else
          // To include UI and overlay in the video frames, expose the swapchian
          // from the app. E.g. just hack in 'public:' to nvvkhl::Application
          m_app->imageToRgba8Linear(cmd, m_context.device, m_context.physicalDevice, m_app->m_swapchain.getNextImage(),
                                    m_framebuffer->gbuffer().getSize(), dstImage, dstImageMemory);
#endif
        }
        m_app->saveImageToFile(m_context.device, dstImage, dstImageMemory, m_framebuffer->size(),
                               fmt::format("frame_{:04}.png", m_cameraPaths->animationFrame()), 98);
        vkUnmapMemory(m_context.device, dstImageMemory);
        vkFreeMemory(m_context.device, dstImageMemory, nullptr);
        vkDestroyImage(m_context.device, dstImage, nullptr);
      });
      m_lastFrameSave = std::move(saveThread);
    }

    {
      VkSemaphoreSubmitInfo signalSubmitInfo = m_queueStates.primary.submitInfoAndAdvance(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
      m_app->addSignalSemaphore(signalSubmitInfo);
    }
  }

  void resize(const glm::uvec2& vpSize)
  {
    // Recreate the framebuffer, keeping the old one alive until the next frame
    // is complete
    m_garbage.push(Garbage{moveAny(std::move(m_framebuffer)), m_queueStates.primary.nextSubmitValue()});
    m_framebuffer = std::make_unique<Framebuffer>(m_context.allocator, m_glslCompiler, m_tonemap, vpSize);

    // DANGER: update renderer references to avoid them becoming dangling
    for(auto& renderer : m_renderers)
    {
      if(renderer.instance)
      {
        RenderParams renderParams{m_app->getInstance(), m_context, m_rendererCommon, *m_framebuffer,
                                  *m_profiler,          m_garbage, m_queueStates};
        renderer.instance->updatedFrambuffer(renderParams);
      }
    }

    // gbuffer is recreated so reset the accumulation count
    m_rendererCommon.m_frameAccumIndex = 0;
  }

private:
  nvvkhl::Application*                          m_app = nullptr;
  Config                                        m_config;
  SynchronizedMemAllocator                      m_memAlloc;
  vkobj::Context                                m_context;
  TimelineQueueContainer                        m_queueStates;
  SampleGlslCompiler                            m_glslCompiler;
  TonemapPipeline                               m_tonemap;
  std::unique_ptr<Framebuffer>                  m_framebuffer;
  RenderableScene                               m_scene;
  std::future<std::optional<SceneFile>>         m_sceneFuture;  // for async loading, optional due to msvc bug fixed in v19.32
  std::optional<TaskProgress>                   m_sceneLoadProgress;
  std::string                                   m_sceneLoadError;  // error message for failed scene loads
  PlotSamples<float, 20>                        m_memoryUsageHistory;
  PlotSamples<uint32_t, 20>                     m_residentGroupsHistory;
  RendererCommon                                m_rendererCommon;
  std::vector<Renderer>                         m_renderers;
  RaytraceConfig                                m_rendererRaytraceConfig;  // external to persist between scene changes
  std::shared_ptr<nvvkhl_copy::ElementProfiler> m_profiler;
  std::shared_ptr<CameraPathsElement>           m_cameraPaths;
  std::queue<Garbage>                           m_garbage;
  fs::path                                      m_cacheDir;
  std::jthread                                  m_lastFrameSave;
  bool                                          m_reloadShadersRequested = false;
  bool                                          m_showSettings           = true;
};

template <typename T>
std::optional<T> envVar(const char* name)
{
  const char* value = std::getenv(name);
  return value ? std::make_optional<T>(value) : std::nullopt;
}

int main(int argc, char** argv)
{
  try
  {
    bool enableValidation = false;
#ifndef NDEBUG
    enableValidation = true;
#endif

    // Paths outside the AppImage, if packaged
    // See: https://docs.appimage.org/packaging-guide/environment-variables.html
    auto outsideExePath = envVar<fs::path>("APPIMAGE").value_or(nvh::getExecutablePath()).parent_path();
    auto currentPath    = envVar<fs::path>("OWD").value_or(fs::current_path());

    // Set log file to write outside AppImage (if packaged) - do this BEFORE any logging!
    {
      fs::path logPath = outsideExePath / (nvh::getExecutablePath().stem().string() + "_log.txt");
      nvprintSetLogFileName(logPath.string().c_str());
    }

    // Parse command line arguments
    fs::path insideExePath = nvh::getExecutablePath().parent_path();  // inside the AppImage, if packaged
    std::vector<std::string> defaultSearchPaths = {
        fs::absolute(insideExePath / PROJECT_DOWNLOAD_RELDIRECTORY).string(),  // regular build
        fs::absolute(insideExePath / "media").string(),                        // install build
    };
    LOGI("Search paths:\n");
    for(auto& path : defaultSearchPaths)
    {
      LOGI("  '%s'\n", path.c_str());
    }

    std::string gltfPath  = nvh::findFile("bunny_v2/bunny.gltf", defaultSearchPaths);
    std::string cacheDir  = currentPath.string();  // or fs::temp_directory_path()?
    bool        printHelp = false;
    nvh::CommandLineParser args("vk_continuous_lod_clusters - a vulkan sample to demo continuous level of detail with ray tracing");
    args.addArgument({"-m", "--mesh"}, &gltfPath, "Mesh filename (*.gltf) or 'generated'");
    args.addArgument({"-c", "--cache-dir"}, &cacheDir, "Directory to keep render cache files. Default is CWD, " + cacheDir);
    args.addArgument({"--validate"}, &enableValidation,
                     "Enable validation layers (may break depending on VK_NV_cluster_acceleration_structure support)");
    args.addArgument({"-h", "--help"}, &printHelp, "Print Help");
    if(!args.parse(argc, argv) || printHelp)
    {
      args.printHelp();
      return printHelp ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if(gltfPath.empty())
    {
      LOGI("Default bunny not found. Falling back to generated scene.\n");
    }
    if(gltfPath == "generated")
    {
      gltfPath = {};  // empty path implies generated scene
    }

    nvvkhl::ApplicationCreateInfo spec;
    spec.name    = PROJECT_NAME " Example";
    spec.vSync   = true;
    spec.useMenu = false;  // Disable menu bar - functionality moved to Settings buttons

    nvvk::ContextCreateInfo vkSetup;
    vkSetup          = nvvk::ContextCreateInfo(enableValidation);
    vkSetup.apiMajor = 1;
    vkSetup.apiMinor = 3;

    vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure{
        .sType                              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext                              = nullptr,
        .accelerationStructure              = VK_FALSE,
        .accelerationStructureCaptureReplay = VK_FALSE,
        .accelerationStructureIndirectBuild = VK_FALSE,
        .accelerationStructureHostCommands  = VK_FALSE,
        .descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE,
    };
    vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelerationStructure);  // To build acceleration structures
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipeline{
        .sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .pNext              = nullptr,
        .rayTracingPipeline = VK_FALSE,
        .rayTracingPipelineShaderGroupHandleCaptureReplay      = VK_FALSE,
        .rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE,
        .rayTracingPipelineTraceRaysIndirect                   = VK_FALSE,
        .rayTraversalPrimitiveCulling                          = VK_FALSE,
    };
    vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rayTracingPipeline);  // To use vkCmdTraceRaysKHR
    vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
    vkSetup.addDeviceExtension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR rayTracingPositionFetch{
        .sType                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
        .pNext                   = nullptr,
        .rayTracingPositionFetch = VK_FALSE,
    };
    vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, false, &rayTracingPositionFetch);
    vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, false);
    vkSetup.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, false);  // for debugPrintfEXT and debug label markers, causes nvvk::Context::initDebugUtils() to be called
    if(enableValidation)
    {
      vkSetup.addDeviceExtension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, false);  // for debugPrintfEXT
    }

    VkPhysicalDeviceMeshShaderFeaturesNV meshShaderStructure{
        .sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV,
        .pNext      = nullptr,
        .taskShader = VK_FALSE,
        .meshShader = VK_FALSE,
    };
    vkSetup.addDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME, false, &meshShaderStructure);

    // Add cluster acceleration structure extension
    static VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clusterAccelerationStructureFeatures = {
        .sType                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV,
        .pNext                        = nullptr,
        .clusterAccelerationStructure = VK_TRUE,
    };
    vkSetup.addDeviceExtension(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &clusterAccelerationStructureFeatures,
                               2 /* VK_NV_CLUSTER_ACCELERATION_STRUCTURE_SPEC_VERSION */);

    // Required for GPU buffer download with
    // nvvk::StagingMemoryManager::cmdFromAddressNV()
    static VkPhysicalDeviceCopyMemoryIndirectFeaturesNV copyMemoryIndirectFeatures = {
        .sType        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES_NV,
        .pNext        = nullptr,
        .indirectCopy = VK_FALSE,
    };
    vkSetup.addDeviceExtension(VK_NV_COPY_MEMORY_INDIRECT_EXTENSION_NAME, false, &copyMemoryIndirectFeatures);

    // Surface extensions
    nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    // Request an extra compute queue for async scene loading. A transfer queue
    // would cover most things, but the scene may use a compute shader if
    // it's procedurally generated.
    vkSetup.addRequestedQueue(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT);

    // #DLSS_RR: Add required NGX instance extensions
    ngx::FeatureDiscovery dlssDiscoveryInfo(0xdeadbeefull, nvh::getExecutablePath().parent_path().wstring(),
                                            NVSDK_NGX_Feature_RayReconstruction,
                                            {
                                                fs::absolute(DLSS_RELPATH_FROM_INSTALL).lexically_normal().wstring(),
                                                fs::absolute(DLSS_RELPATH_FROM_SOURCE).lexically_normal().wstring(),
                                                fs::absolute(DLSS_RELPATH_FROM_BINARY).lexically_normal().wstring(),
                                            });
    for(const auto& ext : ngx::requiredInstanceExtensions(dlssDiscoveryInfo))
    {
      vkSetup.addInstanceExtension(ext.extensionName, false);
    }

    {
      // nvpro_core doesn't provide a way to add device extensions after instance
      // creation, so we have to recreate a temporary one first
      nvvk::Context tmpCtx;
      if(!tmpCtx.init(vkSetup))
      {
        return EXIT_FAILURE;
      }
      // #DLSS_RR: Query and log required NGX device extensions
      // Note: These should have been added by NVIDIA driver during initialization
      for(const auto& ext : ngx::requiredDeviceExtensions(tmpCtx.m_instance, tmpCtx.m_physicalDevice, dlssDiscoveryInfo))
      {
        vkSetup.addDeviceExtension(ext.extensionName, false);
      }
      tmpCtx.deinit();
    }

    // Create the main vulkan context
    nvvk::Context vkctx;
    if(!vkctx.init(vkSetup))
    {
      return EXIT_FAILURE;
    }

    // Check the extension exists
    if(clusterAccelerationStructureFeatures.clusterAccelerationStructure != VK_TRUE)
    {
      LOGE("ERROR: The Cluster Acceleration Structure feature is not supported by the loaded vulkan implementation");
      return EXIT_FAILURE;
    }

    // Create the async load queue
    nvvk::Context::Queue asyncLoadQueue = vkctx.createQueue(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, "asyncLoadQueue");

    // Initialize DLSS-RR. This sets global state and cleans up on destruction.
    ngx::ScopedInit ngx(dlssDiscoveryInfo, vkctx.m_instance, vkctx.m_physicalDevice, vkctx.m_device,
                        vkGetInstanceProcAddr, vkGetDeviceProcAddr);

    // Fill app context parameters
    spec.instance       = vkctx.m_instance;
    spec.device         = vkctx.m_device;
    spec.physicalDevice = vkctx.m_physicalDevice;
    spec.queues         = {
        {vkctx.m_queueGCT.familyIndex, vkctx.m_queueGCT.queueIndex, vkctx.m_queueGCT.queue},
        {vkctx.m_queueC.familyIndex, vkctx.m_queueC.queueIndex, vkctx.m_queueC.queue},
        {vkctx.m_queueT.familyIndex, vkctx.m_queueT.queueIndex, vkctx.m_queueT.queue},
        {asyncLoadQueue.familyIndex, asyncLoadQueue.queueIndex, asyncLoadQueue.queue},
    };

    // UI default docking
    spec.dockSetup = [](ImGuiID viewportID) {
      // Split main area to have left panel for UI
      ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.2F, nullptr, &viewportID);

      // Settings window takes most of the left panel
      ImGui::DockBuilderDockWindow("Settings", settingID);

      // Profiler below
      ImGuiID profilerID = ImGui::DockBuilderSplitNode(settingID, ImGuiDir_Down, 0.3F, nullptr, &settingID);
      ImGui::DockBuilderDockWindow("Profiler", profilerID);

      // Camera Paths at bottom
      ImGuiID cameraPathsID = ImGui::DockBuilderSplitNode(profilerID, ImGuiDir_Down, 0.3F, nullptr, &profilerID);
      ImGui::DockBuilderDockWindow("Camera Paths", cameraPathsID);
    };

    {
      // Create the application
      nvvkhl::Application app(spec);

      // Override ImGui ini file to write outside AppImage (if packaged)
      // HACK: call LoadIniSettingsFromDisk() again to override
      // the one in nvvkhl::Application::init()
      static std::string s_imguiIniFilename = (outsideExePath / (nvh::getExecutablePath().stem().string() + ".ini")).string();
      ImGui::GetIO().IniFilename = s_imguiIniFilename.c_str();
      ImGui::LoadIniSettingsFromDisk(s_imguiIniFilename.c_str());

      // Just because I made one for appimage anyway
      std::optional<Image> icon;
      auto                 iconPath = nvh::findFile("icon.png", defaultSearchPaths);
      if(!iconPath.empty())
        icon = createImage(iconPath, false);
      if(icon)
      {
        GLFWimage image{int(icon->extent.width), int(icon->extent.height),
                        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(icon->data.data()))};
        glfwSetWindowIcon(app.getWindowHandle(), 1, &image);
      }

      // Initialize the large fonts and rebuild the fonts as the application
      // calls ImGUI::CreateContext() and sets up its own fonts
      std::ignore = LargeFont::instance();
      std::ignore = LargeIconFont::instance();
      ImGui::GetIO().Fonts->Build();

      // Add all application elements
      app.addElement(std::make_shared<nvvkhl::ElementCamera>());
      app.addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());  // Window title info
      app.addElement(std::make_shared<SampleAppElement<Sample>>(std::filesystem::path(gltfPath), cacheDir));
      app.run();
    }
  }
  catch(const std::exception& e)
  {
    // Catch-all case. Anything is fatal.
    LOGE("Exception thrown: %s\n", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
