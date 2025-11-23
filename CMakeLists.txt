# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22)

project(VKLOD-Sample)
set(SAMPLE_EXECUTABLE VKLOD-Sample)

set(SOURCE_FILES
  src/acceleration_structures.cpp
  src/acceleration_structures.hpp
  src/gltf_view.hpp
  src/imgui_docking_helper.hpp
  src/lod_streaming_jobs.cpp
  src/lod_streaming_jobs.hpp
  src/lod_streaming_scene.cpp
  src/lod_streaming_scene.hpp
  src/lod_traverser.cpp
  src/lod_traverser.hpp
  src/meshops_array_view.h
  src/debug_range_summary.hpp
  src/renderer_common.cpp
  src/renderer_common.hpp
  src/renderer_raytrace.cpp
  src/renderer_raytrace.hpp
  src/renderer_rasterize.cpp
  src/renderer_rasterize.hpp
  src/sample_allocation.hpp
  src/sample_camera_paths.hpp
  src/sample_camera_paths.cpp
  src/sample_glsl_compiler.hpp
  src/sample_image.hpp
  src/sample_producer_consumer.hpp
  src/sample_profiler.hpp
  src/sample_profiler.cpp
  src/sample_raytracing_objects.hpp
  src/sample_vulkan_objects.hpp
  src/sample_vulkan_context.cpp
  src/sample_vulkan_context.hpp
  src/scene.cpp
  src/scene.hpp
  src/nvhiz_vk.cpp
  src/nvhiz_vk.hpp
  src/generate_mesh.cpp
  src/generate_mesh.hpp
  src/mesh_util.hpp
  src/platform_utils.hpp
  src/main.cpp
  )

file(GLOB GLSL_FILES shaders/*.*)
file(GLOB NVPRO_CORE_LEGACY_SHADERS nvpro_core_legacy/nvvkhl/shaders/*.*)
source_group("Shaders" FILES ${GLSL_FILES} ${NVPRO_CORE_LEGACY_SHADERS})

include(FetchContent)

# Find GLFW for window management (needed by imgui)
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
  FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
    OVERRIDE_FIND_PACKAGE
  )
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(glfw)
endif()

# Use vulkan_objects instead of nvpro_core
set(DECODELESS_SEARCH_DEPENDENCIES OFF)
set(DECODELESS_FETCH_DEPENDENCIES ON)
set(BUILD_DECODELESS_TESTING OFF)

set(VULKAN_OBJECTS_FETCH_VMA ON)
set(VULKAN_OBJECTS_FETCH_SHADERC ON)  # continue using the glsl compiler that was used before
set(VULKAN_OBJECTS_FETCH_SLANG OFF CACHE BOOL "")  # We'll use shaderc instead, but don't force it off
set(VULKAN_OBJECTS_FETCH_VVL ON)  # Useful for debugging validation layer errors
add_subdirectory(vulkan_objects)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.92.5-docking
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(imgui)

# Build imgui as a static library
add_library(imgui_static STATIC 
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
  ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
)
target_compile_definitions(imgui_static PUBLIC 
  IMGUI_DEFINE_MATH_OPERATORS
  IMGUI_IMPL_VULKAN_NO_PROTOTYPES
)
target_include_directories(imgui_static PUBLIC 
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_static PUBLIC glfw Vulkan::Headers)
set_property(TARGET imgui_static PROPERTY FOLDER "ThirdParty")
if(MSVC)
  target_compile_options(imgui_static PRIVATE /w)
endif()

# Alias for compatibility with ImGuiFileDialog expecting imgui::imgui
if(NOT TARGET imgui::imgui)
  add_library(imgui::imgui ALIAS imgui_static)
endif()

# The nvpro_core_legacy directory and target contains dependencies copied from
# the archived https://github.com/nvpro-samples/nvpro_core, before porting most
# of the vulkan utilities to vulkan_objects
add_library(nvpro_core_legacy_static STATIC
  nvpro_core_legacy/nvvkhl/tonemap_postprocess.cpp
  nvpro_core_legacy/fileformats/texture_formats.cpp
  nvpro_core_legacy/fileformats/nv_dds.cpp
  nvpro_core_legacy/fileformats/nv_ktx.cpp
  nvpro_core_legacy/imgui/imgui_icon.cpp
)
target_include_directories(nvpro_core_legacy_static PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/nvpro_core_legacy
  ${CMAKE_CURRENT_SOURCE_DIR}/nvpro_core_legacy/third_party/dxh/include/directx
)
target_link_libraries(nvpro_core_legacy_static
  PUBLIC
    vulkan_objects
    glm::glm
    imgui_static
)
target_compile_definitions(nvpro_core_legacy_static PUBLIC
  NVP_SUPPORTS_VULKANSDK
)

# Disable C5246 warning for nvpro_core_legacy (legacy code doesn't use C++20
# aggregate init style)
if(MSVC)
  target_compile_options(nvpro_core_legacy_static PRIVATE /wd5246)
endif()

# Fetch implot
FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot.git
  GIT_TAG v0.17
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(implot)
add_library(implot_static STATIC
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
)
target_compile_definitions(implot_static PRIVATE IMPLOT_IMPLEMENTATION)
target_include_directories(implot_static PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot_static PRIVATE imgui_static)
set_property(TARGET implot_static PROPERTY FOLDER "ThirdParty")

# Fetch ImGuiFileDialog
FetchContent_Declare(
  ImGuiFileDialog
  GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog.git
  GIT_TAG v0.6.8
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(ImGuiFileDialog)
target_link_libraries(ImGuiFileDialog PUBLIC imgui_static)
set_property(TARGET ImGuiFileDialog PROPERTY FOLDER "ThirdParty")

# Fetch args for command line parsing
FetchContent_Declare(
  args
  GIT_REPOSITORY https://github.com/Taywee/args.git
  GIT_TAG 6.4.7
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
)
set(ARGS_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(ARGS_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(args)

# Find NVToolsExt for profiling
find_package(NVToolsExt QUIET)
if(NVToolsExt_FOUND)
  add_definitions(-DNVP_SUPPORTS_NVTOOLSEXT)
endif()

# Download the bunny as the default mesh to load (optional)
set(DOWNLOAD_TARGET_DIR "${CMAKE_BINARY_DIR}/downloaded_resources")
if(NOT EXISTS "${DOWNLOAD_TARGET_DIR}/bunny_v2.zip")
  message(STATUS "Attempting to download bunny_v2.zip (optional default mesh)...")
  file(DOWNLOAD
    "https://developer.download.nvidia.com/ProGraphics/nvpro-samples/bunny_v2.zip"
    "${DOWNLOAD_TARGET_DIR}/bunny_v2.zip"
    STATUS DOWNLOAD_STATUS
    TIMEOUT 10
  )
  list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
  if(NOT DOWNLOAD_RESULT EQUAL 0)
    message(WARNING "Could not download bunny_v2.zip (offline or timeout).")
  endif()
endif()

if(EXISTS "${DOWNLOAD_TARGET_DIR}/bunny_v2.zip" AND NOT EXISTS "${DOWNLOAD_TARGET_DIR}/bunny_v2")
  message(STATUS "Extracting bunny_v2.zip...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_TARGET_DIR}/bunny_v2.zip"
    WORKING_DIRECTORY "${DOWNLOAD_TARGET_DIR}"
    RESULT_VARIABLE EXTRACT_RESULT
  )
  if(NOT EXTRACT_RESULT EQUAL 0)
    message(WARNING "Failed to extract bunny_v2.zip")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Threads REQUIRED)

set(BUILD_NV_CLUSTER_LOD_BUILDER_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_NV_CLUSTER_BUILDER_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(nv_cluster_lod_builder)


# Add cgltf implementation
if(NOT TARGET cgltf_static)
  FetchContent_Declare(
    cgltf
    GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
    GIT_TAG v1.13
    GIT_SHALLOW TRUE
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(cgltf)
  add_library(cgltf_static src/cgltf_impl.cpp)
  target_include_directories(cgltf_static PUBLIC ${cgltf_SOURCE_DIR})
  set_property(TARGET cgltf_static PROPERTY FOLDER "ThirdParty")
endif()

# Add stb_image implementation
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG f1c79c02822848a9bed4315b12c8c8f3761e1296
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(stb)
add_library(stb_image_static src/stb_image_impl.cpp)
target_include_directories(stb_image_static PUBLIC ${stb_SOURCE_DIR})
set_property(TARGET stb_image_static PROPERTY FOLDER "ThirdParty")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${SAMPLE_EXECUTABLE}
  ${SOURCE_FILES}
  ${GLSL_FILES})

if(MSVC)
    target_compile_options(${SAMPLE_EXECUTABLE} PRIVATE
        /W4
        /WX

        # Enable some specific extras
        /w45038 # warning C5038: data member 'member1' will be initialized after data member 'member2'
        /w45219 # warning C5219: implicit conversion from 'int' to 'float', possible loss of data
        /w44265 # warning C4265: 'NVPWindow': class has virtual functions, but its non-trivial destructor is not virtual; instances of this class may not be destructed correctly
        /w44355 # warning C4355: 'this': used in base member initializer list
        /w45246 # warning C5246: '_Elems': the initialization of a subobject should be wrapped in braces

        # Ignore external header warnings from /W4
        /experimental:external /external:W3
        
        # Mark nvpro_core_legacy as external to suppress its warnings
        /external:I ${CMAKE_CURRENT_SOURCE_DIR}/nvpro_core_legacy
        
        # Mark imgui as external to suppress its warnings
        "/external:I$<SHELL_PATH:${imgui_SOURCE_DIR}>"

        # Don't default to buggy parallel builds... like why is this not just on?
        /FS

        # Extra compile error diagnostics
        /diagnostics:caret

        # Static analysis, ignoring external headers
        /analyze /analyze:external-
    )
    target_compile_definitions(${SAMPLE_EXECUTABLE} PRIVATE WIN32_LEAN_AND_MEAN=1 NOMINMAX)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${SAMPLE_EXECUTABLE})
else()
    target_compile_options(${SAMPLE_EXECUTABLE} PRIVATE
        -Wall
        -Wextra
        #-Wpedantic
        -Wconversion
        -Werror

        # Strange false positives/bug?
        $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>
    )

    set_property(SOURCE src/renderer_rasterize.cpp APPEND_STRING PROPERTY COMPILE_OPTIONS
        -Wno-missing-field-initializers
        -Wno-unused-parameter
    )

    target_compile_definitions(${SAMPLE_EXECUTABLE} PRIVATE
        $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>
        # Can't use. Conflicts with std::vector use in the nvpro_core API
        #$<$<CONFIG:Debug>:_GLIBCXX_DEBUG>           # DANGER: ABI breaking!
        #$<$<CONFIG:Debug>:_GLIBCXX_DEBUG_BACKTRACE> # DANGER: ABI breaking!
    )

    # Ignore warnings from external headers
    target_include_directories(${SAMPLE_EXECUTABLE} SYSTEM PRIVATE
      ${vulkan_headers_INCLUDE_DIR}
      ${imgui_SOURCE_DIR}
      )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # for when you write too much code without building
        #target_compile_options(${SAMPLE_EXECUTABLE} PRIVATE -fmax-errors=5)

        # Optional: faster repeat gdb launch at cost of post-build step
        #if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        #    add_custom_command(TARGET ${SAMPLE_EXECUTABLE} POST_BUILD
        #        COMMAND gdb-add-index $<TARGET_FILE:${SAMPLE_EXECUTABLE}>
        #        COMMENT "gdb-add-index ${SAMPLE_EXECUTABLE}"
        #    )
        #endif()
    endif()
endif()

# Compute relative paths to DLSS dynamic libraries with variables from
# FindNGX.cmake
find_package(NGX REQUIRED)
if(NOT TARGET ngx OR NOT DLSS_DLLS)
  message(FATAL_ERROR "Failed to download/find DLSS binaries")
endif()
list(GET DLSS_DLLS 0 _DLSS_FIRST_DLL)
get_filename_component(_DLSS_LIB_DIR "${_DLSS_FIRST_DLL}" DIRECTORY)
get_filename_component(_DLSS_LIB_PARENT_DIR "${_DLSS_LIB_DIR}" DIRECTORY)
file(RELATIVE_PATH DLSS_PARENT_RELPATH_FROM_SOURCE "${CMAKE_SOURCE_DIR}" "${_DLSS_LIB_PARENT_DIR}")

target_compile_definitions(${SAMPLE_EXECUTABLE} PRIVATE
  DLSS_RELPATH_FROM_INSTALL="."
  # Workaround for DLSS_USE_DEVELOP_LIBRARIES. Use dev for Debug builds, rel for others
  DLSS_RELPATH_FROM_SOURCE="${DLSS_PARENT_RELPATH_FROM_SOURCE}/$<IF:$<CONFIG:Debug>,dev,rel>"
  DLSS_RELPATH_FROM_BINARY="$<PATH:RELATIVE_PATH,${_DLSS_LIB_PARENT_DIR}/$<IF:$<CONFIG:Debug>,dev,rel>,$<TARGET_FILE_DIR:${SAMPLE_EXECUTABLE}>>"
  DOWNLOAD_RELPATH_FROM_BINARY="$<PATH:RELATIVE_PATH,${DOWNLOAD_TARGET_DIR},$<TARGET_FILE_DIR:${SAMPLE_EXECUTABLE}>>"
  # Relative paths to shader directories from the binary location (for running from build dir)
  SHADER_DIR_RELPATH_FROM_BINARY="$<PATH:RELATIVE_PATH,${CMAKE_SOURCE_DIR}/shaders,$<TARGET_FILE_DIR:${SAMPLE_EXECUTABLE}>>"
  NVPRO_LEGACY_RELPATH_FROM_BINARY="$<PATH:RELATIVE_PATH,${CMAKE_SOURCE_DIR}/nvpro_core_legacy,$<TARGET_FILE_DIR:${SAMPLE_EXECUTABLE}>>"
  INSTALL_SUBDIRECTORY="media"
  )

target_precompile_headers(${SAMPLE_EXECUTABLE} PRIVATE
  # Standard library
  <algorithm>
  <any>
  <array>
  <atomic>
  <bit>
  <chrono>
  <cmath>
  <condition_variable>
  <cstddef>
  <cstdint>
  <cstdlib>
  <deque>
  <exception>
  <filesystem>
  <format>
  <fstream>
  <functional>
  <future>
  <initializer_list>
  <iomanip>
  <iostream>
  <limits>
  <map>
  <memory>
  <memory_resource>
  <mutex>
  <numeric>
  <optional>
  <ostream>
  <print>
  <queue>
  <ranges>
  <ratio>
  <set>
  <span>
  <sstream>
  <stdexcept>
  <string>
  <string_view>
  <thread>
  <type_traits>
  <unordered_map>
  <unordered_set>
  <utility>
  <variant>
  <vector>
  # Third-party libraries
  <GLFW/glfw3.h>
  <backends/imgui_impl_glfw.h>
  <backends/imgui_impl_vulkan.h>
  <cgltf.h>
  <glm/common.hpp>
  <glm/glm.hpp>
  <glm/gtc/matrix_transform.hpp>
  <glm/gtc/noise.hpp>
  <glm/gtc/random.hpp>
  <glm/gtc/type_ptr.hpp>
  <glm/gtx/euler_angles.hpp>
  <glm/gtx/hash.hpp>
  <glm/gtx/string_cast.hpp>
  <imgui.h>
  <imgui_internal.h>
  <implot.h>
  <meshops_array_view.h>
  <shaderc/shaderc.hpp>
  <stb_image.h>
  <vk_mem_alloc.h>
  <vulkan/vulkan_core.h>
  # vko (comment out if editing these frequently)
  <vko/allocator.hpp>
  <vko/bindings.hpp>
  <vko/bound_buffer.hpp>
  <vko/bound_image.hpp>
  <vko/command_recording.hpp>
  <vko/functions.hpp>
  <vko/glfw_objects.hpp>
  <vko/handles.hpp>
  <vko/imgui_objects.hpp>
  <vko/implot_objects.hpp>
  <vko/nv_ngx.hpp>
  <vko/query_pool.hpp>
  <vko/ray_tracing.hpp>
  <vko/shaderc_compiler.hpp>
  <vko/shortcuts.hpp>
  #<vko/staging_memory.hpp>
  #<vko/swapchain.hpp>
  <vko/timeline_queue.hpp>
  )

# Fetch decodeless dependencies
FetchContent_Declare(
  decodeless_offset_ptr
  GIT_REPOSITORY https://github.com/decodeless/offset_ptr.git
  GIT_TAG 38ceefc6ce63fb4667cd207424b1277c3eed5f8d
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE)
FetchContent_MakeAvailable(decodeless_offset_ptr)
FetchContent_Declare(
  decodeless_allocator
  GIT_REPOSITORY https://github.com/decodeless/allocator.git
  GIT_TAG 035b142ff5cb016262ec1ce505e160caa5c30660
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE)
FetchContent_MakeAvailable(decodeless_allocator)
FetchContent_Declare(
  decodeless_mappedfile
  GIT_REPOSITORY https://github.com/decodeless/mappedfile.git
  GIT_TAG d3425b2508e76ed7a09e55f0c6b580b4107f04db
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE)
FetchContent_MakeAvailable(decodeless_mappedfile)
FetchContent_Declare(
  decodeless_writer
  GIT_REPOSITORY https://github.com/decodeless/writer.git
  GIT_TAG 6f9b747d7671afe2c7abd1d48c9d084402aec1d8
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE)
FetchContent_MakeAvailable(decodeless_writer)

# Fetch GLM for math
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.2
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(glm)

# Download https://github.com/scottt/debugbreak
set(DEBUGBREAK_DIR "${CMAKE_BINARY_DIR}/debugbreak")
set(DEBUGBREAK_HEADER "${DEBUGBREAK_DIR}/debugbreak.h")
file(MAKE_DIRECTORY "${DEBUGBREAK_DIR}")
if(NOT EXISTS "${DEBUGBREAK_HEADER}")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/scottt/debugbreak/5dcbe41d2bd4712c8014aa7e843723ad7b40fd74/debugbreak.h"
        "${DEBUGBREAK_HEADER}"
        EXPECTED_HASH SHA256=f691efbd23848c927d9ee98d7b5264a86cf3a36607633b08bd6936f44e2b71c8
        TLS_VERIFY ON
    )
endif()

target_include_directories(${SAMPLE_EXECUTABLE} PRIVATE src shaders nvpro_core_legacy nvpro_core_legacy/nvtx3 ${DEBUGBREAK_DIR})

target_link_libraries(${SAMPLE_EXECUTABLE}
  vulkan_objects
  vma_static
  nv_cluster_lod_builder
  cgltf_static
  stb_image_static
  nvpro_core_legacy_static
  imgui_static
  implot_static
  ImGuiFileDialog
  glfw
  glm::glm
  args
  decodeless::writer
  decodeless::offset_ptr
  ngx
  Threads::Threads)

if(NVToolsExt_FOUND)
  target_link_libraries(${SAMPLE_EXECUTABLE} NVToolsExt::NVToolsExt)
endif()

if(TARGET vulkan_objects_glfw_xcb)
  target_link_libraries(${SAMPLE_EXECUTABLE} vulkan_objects_glfw_xcb) # contains workaround for native XCB
endif()

target_compile_definitions(${SAMPLE_EXECUTABLE} PRIVATE
  GLM_FORCE_CTOR_INIT  # safety
  GLM_ENABLE_EXPERIMENTAL
  )

# Install shaders. Unlike some samples, these are loaded at runtime and not embedded in the binary
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/_install" CACHE PATH "Install path prefix")
install(FILES ${GLSL_FILES} CONFIGURATIONS Release DESTINATION "bin_${CMAKE_SYSTEM_PROCESSOR}/GLSL_${SAMPLE_EXECUTABLE}")
install(FILES ${GLSL_FILES} CONFIGURATIONS Debug DESTINATION "bin_${CMAKE_SYSTEM_PROCESSOR}_debug/GLSL_${SAMPLE_EXECUTABLE}")
install(FILES ${DLSS_DLLS} CONFIGURATIONS Release DESTINATION "bin_${CMAKE_SYSTEM_PROCESSOR}")
install(FILES ${DLSS_DLLS} CONFIGURATIONS Debug DESTINATION "bin_${CMAKE_SYSTEM_PROCESSOR}_debug")
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/doc/icon.png CONFIGURATIONS Release DESTINATION "bin_${CMAKE_SYSTEM_PROCESSOR}/media")
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/doc/icon.png CONFIGURATIONS Debug DESTINATION "bin_${CMAKE_SYSTEM_PROCESSOR}_debug/media")

# Debug in the source directory to avoid having a set of relative shader include
# paths just for debugging with visual studio
set_property(TARGET ${SAMPLE_EXECUTABLE} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Custom rule to create a more portable AppImage
# To build: cmake --build <build_dir> --target appimage
if(UNIX AND CMAKE_BUILD_TYPE STREQUAL "Release")
  set(APPIMAGE_OUTPUT "${CMAKE_BINARY_DIR}/${SAMPLE_EXECUTABLE}.AppImage")
  set(APPIMAGE_TOOL "${CMAKE_BINARY_DIR}/appimagetool-x86_64.AppImage")
  set(APPDIR "${CMAKE_BINARY_DIR}/AppDir")

  # Media files for dependency tracking
  file(GLOB_RECURSE MEDIA_FILES "${DOWNLOAD_TARGET_DIR}/bunny_v2/*")

  # Generate an AppRun script
  set(APPRUN_SCRIPT "${CMAKE_BINARY_DIR}/AppRun")
  file(WRITE "${APPRUN_SCRIPT}" "#!/bin/sh\nexport VK_ADD_LAYER_PATH=\".\${VK_ADD_LAYER_PATH:+:\$VK_ADD_LAYER_PATH}\"\ncd \"$(dirname \"$0\")\"\n./${SAMPLE_EXECUTABLE} \"$@\"\n")
  file(CHMOD "${APPRUN_SCRIPT}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

  # Download appimagetool
  if(NOT EXISTS "${APPIMAGE_TOOL}")
    message(STATUS "Downloading appimagetool...")
    file(DOWNLOAD
         "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
         "${APPIMAGE_TOOL}"
         SHOW_PROGRESS)
    file(CHMOD "${APPIMAGE_TOOL}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
  endif()

  # Create the AppImage
  add_custom_command(
      OUTPUT "${APPIMAGE_OUTPUT}"
      # Clean and create AppDir
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${APPDIR}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${APPDIR}"

      # Copy executable, shaders, libraries, media
      COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${SAMPLE_EXECUTABLE}>" "${APPDIR}/"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${APPDIR}/shaders"
      COMMAND ${CMAKE_COMMAND} -E copy ${GLSL_FILES} "${APPDIR}/shaders/"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${APPDIR}/nvpro_core_legacy/nvvkhl/shaders"
      COMMAND ${CMAKE_COMMAND} -E copy ${NVPRO_CORE_LEGACY_SHADERS} "${APPDIR}/nvpro_core_legacy/nvvkhl/shaders/"
      COMMAND ${CMAKE_COMMAND} -E copy ${DLSS_DLLS} "${APPDIR}/"
      COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:vvl>" "${APPDIR}/"  # 10mb extra for optional validation layers. worth it?
      COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE_DIR:vvl>/VkLayer_khronos_validation.json" "${APPDIR}/"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${APPDIR}/media"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/doc/icon.png" "${APPDIR}/media/"
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${DOWNLOAD_TARGET_DIR}/bunny_v2" "${APPDIR}/media/bunny_v2"

      # Copy launch script and metadata
      COMMAND ${CMAKE_COMMAND} -E copy "${APPRUN_SCRIPT}" "${APPDIR}/AppRun"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${APPDIR}/usr/share/applications/"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/package/VKLOD-Sample.desktop" "${APPDIR}/usr/share/applications/com.github.pknowles.VKLOD-Sample.desktop"
      # symlink as per the docs https://docs.appimage.org/reference/appdir.html#usr-subdirectory
      COMMAND ${CMAKE_COMMAND} -E create_symlink "usr/share/applications/com.github.pknowles.VKLOD-Sample.desktop" "${APPDIR}/com.github.pknowles.VKLOD-Sample.desktop"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${APPDIR}/usr/share/metainfo/"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/package/VKLOD-Sample.appdata.xml" "${APPDIR}/usr/share/metainfo/com.github.pknowles.VKLOD-Sample.appdata.xml"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/doc/icon.png" "${APPDIR}/.DirIcon"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/doc/icon.svg" "${APPDIR}/${SAMPLE_EXECUTABLE}.svg"

      # Make the AppImage package and clean up
      COMMAND "${APPIMAGE_TOOL}" "${APPDIR}" "${APPIMAGE_OUTPUT}"
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${APPDIR}"

      DEPENDS ${SAMPLE_EXECUTABLE}
              ${GLSL_FILES}
              ${DLSS_DLLS}
              "${APPRUN_SCRIPT}"
              "${CMAKE_SOURCE_DIR}/doc/icon.svg"
              "${CMAKE_SOURCE_DIR}/doc/icon.png"
              "${CMAKE_SOURCE_DIR}/package/VKLOD-Sample.desktop"
              "${CMAKE_SOURCE_DIR}/package/VKLOD-Sample.appdata.xml"
              ${MEDIA_FILES}
      COMMENT "Creating ${APPIMAGE_OUTPUT}"
  )

  # appimage target that depends on the output file
  add_custom_target(appimage
      DEPENDS "${APPIMAGE_OUTPUT}"
  )
endif()

