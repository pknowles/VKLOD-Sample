# FindNGX.cmake
#
# Downloads the DLSS-RR SDK. Link with the `ngx` target, then copy `DLSS_DLLS`
# to the output directory. In the future we might add more optional components,
# but you'll still only need to link against ngx -- just copy different DLLs
# into the output.
#
# Usage:
# ```cmake
# find_package(NGX REQUIRED)
# target_link_libraries(${PROJECT_NAME} PRIVATE ngx)
# copy_to_runtime_and_install(${PROJECT_NAME} FILES ${DLSS_DLLS})
# ```
#
# Options (all cached):
# - DLSS_VERSION: The SDK version. Defaults to "310.4.0".
#     While you can change this per-project, we recommend changing it in this
#     file instead so that DLSS updates apply to all nvpro-samples uniformly.
# - DLSS_USE_DEVELOP_FILES: Defaults to OFF.
#     Whether to use the dev/ DLL/SO files from DLSS instead of rel/.
#     The dev/ files have a debug overlay (on Windows, press Ctrl-Alt-F12 on to
#     enable the overlay and Ctrl-Alt-F11 to switch between views), but
#     shouldn't be shipped; they also have debug text and may print additional
#     messages to the console.
# - NGX_USE_STATIC_MSVCRT: Defaults to OFF.
#     Whether to use NGX libraries linked for the static MSVC runtime library.
#     Has no effect on Linux.
#
# Defines the following variables:
# - DLSS_ROOT: Path to the DLSS SDK root directory.
# - DLSS_DLLS: DLL/SO files used by DLSS. Because these are loaded at
#     runtime, you'll need to copy them to the output using
#       copy_to_runtime_and_install(${PROJECT_NAME} FILES ${DLSS_DLLS}) .
# - NGX_INCLUDE_DIR (cached): NGX SDK include directory.
#
# And the following target:
# - ngx: IMPORTED library for NGX.
#     If multiple versions of NGX are pulled, first one wins.

#-------------------------------------------------------------------------------
# Download the DLSS SDK using standard CMake functions.
set(DLSS_VERSION "310.4.0" CACHE STRING "DLSS version to download from https://github.com/NVIDIA/DLSS")
set(_DLSS_URL "https://github.com/NVIDIA/DLSS/archive/refs/tags/v${DLSS_VERSION}.zip")
set(_DLSS_TARGET_DIR "${CMAKE_BINARY_DIR}/_deps/DLSSRR-${DLSS_VERSION}")
set(_DLSS_ZIP_FILE "${_DLSS_TARGET_DIR}/DLSS-${DLSS_VERSION}.zip")
set(_DLSS_EXPECTED_ROOT "${_DLSS_TARGET_DIR}/DLSS-${DLSS_VERSION}")

# Check if we already have the package downloaded and extracted
# Use absolute path to avoid any resolution issues
set(_DLSS_INCLUDE_CHECK "${_DLSS_EXPECTED_ROOT}/include")
if(EXISTS "${_DLSS_INCLUDE_CHECK}")
  message(STATUS "Found existing DLSS-RR (v${DLSS_VERSION}) at '${_DLSS_EXPECTED_ROOT}'")
else()
  # Path doesn't exist - need to download
  if(NOT EXISTS "${_DLSS_ZIP_FILE}")
    message(STATUS "Downloading DLSS-RR (v${DLSS_VERSION}) from ${_DLSS_URL}")
    file(DOWNLOAD ${_DLSS_URL} "${_DLSS_ZIP_FILE}"
         SHOW_PROGRESS
         STATUS _DOWNLOAD_STATUS)
    list(GET _DOWNLOAD_STATUS 0 _DOWNLOAD_STATUS_CODE)
    list(GET _DOWNLOAD_STATUS 1 _DOWNLOAD_STATUS_MESSAGE)
    if(NOT _DOWNLOAD_STATUS_CODE EQUAL 0)
      file(REMOVE "${_DLSS_ZIP_FILE}")
      message(FATAL_ERROR "Failed to download DLSS: ${_DOWNLOAD_STATUS_MESSAGE}")
    endif()
  endif()
  
  # Extract the zip file
  message(STATUS "Extracting DLSS-RR (v${DLSS_VERSION}) from ${_DLSS_ZIP_FILE}")
  file(ARCHIVE_EXTRACT
       INPUT "${_DLSS_ZIP_FILE}"
       DESTINATION "${_DLSS_TARGET_DIR}")

  message(STATUS "Using downloaded DLSS-RR (v${DLSS_VERSION}) at '${_DLSS_EXPECTED_ROOT}'")
endif()

set(DLSS_ROOT "${_DLSS_EXPECTED_ROOT}")
set(DLSS_SOURCE_DIR "${_DLSS_TARGET_DIR}")

if (WIN32)
  set(_DLSS_OS "Windows")
  set(_DLSS_BIN_WILDCARD "nvngx_*.dll")
else()
  set(_DLSS_OS "Linux")
  set(_DLSS_BIN_WILDCARD "libnvidia-ngx-*.so.*")
endif()

if(ARCH_PROC MATCHES "^(arm|aarch64)")
  set(_DLSS_ARCH "aarch64")
else()
  set(_DLSS_ARCH "x86_64")
endif()

file(GLOB _DLSS_DLLS_DEV "${DLSS_ROOT}/lib/${_DLSS_OS}_${_DLSS_ARCH}/dev/${_DLSS_BIN_WILDCARD}")
file(GLOB _DLSS_DLLS_REL "${DLSS_ROOT}/lib/${_DLSS_OS}_${_DLSS_ARCH}/rel/${_DLSS_BIN_WILDCARD}")

if(NOT _DLSS_DLLS_DEV)
  message(WARNING "DLSS dev binary not found at ${DLSS_ROOT}/lib/${_DLSS_OS}_${_DLSS_ARCH}/dev/${_DLSS_BIN_WILDCARD}")
endif()
if(NOT _DLSS_DLLS_REL)
  message(WARNING "DLSS dev binary not found at ${DLSS_ROOT}/lib/${_DLSS_OS}_${_DLSS_ARCH}/rel/${_DLSS_BIN_WILDCARD}")
endif()

option(DLSS_USE_DEVELOP_LIBRARIES "Use non-distributable DLSS libraries with a debug overlay. On Windows, press Ctrl-Alt-F12 to enable the debug overlay and Ctrl-Alt-F11 to switch views." OFF)
if(DLSS_USE_DEVELOP_LIBRARIES)
  set(DLSS_DLLS ${_DLSS_DLLS_DEV})
else()
  set(DLSS_DLLS ${_DLSS_DLLS_REL})
endif()

#-------------------------------------------------------------------------------
# Set up the NGX target. These are static libraries we link to that load modules
# like DLSS.
if(NOT TARGET ngx)
  add_library(ngx IMPORTED STATIC GLOBAL)
  
  # Map MinSizeRel and RelWithDebInfo to Release.
  set_property(TARGET ngx APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
  set_property(TARGET ngx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
  set_target_properties(ngx PROPERTIES
    MAP_IMPORTED_CONFIG_MINSIZEREL Release
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
  )

  if (WIN32)
    option(NGX_USE_STATIC_MSVCRT "[Deprecated?]Use NGX libs with static VC runtime (/MT), otherwise dynamic (/MD)" OFF)

    if(NGX_USE_STATIC_MSVCRT)
      set_target_properties(ngx PROPERTIES IMPORTED_IMPLIB_DEBUG ${DLSS_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_s_dbg.lib)
      set_target_properties(ngx PROPERTIES IMPORTED_IMPLIB_RELEASE ${DLSS_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_s.lib)
      set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_DEBUG ${DLSS_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_s_dbg.lib)
      set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_RELEASE ${DLSS_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_s.lib)
    else()
      set_target_properties(ngx PROPERTIES IMPORTED_IMPLIB_DEBUG ${DLSS_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib)
      set_target_properties(ngx PROPERTIES IMPORTED_IMPLIB_RELEASE ${DLSS_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib)
      set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_DEBUG ${DLSS_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib)
      set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_RELEASE ${DLSS_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib)
    endif()
  else ()
    set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_DEBUG ${DLSS_ROOT}/lib/Linux_x86_64/libnvsdk_ngx.a)
    set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_RELEASE ${DLSS_ROOT}/lib/Linux_x86_64/libnvsdk_ngx.a)
  endif()

  set(NGX_INCLUDE_DIR "${DLSS_ROOT}/include" CACHE PATH "NGX SDK include directory.")
  mark_as_advanced(NGX_INCLUDE_DIR)
  target_include_directories(ngx INTERFACE "${NGX_INCLUDE_DIR}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NGX
  REQUIRED_VARS
    NGX_INCLUDE_DIR
)
