/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Pyarelal Knowles
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdlib>
#include <filesystem>
#include <optional>

#if _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <climits>  // for PATH_MAX
#include <mach-o/dyld.h>
#include <stdexcept>  // for std::runtime_error
#endif

namespace fs = std::filesystem;

// Get environment variable as optional<T>
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

// Get the path to the current executable (cross-platform)
inline fs::path executablePath()
{
#if _WIN32
  // Windows: Use GetModuleFileName
  wchar_t path[MAX_PATH];
  GetModuleFileNameW(NULL, path, MAX_PATH);
  return fs::canonical(path);
#elif defined(__APPLE__)
  // macOS: Use _NSGetExecutablePath
  char     path[PATH_MAX];
  uint32_t size = sizeof(path);
  if(_NSGetExecutablePath(path, &size) == 0)
  {
    return fs::canonical(path);
  }
  throw std::runtime_error("Failed to get executable path");
#else
  // Linux/Unix: Use /proc/self/exe symlink
  return fs::canonical("/proc/self/exe");
#endif
}
