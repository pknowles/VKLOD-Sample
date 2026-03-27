#!/bin/bash

# Script to build and test vk_unified_memory
# Checks for VULKAN_SDK, configures and builds, then runs screenshot test

set -e

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

# Configure with CMake (Debug build)
echo "Configuring CMake (Debug)..."
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build
echo "Building..."
cmake --build . -j$(nproc)

# Go back to project root
cd ..

# Run screenshot test
echo "Running screenshot test..."
./screenshot.sh "$@"

echo "Build and test complete!"
