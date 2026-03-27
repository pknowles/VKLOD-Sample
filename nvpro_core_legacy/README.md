# Legacy nvpro_core Components

This directory contains a small selection of components copied from the
deprecated [`nvpro_core`](https://github.com/nvpro-samples/nvpro_core) library
and partially ported to use
[`vulkan_objects`](https://github.com/pknowles/vulkan_objects). Many objects are
still not move-safe, require delayed initialization and explicit destruction.

- `fileformats` and `third_party/dxh` for texture loading
- `imgui` for an icon font
- `nvh` and `nvtx3` for NVTX marker definitions (should prob be using upstream)
- `nvvkhl` for the embedded roboto font, tonemapper and shader fragments
- `third_party/tinygltf/json.hpp` for a json parser
