# Poisson Texture Tiling

## Requirements
- **OpenCV** (4.13.0_6)
- **CMake** (4.2.3)
- **GNU Make** (3.81)

This project has been developed and tested only with the versions above.

### Usage
```bash
# clone
git clone --recursive https://github.com/poisson-texture-tiling
cd poisson-texture-tiling

# build
mkdir build
cmake -S . -B build/
cmake --build build/

# execute
./build/poisson_texture_tiling
```