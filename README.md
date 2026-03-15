# Poisson Texture Tiling

## Overview

This tool converts an input image into a **seamlessly tileable texture**.  
The generated texture can be tiled infinitely without visible seams.

This program contains following three steps to generate the texture.

1. Build the sparse linear system from the input image ( $Ax = b$ )
    - Compute the gradient-based Laplacian from the input image ( $b$ )
    - Construct a sparse Laplacian matrix with periodic boundary conditions ( $A$ )
2. Solve the sparse linear system
3. Adjust the global color offset

This program uses Eigen::SimplicialLDLT with Eigen::SparseMatrix to solve Poisson equation. 
It seems time complexity is O(N^1.5), where N is the number of pixels in the image.
So make sure not to input too large images.

TODO
- FFT-based Poisson solver

---

## Requirements
- **OpenCV** (4.13.0_6)
- **CMake** (4.2.3)
- **GNU Make** (3.81)

This project has been developed and tested only with the versions above.

---

## Usage
```bash
# clone
git clone --recursive https://github.com/poisson-texture-tiling
cd poisson-texture-tiling

# build
mkdir build
cmake -S . -B build/
cmake --build build/

# execute
./build/poisson_texture_tiling <path-to-target-image>

# output two images, "poisson_texture_tiling_output.png" and "poisson_texture_tiling_output_tiled3x3.png".
# the former is a tileable texture, and the latter is an image showing an example of tiling.
```

---

## Note

**TODO**
write mathematical explanation
