# Poisson Texture Tiling

## Overview

This tool converts an input image into a **seamlessly tileable texture**.  

This program contains following three steps to generate the texture.

1. Build the sparse linear system from the input image ( $Ax = b$ )
   
    - Compute a Laplacian from the input image ( $b$ )

    - Construct a sparse Laplacian matrix with periodic boundary conditions ( $A$ )
2. Solve the sparse linear system
3. Adjust the global color offset

This program provides two calculation methods: FFT ( O(NlogN) ) and Direct method using Cholesky decomposition ( O(N^1.5) )

### [A demo using WASM (please give it a try!)](https://tonyu0.github.io/poisson-texture-tiling/)

---

### DEMO
| Input image | Output image |
| ----------- | ------------ |
| <img width="600" height="450" alt="Image" src="https://github.com/user-attachments/assets/9e24e848-ac00-4cd5-be52-451b9f590999" /> | <img width="600" height="450" alt="Image" src="https://github.com/user-attachments/assets/89390dee-ed8b-47e3-8969-0f43d7a0f98e" /> |
| Tiling without poisson | Tiling with poisson |
| <img width="534" height="450" alt="Image" src="https://github.com/user-attachments/assets/76d8c22e-1a74-4467-b581-4c4824fdd1bb" /> | <img width="534" height="450" alt="Image" src="https://github.com/user-attachments/assets/382d510b-6b90-44c3-b3ef-8c87458893ed" /> |

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
