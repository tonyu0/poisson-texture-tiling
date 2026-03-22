
# WASM with OpenCV setup notes

```bash
# Install Emscripten
git clone https://github.com/emscripten-core/emsdk
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh


# Build OpenCV binary for wasm
git clone https://github.com/opencv/opencv.git
cd opencv

# https://github.com/opencv/opencv/tree/4.x/platforms/js
# In addition, 
# - Add -DCMAKE_CXX_STANDARD=17 to emcmake.
# - Change settings of opencv/modules/js/CMakeLists.txt (comment out the line setting "-s DEMANGLE_SUPPORT=1" build option).
# -- It appeared to be handling this option depending on the Emscripten version, but it didn't work properly in my environment.
emcmake python3 /path/to/opencv/platforms/js/build_js.py /path/to/opencv/build_wasm -DCMAKE_CXX_STANDARD=17

cd poisson-texture-tiling/docs

emcc poisson-texture-tiling-wasm.cpp \
-o index.js \
-I../external/eigen \
-I/path/to/opencv/include \
/path/to/opencv/build_wasm/lib/libopencv_photo.a \
/path/to/opencv/build_wasm/lib/libopencv_imgproc.a \
/path/to/opencv/build_wasm/lib/libopencv_core.a \
-s MODULARIZE=1 \
-s EXPORT_NAME='createModule' \
-s EXPORTED_FUNCTIONS='["_malloc", "_free"]' \
-s EXPORTED_RUNTIME_METHODS='["HEAPU8"]' \
-s ALLOW_MEMORY_GROWTH=1 \
--bind

# ALLOW_MEMORY_GROWTH: need a lot of memory when solving poisson equation

```