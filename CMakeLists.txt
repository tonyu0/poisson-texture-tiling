cmake_minimum_required(VERSION 3.5)
project(PoissonTextureTiling
    VERSION 0.1.0
    DESCRIPTION "seamless tiling using poisson blending"
    HOMEPAGE_URL "https://github.com/tonyu0/poisson-texture-tiling"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)

# Eigen (header only library)
add_subdirectory(external/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(poisson_texture_tiling src/main.cpp)

target_link_libraries(poisson_texture_tiling PRIVATE ${OpenCV_LIBS})