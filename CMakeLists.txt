cmake_minimum_required(VERSION 3.20)

# Start with C++ only - CUDA will be enabled when needed
project(aod-dynamic-server LANGUAGES CXX)

# Optionally enable CUDA if available
option(ENABLE_CUDA "Enable CUDA support" OFF)

if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    message(STATUS "CUDA support enabled")
else()
    message(STATUS "CUDA support disabled (use -DENABLE_CUDA=ON to enable)")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA architecture (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find required packages
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)

# Find ZeroMQ
pkg_check_modules(ZMQ REQUIRED libzmq)

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${ZMQ_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(proto)
add_subdirectory(src)

# Print configuration summary
message(STATUS "")
message(STATUS "AOD Dynamic Server Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Protobuf version: ${Protobuf_VERSION}")
message(STATUS "  ZeroMQ found: ${ZMQ_FOUND}")
message(STATUS "  yaml-cpp found: ${yaml-cpp_FOUND}")
message(STATUS "")
