cmake_minimum_required(VERSION 3.20)

# Workaround for CUDA 12.8 with GCC 13 compatibility issues
# Use GCC-12 as CUDA host compiler
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-12)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")

# Load user configuration
if(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
else()
    message(FATAL_ERROR
        "Configuration file not found!\n"
        "Please copy config.cmake.template to config.cmake and customize it:\n"
        "  cp config.cmake.template config.cmake\n"
        "Then edit config.cmake with your hardware parameters.")
endif()

# Set CUDA architecture BEFORE project() command (Ada Lovelace = 86)
# This must be set before enabling CUDA language
set(CMAKE_CUDA_ARCHITECTURES 86)

# Enable C++ and CUDA (CUDA is required for GPU waveform generation)
project(aod-dynamic-server LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "CUDA support enabled (required)")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find ZeroMQ
pkg_check_modules(ZMQ REQUIRED libzmq)

# Spectrum Instrumentation SDK
set(SPECTRUM_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/external/spectrum)
set(SPECTRUM_LIBRARY /usr/lib/x86_64-linux-gnu/libspcm_linux.so)

# Check if Spectrum library exists (REQUIRED)
if(EXISTS ${SPECTRUM_LIBRARY})
    message(STATUS "Found Spectrum library: ${SPECTRUM_LIBRARY}")
    set(SPECTRUM_FOUND TRUE)
else()
    message(FATAL_ERROR "Spectrum library not found at ${SPECTRUM_LIBRARY}\n"
            "Please install the Spectrum Instrumentation SDK or update SPECTRUM_LIBRARY path in CMakeLists.txt")
endif()

# Convert CMake booleans to C++ booleans
if(AWG_EXTERNAL_CLOCK_ENABLED)
    set(AWG_EXTERNAL_CLOCK_ENABLED_CPP "true")
else()
    set(AWG_EXTERNAL_CLOCK_ENABLED_CPP "false")
endif()

if(GPU_ENABLED)
    set(GPU_ENABLED_CPP "true")
else()
    set(GPU_ENABLED_CPP "false")
endif()

# Note: Buffer alignment validation skipped in CMake due to binary literal parsing complexity
# Validation will be done at runtime in C++ code

# Generate configuration header from template
configure_file(
    ${CMAKE_SOURCE_DIR}/src/config.h.in
    ${CMAKE_BINARY_DIR}/config.h
    @ONLY
)

# Include directories
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${ZMQ_INCLUDE_DIRS})
include_directories(${SPECTRUM_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/external/nlohmann)

# Add subdirectories
add_subdirectory(src)

# Print configuration summary
message(STATUS "")
message(STATUS "AOD Dynamic Server Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  ZeroMQ found: ${ZMQ_FOUND}")
message(STATUS "  Spectrum SDK found: ${SPECTRUM_FOUND}")
message(STATUS "")
message(STATUS "Hardware Configuration:")
message(STATUS "  Server port: ${SERVER_PORT}")
message(STATUS "  AWG serial number: ${AWG_SERIAL_NUMBER}")
message(STATUS "  AWG channel mask: ${AWG_CHANNEL_MASK}")
message(STATUS "  AWG sample rate: ${AWG_SAMPLE_RATE} Hz")
message(STATUS "  Max tones: ${AOD_MAX_TONES}")
message(STATUS "")
