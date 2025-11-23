# Source files for the server executable
set(SERVER_SOURCES
    main.cpp
    server.cpp
    awg_interface.cpp
    shared_memory.cpp
    cuda/gpu_buffers.cu
)

# Create executable
add_executable(aod-server ${SERVER_SOURCES})

# Set CUDA properties
set_target_properties(aod-server PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(aod-server
    ${ZMQ_LIBRARIES}
    pthread
    rt  # POSIX shared memory
)

# Link Spectrum library (required)
target_link_libraries(aod-server ${SPECTRUM_LIBRARY})

# Include directories
target_include_directories(aod-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ZMQ_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/external
)

# Install target
install(TARGETS aod-server DESTINATION bin)

# Set output directory
set_target_properties(aod-server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
