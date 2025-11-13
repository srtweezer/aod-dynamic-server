# Source files for the server executable
set(SERVER_SOURCES
    main.cpp
    server.cpp
    config.cpp
    awg_interface.cpp
)

# Create executable
add_executable(aod-server ${SERVER_SOURCES})

# Link libraries
target_link_libraries(aod-server
    aod_proto
    ${Protobuf_LIBRARIES}
    ${ZMQ_LIBRARIES}
    yaml-cpp
    pthread
)

# Include directories
target_include_directories(aod-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ZMQ_INCLUDE_DIRS}
)

# Install target
install(TARGETS aod-server DESTINATION bin)

# Set output directory
set_target_properties(aod-server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
