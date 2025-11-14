# Source files for the server executable
set(SERVER_SOURCES
    main.cpp
    server.cpp
    awg_interface.cpp
)

# Create executable
add_executable(aod-server ${SERVER_SOURCES})

# Link libraries
target_link_libraries(aod-server
    aod_proto
    ${Protobuf_LIBRARIES}
    ${ZMQ_LIBRARIES}
    pthread
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
