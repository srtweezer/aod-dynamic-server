# Protobuf compilation

# Find protobuf compiler
find_package(Protobuf REQUIRED)

# Define proto file
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/aod_server.proto)

# Set output files
set(PROTO_SRCS ${CMAKE_CURRENT_BINARY_DIR}/aod_server.pb.cc)
set(PROTO_HDRS ${CMAKE_CURRENT_BINARY_DIR}/aod_server.pb.h)

# Custom command to generate C++ code from proto files
add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR}
         -I ${CMAKE_CURRENT_SOURCE_DIR}
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating C++ protocol buffer code from aod_server.proto"
    VERBATIM
)

# Create library from generated files
add_library(aod_proto ${PROTO_SRCS} ${PROTO_HDRS})

# Link protobuf library
target_link_libraries(aod_proto ${Protobuf_LIBRARIES})

# Make generated headers available
target_include_directories(aod_proto PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIRS}
)

# Print generated files for debugging
message(STATUS "Generated Protobuf files:")
message(STATUS "  Sources: ${PROTO_SRCS}")
message(STATUS "  Headers: ${PROTO_HDRS}")
