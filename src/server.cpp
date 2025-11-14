#include "server.h"
#include "awg_interface.h"
#include <config.h>
#include <iostream>
#include <chrono>

namespace aod {

AODServer::AODServer(std::shared_ptr<AWGInterface> awg)
    : awg_(awg), running_(false) {
}

AODServer::~AODServer() {
    stop();
}

bool AODServer::initialize() {
    using namespace aod::config;

    try {
        // Create ZMQ context
        context_ = std::make_unique<zmq::context_t>(1);

        // Create REP socket
        socket_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::rep);

        // Build bind address
        std::string bind_addr = std::string(SERVER_BIND_ADDRESS) + ":" + std::to_string(SERVER_PORT);

        // Bind socket
        socket_->bind(bind_addr);

        std::cout << "[Server] Listening on " << bind_addr << std::endl;
        return true;

    } catch (const zmq::error_t& e) {
        std::cerr << "[Server] ZMQ error during initialization: " << e.what() << std::endl;
        return false;
    }
}

void AODServer::run() {
    running_ = true;
    std::cout << "[Server] Starting server loop..." << std::endl;

    while (running_) {
        try {
            // Receive request
            zmq::message_t zmq_request;
            auto result = socket_->recv(zmq_request, zmq::recv_flags::none);

            if (!result) {
                continue;  // No message received
            }

            // Parse protobuf request
            Request request;
            if (!request.ParseFromArray(zmq_request.data(), zmq_request.size())) {
                std::cerr << "[Server] Failed to parse protobuf request" << std::endl;
                continue;
            }

            // Handle request
            Response response = handleRequest(request);

            // Serialize response
            std::string response_data;
            if (!response.SerializeToString(&response_data)) {
                std::cerr << "[Server] Failed to serialize protobuf response" << std::endl;
                continue;
            }

            // Send response
            zmq::message_t zmq_response(response_data.size());
            memcpy(zmq_response.data(), response_data.data(), response_data.size());
            socket_->send(zmq_response, zmq::send_flags::none);

        } catch (const zmq::error_t& e) {
            if (e.num() == ETERM) {
                // Context was terminated, exit gracefully
                break;
            }
            std::cerr << "[Server] ZMQ error: " << e.what() << std::endl;
        }
    }

    std::cout << "[Server] Server loop stopped" << std::endl;
}

void AODServer::stop() {
    if (running_) {
        std::cout << "[Server] Stopping server..." << std::endl;
        running_ = false;

        // Close socket and context
        if (socket_) {
            socket_->close();
        }
        if (context_) {
            context_->close();
        }
    }
}

Response AODServer::handleRequest(const Request& request) {
    Response response;

    switch (request.command_case()) {
        case Request::kPing:
            response = handlePing(request.ping());
            break;

        case Request::COMMAND_NOT_SET:
            std::cerr << "[Server] Received request with no command set" << std::endl;
            break;

        default:
            std::cerr << "[Server] Unknown command type" << std::endl;
            break;
    }

    return response;
}

Response AODServer::handlePing(const PingRequest& request) {
    // Get current time in nanoseconds
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    // Create response
    Response response;
    PingResponse* ping_response = response.mutable_ping();
    ping_response->set_timestamp_ns(nanoseconds);

    std::cout << "[Server] Ping received, responding with timestamp: " << nanoseconds << std::endl;

    return response;
}

} // namespace aod
