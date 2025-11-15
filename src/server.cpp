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

        case Request::kInitialize:
            response = handleInitialize(request.initialize());
            break;

        case Request::kStop:
            response = handleStop(request.stop());
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

Response AODServer::handleInitialize(const InitializeRequest& request) {
    std::cout << "[Server] Initialize command received" << std::endl;

    // Extract amplitudes
    std::vector<int32> amplitudes_mv;
    for (int i = 0; i < request.channel_amplitudes_mv_size(); i++) {
        amplitudes_mv.push_back(request.channel_amplitudes_mv(i));
    }

    std::cout << "[Server] Amplitudes (mV): [";
    for (size_t i = 0; i < amplitudes_mv.size(); i++) {
        std::cout << amplitudes_mv[i];
        if (i < amplitudes_mv.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Create command for AWG thread
    WaveformCommand cmd;
    cmd.type = AWGCommandType::INITIALIZE;
    cmd.initialize_data = std::make_shared<WaveformCommand::InitializeData>();
    cmd.initialize_data->amplitudes_mv = amplitudes_mv;

    // Queue command and wait for completion (1 second timeout)
    CommandResult result = awg_->queueCommandAndWait(cmd, 1000);

    // Create response
    Response response;
    InitializeResponse* init_response = response.mutable_initialize();
    init_response->set_success(result.success);
    init_response->set_error_message(result.error_message);

    if (result.success) {
        std::cout << "[Server] Initialize completed successfully" << std::endl;
    } else {
        std::cout << "[Server] Initialize failed: " << result.error_message << std::endl;
    }

    return response;
}

Response AODServer::handleStop(const StopRequest& request) {
    std::cout << "[Server] Stop command received" << std::endl;

    // Create command for AWG thread
    WaveformCommand cmd;
    cmd.type = AWGCommandType::STOP;

    // Queue command and wait for completion (1 second timeout)
    CommandResult result = awg_->queueCommandAndWait(cmd, 1000);

    // Create response
    Response response;
    StopResponse* stop_response = response.mutable_stop();
    stop_response->set_success(result.success);
    stop_response->set_error_message(result.error_message);

    if (result.success) {
        std::cout << "[Server] Stop completed successfully" << std::endl;
    } else {
        std::cout << "[Server] Stop failed: " << result.error_message << std::endl;
    }

    return response;
}

} // namespace aod
