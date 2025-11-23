#include "server.h"
#include "awg_interface.h"
#include <config.h>
#include <iostream>

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
            // Receive first part (JSON metadata)
            zmq::message_t meta_msg;
            auto result = socket_->recv(meta_msg, zmq::recv_flags::none);

            if (!result) {
                continue;  // No message received
            }

            // Parse JSON metadata
            json request;
            try {
                std::string meta_str(static_cast<char*>(meta_msg.data()), meta_msg.size());
                request = json::parse(meta_str);
            } catch (const json::exception& e) {
                std::cerr << "[Server] JSON parse error: " << e.what() << std::endl;
                json error_response = {
                    {"success", false},
                    {"error_message", std::string("JSON parse error: ") + e.what()}
                };
                std::string response_str = error_response.dump();
                socket_->send(zmq::buffer(response_str), zmq::send_flags::none);
                continue;
            }

            // Get command type
            std::string cmd = request.value("command", "");
            std::cout << "[Server] Received command: " << cmd << std::endl;

            // Handle command
            json response;
            try {
                if (cmd == "INITIALIZE") {
                    response = handleInitialize(request);
                } else if (cmd == "START") {
                    response = handleStart(request);
                } else if (cmd == "STOP") {
                    response = handleStop(request);
                } else if (cmd == "WAVEFORM_BATCH") {
                    response = handleWaveformBatch(request, *socket_);
                } else {
                    response = {
                        {"success", false},
                        {"error_message", "Unknown command: " + cmd}
                    };
                }
            } catch (const std::exception& e) {
                response = {
                    {"success", false},
                    {"error_message", std::string("Exception: ") + e.what()}
                };
            }

            // Send response
            std::string response_str = response.dump();
            socket_->send(zmq::buffer(response_str), zmq::send_flags::none);

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

json AODServer::handleInitialize(const json& request) {
    std::cout << "[Server] Initialize command received" << std::endl;

    // Extract amplitudes
    std::vector<int32_t> amplitudes_mv = request["amplitudes_mv"];

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

    // Queue command and wait for completion
    CommandResult result = awg_->queueCommandAndWait(cmd, 1000);

    if (result.success) {
        std::cout << "[Server] Initialize completed successfully" << std::endl;
    } else {
        std::cout << "[Server] Initialize failed: " << result.error_message << std::endl;
    }

    return {
        {"success", result.success},
        {"error_message", result.error_message}
    };
}

json AODServer::handleStart(const json& request) {
    std::cout << "[Server] Start command received" << std::endl;

    // Create command for AWG thread
    WaveformCommand cmd;
    cmd.type = AWGCommandType::START;

    // Queue command and wait for completion
    CommandResult result = awg_->queueCommandAndWait(cmd, 1000);

    if (result.success) {
        std::cout << "[Server] Start completed successfully" << std::endl;
    } else {
        std::cout << "[Server] Start failed: " << result.error_message << std::endl;
    }

    return {
        {"success", result.success},
        {"error_message", result.error_message}
    };
}

json AODServer::handleStop(const json& request) {
    std::cout << "[Server] Stop command received" << std::endl;

    // Create command for AWG thread
    WaveformCommand cmd;
    cmd.type = AWGCommandType::STOP;

    // Queue command and wait for completion
    CommandResult result = awg_->queueCommandAndWait(cmd, 1000);

    if (result.success) {
        std::cout << "[Server] Stop completed successfully" << std::endl;
    } else {
        std::cout << "[Server] Stop failed: " << result.error_message << std::endl;
    }

    return {
        {"success", result.success},
        {"error_message", result.error_message}
    };
}

json AODServer::handleWaveformBatch(const json& request, zmq::socket_t& socket) {
    using namespace aod::config;

    std::cout << "[Server] WaveformBatch command received" << std::endl;

    // Extract metadata (client provides batch_id)
    int batch_id = request["batch_id"];
    int num_timesteps = request["num_timesteps"];
    int num_tones = request["num_tones"];
    std::string trigger_type = request.value("trigger_type", "software");

    // Calculate num_channels from compile-time config (not from client)
    int num_channels = __builtin_popcount(AWG_CHANNEL_MASK);

    std::cout << "[Server] Batch ID: " << batch_id << std::endl;
    std::cout << "[Server] Timesteps: " << num_timesteps << std::endl;
    std::cout << "[Server] Channels: " << num_channels << " (from config)" << std::endl;
    std::cout << "[Server] Tones: " << num_tones << std::endl;
    std::cout << "[Server] Trigger: " << trigger_type << std::endl;

    // Receive 5 array parts
    zmq::message_t arrays[5];
    for (int i = 0; i < 5; i++) {
        auto result = socket.recv(arrays[i], zmq::recv_flags::none);
        if (!result) {
            return {
                {"success", false},
                {"error_message", "Failed to receive array part " + std::to_string(i)}
            };
        }
    }

    // Get raw pointers (zero-copy access to ZMQ buffers)
    int32_t* h_timesteps = static_cast<int32_t*>(arrays[0].data());
    uint8_t* h_do_generate = static_cast<uint8_t*>(arrays[1].data());
    float* h_frequencies = static_cast<float*>(arrays[2].data());
    float* h_amplitudes = static_cast<float*>(arrays[3].data());
    float* h_offset_phases = static_cast<float*>(arrays[4].data());

    // Validate sizes (based on client's num_tones)
    size_t expected_size = static_cast<size_t>(num_timesteps) * num_channels * num_tones;
    if (arrays[0].size() != static_cast<size_t>(num_timesteps) * sizeof(int32_t) ||
        arrays[1].size() != static_cast<size_t>(num_timesteps) * sizeof(uint8_t) ||
        arrays[2].size() != expected_size * sizeof(float) ||
        arrays[3].size() != expected_size * sizeof(float) ||
        arrays[4].size() != expected_size * sizeof(float)) {
        std::string error = "Array size mismatch. Expected " +
                           std::to_string(expected_size) + " floats for freq/amp/phase, got " +
                           std::to_string(arrays[2].size() / sizeof(float));
        return {
            {"success", false},
            {"error_message", error}
        };
    }

    // Create command - pass pointers to ZMQ buffers
    WaveformCommand cmd;
    cmd.type = AWGCommandType::WAVEFORM_BATCH;
    cmd.waveform_batch_data = std::make_shared<WaveformCommand::WaveformBatchData>();
    cmd.waveform_batch_data->batch_id = batch_id;
    cmd.waveform_batch_data->num_timesteps = num_timesteps;
    cmd.waveform_batch_data->num_tones = num_tones;
    cmd.waveform_batch_data->trigger_type = trigger_type;
    cmd.waveform_batch_data->h_timesteps = h_timesteps;
    cmd.waveform_batch_data->h_do_generate = h_do_generate;
    cmd.waveform_batch_data->h_frequencies = h_frequencies;
    cmd.waveform_batch_data->h_amplitudes = h_amplitudes;
    cmd.waveform_batch_data->h_offset_phases = h_offset_phases;

    // Queue command and wait for completion (synchronous copy to GPU in AWG thread)
    CommandResult result = awg_->queueCommandAndWait(cmd, 1000);

    if (result.success) {
        std::cout << "[Server] WaveformBatch uploaded successfully" << std::endl;
    } else {
        std::cout << "[Server] WaveformBatch failed: " << result.error_message << std::endl;
    }

    return {
        {"success", result.success},
        {"error_message", result.error_message},
        {"batch_id", batch_id}
    };
}

} // namespace aod
