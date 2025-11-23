#ifndef AOD_SERVER_H
#define AOD_SERVER_H

#include <config.h>
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <memory>
#include <atomic>
#include <string>

namespace aod {

// Forward declaration
class AWGInterface;

// Use json type alias
using json = nlohmann::json;

class AODServer {
public:
    AODServer(std::shared_ptr<AWGInterface> awg);
    ~AODServer();

    // Initialize the server (create socket, bind)
    bool initialize();

    // Run the server loop (blocks until stop() is called)
    void run();

    // Stop the server
    void stop();

    // Check if server is running
    bool isRunning() const { return running_; }

private:
    // Command handlers (return JSON response objects)
    json handleInitialize(const json& request);
    json handleStart(const json& request);
    json handleStop(const json& request);
    json handleWaveformBatch(const json& request, zmq::socket_t& socket);

    // ZMQ context and socket
    std::unique_ptr<zmq::context_t> context_;
    std::unique_ptr<zmq::socket_t> socket_;

    // AWG interface
    std::shared_ptr<AWGInterface> awg_;

    // Running flag
    std::atomic<bool> running_;
};

} // namespace aod

#endif // AOD_SERVER_H
