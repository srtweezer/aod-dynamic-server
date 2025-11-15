#ifndef AOD_SERVER_H
#define AOD_SERVER_H

#include <config.h>
#include "aod_server.pb.h"
#include <zmq.hpp>
#include <memory>
#include <atomic>

namespace aod {

// Forward declaration
class AWGInterface;

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
    // Handle incoming request
    Response handleRequest(const Request& request);

    // Command handlers
    Response handlePing(const PingRequest& request);
    Response handleInitialize(const InitializeRequest& request);
    Response handleStop(const StopRequest& request);

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
