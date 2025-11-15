#include "server.h"
#include "awg_interface.h"
#include <config.h>
#include <iostream>
#include <csignal>
#include <memory>
#include <string>
#include <bitset>

// Global server pointer for signal handler
std::unique_ptr<aod::AODServer> g_server;

void signalHandler(int signum) {
    std::cout << "\n[Main] Interrupt signal (" << signum << ") received." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help           Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Configuration is compiled in from config.cmake" << std::endl;
    std::cout << "To change settings, edit config.cmake and rebuild." << std::endl;
}

int main(int argc, char** argv) {
    using namespace aod::config;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "[Main] AOD Dynamic Server" << std::endl;
    std::cout << "[Main] Configuration (compiled):" << std::endl;
    std::cout << "[Main]   Server port: " << SERVER_PORT << std::endl;
    std::cout << "[Main]   AWG serial: " << AWG_SERIAL_NUMBER << std::endl;
    std::cout << "[Main]   AWG channel mask: 0b" << std::bitset<4>(AWG_CHANNEL_MASK) << std::endl;
    std::cout << "[Main]   Sample rate: " << AWG_SAMPLE_RATE << " Hz" << std::endl;

    // Initialize and start AWG interface thread
    auto awg = std::make_shared<aod::AWGInterface>();
    if (!awg->start()) {
        std::cerr << "Error: Failed to start AWG thread" << std::endl;
        return 1;
    }

    // Create server
    g_server = std::make_unique<aod::AODServer>(awg);

    // Initialize server
    if (!g_server->initialize()) {
        std::cerr << "Error: Failed to initialize server" << std::endl;
        return 1;
    }

    // Register signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::cout << "[Main] AOD Dynamic Server starting..." << std::endl;
    std::cout << "[Main] Press Ctrl+C to stop" << std::endl;

    // Run server (blocks until stop() is called)
    g_server->run();

    // Cleanup
    std::cout << "[Main] Shutting down..." << std::endl;
    g_server.reset();
    awg->stop();

    std::cout << "[Main] Server stopped" << std::endl;
    return 0;
}
