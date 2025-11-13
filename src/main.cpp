#include "server.h"
#include "config.h"
#include "awg_interface.h"
#include <iostream>
#include <csignal>
#include <memory>
#include <string>

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
    std::cout << "  --config PATH    Path to configuration file (default: config.yml)" << std::endl;
    std::cout << "  --help           Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    std::string config_file = "config.yml";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--config") {
            if (i + 1 < argc) {
                config_file = argv[++i];
            } else {
                std::cerr << "Error: --config requires a path argument" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Check if config file exists
    if (!aod::Config::exists(config_file)) {
        std::cerr << "Error: Configuration file not found: " << config_file << std::endl;
        std::cerr << std::endl;
        std::cerr << "Please create a configuration file by copying the template:" << std::endl;
        std::cerr << "  cp config.yml.template config.yml" << std::endl;
        std::cerr << "Then edit config.yml with your hardware parameters." << std::endl;
        return 1;
    }

    // Load configuration
    aod::Config config;
    try {
        config.load(config_file);
        std::cout << "[Main] Loaded configuration from: " << config_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading configuration: " << e.what() << std::endl;
        return 1;
    }

    // Validate configuration
    std::string error_msg;
    if (!config.validate(error_msg)) {
        std::cerr << "Configuration validation failed: " << error_msg << std::endl;
        return 1;
    }

    std::cout << "[Main] Configuration validated successfully" << std::endl;

    // Initialize AWG interface
    auto awg = std::make_shared<aod::AWGInterface>(config.awg());
    if (!awg->connect()) {
        std::cerr << "Error: Failed to connect to AWG" << std::endl;
        return 1;
    }

    // Create server
    g_server = std::make_unique<aod::AODServer>(config.server(), awg);

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
    awg->disconnect();
    g_server.reset();

    std::cout << "[Main] Server stopped" << std::endl;
    return 0;
}
