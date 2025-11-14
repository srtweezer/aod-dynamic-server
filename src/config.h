#ifndef AOD_CONFIG_H
#define AOD_CONFIG_H

#include <string>
#include <yaml-cpp/yaml.h>

namespace aod {

// Server configuration
struct ServerConfig {
    int port = 5555;
    std::string bind_address = "tcp://*";
};

// AWG configuration
struct AWGConfig {
    int serial_number = 0;         // Serial number for device identification
    int sample_rate = 200000000;   // 200 MSample/s
    double max_amplitude = 0.18;   // Volts
    int num_channels = 4;

    struct ExternalClock {
        bool enabled = false;
        int frequency = 10000000;  // 10 MHz
    } external_clock;
};

// Logging configuration
struct LoggingConfig {
    std::string level = "info";
    std::string file = "";  // Empty means console only
};

// GPU configuration
struct GPUConfig {
    int device_id = 0;
    bool enabled = true;
};

// Main configuration class
class Config {
public:
    Config();

    // Load configuration from YAML file
    void load(const std::string& config_file);

    // Check if config file exists
    static bool exists(const std::string& config_file);

    // Get configuration sections
    const ServerConfig& server() const { return server_; }
    const AWGConfig& awg() const { return awg_; }
    const LoggingConfig& logging() const { return logging_; }
    const GPUConfig& gpu() const { return gpu_; }

    // Validation
    bool validate(std::string& error_msg) const;

private:
    ServerConfig server_;
    AWGConfig awg_;
    LoggingConfig logging_;
    GPUConfig gpu_;

    // Helper methods for parsing
    void parseServer(const YAML::Node& node);
    void parseAWG(const YAML::Node& node);
    void parseLogging(const YAML::Node& node);
    void parseGPU(const YAML::Node& node);
};

} // namespace aod

#endif // AOD_CONFIG_H
