#include "config.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace aod {

Config::Config() {
    // Default values are set in struct definitions
}

bool Config::exists(const std::string& config_file) {
    std::ifstream f(config_file);
    return f.good();
}

void Config::load(const std::string& config_file) {
    try {
        YAML::Node config = YAML::LoadFile(config_file);

        if (config["server"]) {
            parseServer(config["server"]);
        }

        if (config["awg"]) {
            parseAWG(config["awg"]);
        }

        if (config["logging"]) {
            parseLogging(config["logging"]);
        }

        if (config["gpu"]) {
            parseGPU(config["gpu"]);
        }

    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to parse config file: " + std::string(e.what()));
    }
}

void Config::parseServer(const YAML::Node& node) {
    if (node["port"]) {
        server_.port = node["port"].as<int>();
    }
    if (node["bind_address"]) {
        server_.bind_address = node["bind_address"].as<std::string>();
    }
}

void Config::parseAWG(const YAML::Node& node) {
    if (node["device_path"]) {
        awg_.device_path = node["device_path"].as<std::string>();
    }
    if (node["serial_number"]) {
        awg_.serial_number = node["serial_number"].as<int>();
    }
    if (node["sample_rate"]) {
        awg_.sample_rate = node["sample_rate"].as<int>();
    }
    if (node["max_amplitude"]) {
        awg_.max_amplitude = node["max_amplitude"].as<double>();
    }
    if (node["num_channels"]) {
        awg_.num_channels = node["num_channels"].as<int>();
    }
    if (node["external_clock"]) {
        const YAML::Node& ext_clk = node["external_clock"];
        if (ext_clk["enabled"]) {
            awg_.external_clock.enabled = ext_clk["enabled"].as<bool>();
        }
        if (ext_clk["frequency"]) {
            awg_.external_clock.frequency = ext_clk["frequency"].as<int>();
        }
    }
}

void Config::parseLogging(const YAML::Node& node) {
    if (node["level"]) {
        logging_.level = node["level"].as<std::string>();
    }
    if (node["file"]) {
        logging_.file = node["file"].as<std::string>();
    }
}

void Config::parseGPU(const YAML::Node& node) {
    if (node["device_id"]) {
        gpu_.device_id = node["device_id"].as<int>();
    }
    if (node["enabled"]) {
        gpu_.enabled = node["enabled"].as<bool>();
    }
}

bool Config::validate(std::string& error_msg) const {
    // Validate server config
    if (server_.port <= 0 || server_.port > 65535) {
        error_msg = "Invalid server port: " + std::to_string(server_.port);
        return false;
    }

    // Validate AWG config
    if (awg_.sample_rate <= 0) {
        error_msg = "Invalid AWG sample rate: " + std::to_string(awg_.sample_rate);
        return false;
    }

    if (awg_.max_amplitude <= 0.0) {
        error_msg = "Invalid AWG max amplitude: " + std::to_string(awg_.max_amplitude);
        return false;
    }

    if (awg_.num_channels <= 0 || awg_.num_channels > 4) {
        error_msg = "Invalid AWG channel count: " + std::to_string(awg_.num_channels);
        return false;
    }

    // Validate logging config
    if (logging_.level != "debug" && logging_.level != "info" &&
        logging_.level != "warning" && logging_.level != "error") {
        error_msg = "Invalid logging level: " + logging_.level;
        return false;
    }

    // Validate GPU config
    if (gpu_.device_id < 0) {
        error_msg = "Invalid GPU device ID: " + std::to_string(gpu_.device_id);
        return false;
    }

    return true;
}

} // namespace aod
