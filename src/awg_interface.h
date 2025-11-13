#ifndef AOD_AWG_INTERFACE_H
#define AOD_AWG_INTERFACE_H

#include "config.h"
#include <string>

namespace aod {

// Interface to Spectrum Instrumentation AWG hardware
// Currently a placeholder - will be implemented with SDK integration
class AWGInterface {
public:
    AWGInterface(const AWGConfig& config);
    ~AWGInterface();

    // Connect to AWG hardware
    bool connect();

    // Disconnect from AWG hardware
    void disconnect();

    // Check if connected
    bool isConnected() const { return connected_; }

    // Get configuration
    const AWGConfig& config() const { return config_; }

    // Placeholder for future waveform streaming methods
    // bool startStream();
    // bool writeBuffer(const float* data, size_t length);
    // bool stopStream();

private:
    AWGConfig config_;
    bool connected_;

    // Future: Spectrum SDK card handle
    // void* card_handle_;
};

} // namespace aod

#endif // AOD_AWG_INTERFACE_H
