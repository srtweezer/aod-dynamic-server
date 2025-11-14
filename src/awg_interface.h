#ifndef AOD_AWG_INTERFACE_H
#define AOD_AWG_INTERFACE_H

#include "config.h"
#include <string>

#include <spectrum/dlltyp.h>
#include <spectrum/regs.h>
#include <spectrum/spcerr.h>
#include <spectrum/spcm_drv.h>

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
    drv_handle card_handle_;
};

} // namespace aod

#endif // AOD_AWG_INTERFACE_H
