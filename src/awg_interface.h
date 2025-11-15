#ifndef AOD_AWG_INTERFACE_H
#define AOD_AWG_INTERFACE_H

#include <config.h>
#include <string>
#include <thread>
#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>

#include <spectrum/dlltyp.h>
#include <spectrum/regs.h>
#include <spectrum/spcerr.h>
#include <spectrum/spcm_drv.h>

namespace aod {

// Forward declarations
template<typename T> class ThreadSafeQueue;

// Waveform command structure (placeholder for now)
struct WaveformCommand {
    // TODO: Define actual waveform parameters
    // For now, just a placeholder
    int command_id;
};

// Interface to Spectrum Instrumentation AWG hardware
// Runs in a dedicated thread for FIFO streaming
class AWGInterface {
public:
    AWGInterface();
    ~AWGInterface();

    // Start AWG thread (connects to hardware and starts thread)
    bool start();

    // Stop AWG thread (disconnects and joins thread)
    void stop();

    // Check if running
    bool isRunning() const { return running_; }

    // Queue a waveform command for execution
    void queueCommand(const WaveformCommand& cmd);

private:
    // Thread entry point
    void threadLoop();

    // Initialize AWG hardware (called from thread)
    bool connectHardware();

    // Disconnect AWG hardware (called from thread)
    void disconnectHardware();

    // Process commands from queue (placeholder)
    void processCommand(const WaveformCommand& cmd);

    // Thread and synchronization
    std::unique_ptr<std::thread> thread_;
    std::atomic<bool> running_;
    std::atomic<bool> shutdown_requested_;
    std::mutex init_mutex_;
    std::condition_variable init_cv_;
    bool init_complete_;

    // Command queue
    std::unique_ptr<ThreadSafeQueue<WaveformCommand>> command_queue_;

    // AWG hardware
    std::atomic<bool> connected_;
    drv_handle card_handle_;
};

} // namespace aod

#endif // AOD_AWG_INTERFACE_H
