#ifndef AOD_AWG_INTERFACE_H
#define AOD_AWG_INTERFACE_H

#include <config.h>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>

#include <spectrum/dlltyp.h>
#include <spectrum/regs.h>
#include <spectrum/spcerr.h>
#include <spectrum/spcm_drv.h>

#include "cuda/gpu_buffers.h"

namespace aod {

// Forward declarations
template<typename T> class ThreadSafeQueue;

// Command result structure
struct CommandResult {
    bool success;
    std::string error_message;
};

// AWG command types
enum class AWGCommandType {
    INITIALIZE,
    START,
    STOP,
    // Future: GENERATE_WAVEFORM, etc.
};

// Waveform command structure
struct WaveformCommand {
    AWGCommandType type;

    // Command-specific data
    struct InitializeData {
        std::vector<int32> amplitudes_mv;  // mV for each active channel
    };

    // Union of command data (for now just Initialize)
    std::shared_ptr<InitializeData> initialize_data;
};

// AWG state
enum class AWGState {
    DISCONNECTED,  // No hardware connection
    CONNECTED,     // Connected but not initialized
    INITIALIZED,   // Configured and ready
    STREAMING      // Actively outputting waveforms
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

    // Get current state
    AWGState getState() const { return state_; }

    // Queue a waveform command for execution
    void queueCommand(const WaveformCommand& cmd);

    // Queue command and wait for completion (blocks until done or timeout)
    CommandResult queueCommandAndWait(const WaveformCommand& cmd, int timeout_ms = 1000);

private:
    // Thread entry point
    void threadLoop();

    // Connect to AWG hardware (called from thread)
    bool connectHardware();

    // Disconnect AWG hardware (called from thread)
    void disconnectHardware();

    // Allocate GPU buffers
    bool allocateGPU();

    // Free GPU buffers
    void freeGPU();

    // Initialize AWG for FIFO streaming (called from thread)
    bool initializeAWG(const std::vector<int32>& amplitudes_mv);

    // Stop AWG output and DMA (called from thread)
    bool stopAWG();

    // Zero the software buffer
    void zeroBuffer();

    // Process commands from queue
    void processCommand(const WaveformCommand& cmd);

    // Thread and synchronization
    std::unique_ptr<std::thread> thread_;
    std::atomic<bool> running_;
    std::atomic<bool> shutdown_requested_;

    // Initialization synchronization
    std::mutex init_mutex_;
    std::condition_variable init_cv_;
    bool init_complete_;

    // Command result synchronization
    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    CommandResult last_result_;
    bool result_ready_;

    // Command queue
    std::unique_ptr<ThreadSafeQueue<WaveformCommand>> command_queue_;

    // AWG hardware and state
    std::atomic<bool> connected_;
    std::atomic<AWGState> state_;
    drv_handle card_handle_;

    // FIFO buffers
    void* sw_buffer_;
    size_t sw_buffer_size_;
    size_t notify_size_;

    // GPU buffers
    GPUBuffers gpu_buffers_;
};

} // namespace aod

#endif // AOD_AWG_INTERFACE_H
