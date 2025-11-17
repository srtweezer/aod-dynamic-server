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
    WAVEFORM_BATCH,
};

// Single waveform with interpolated tone parameters
struct WaveformData {
    int32_t duration;          // In WAVEFORM_TIMESTEP units
    int32_t num_tones;
    int32_t num_steps;

    std::vector<int32_t> time_steps;      // Size: num_steps (interpolation x-coordinates)
    std::vector<float> frequencies;       // Size: num_steps * num_channels * num_tones
    std::vector<float> amplitudes;        // Size: num_steps * num_channels * num_tones
    std::vector<float> offset_phases;     // Size: num_steps * num_channels * num_tones
};

// Batch of waveforms
struct WaveformBatch {
    int trigger_type;                     // TriggerType enum from protobuf
    int32_t delay;                        // In WAVEFORM_TIMESTEP units
    std::vector<WaveformData> waveforms;
    int32_t batch_id;                     // Unique ID
};

// Waveform command structure
struct WaveformCommand {
    AWGCommandType type;

    // Command-specific data
    struct InitializeData {
        std::vector<int32> amplitudes_mv;  // mV for each active channel
    };

    struct WaveformBatchData {
        WaveformBatch batch;
    };

    // Union of command data
    std::shared_ptr<InitializeData> initialize_data;
    std::shared_ptr<WaveformBatchData> waveform_batch_data;
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

    // Store waveform batch in queue (called from thread)
    bool storeBatch(const WaveformBatch& batch);

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

    // Waveform batch queue
    std::unique_ptr<WaveformBatch[]> batch_queue_;  // Fixed-size array allocated in constructor
    std::atomic<int> num_batches_;                  // Current number of queued batches
    std::atomic<int> next_batch_id_;                // For generating unique batch IDs
    int max_batches_;                                // Size of batch_queue_ array
};

} // namespace aod

#endif // AOD_AWG_INTERFACE_H
