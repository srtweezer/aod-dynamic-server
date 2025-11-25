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
#include <unordered_map>

#include <spectrum/dlltyp.h>
#include <spectrum/regs.h>
#include <spectrum/spcerr.h>
#include <spectrum/spcm_drv.h>

#include "cuda/gpu_buffers.h"
#include "shared_memory.h"

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
    FINISH,           // Graceful exit after all batches complete
    WAVEFORM_BATCH,
};

// Waveform command structure
struct WaveformCommand {
    AWGCommandType type;

    // Command-specific data structures
    struct InitializeData {
        std::vector<int32_t> amplitudes_mv;  // mV for each active channel
    };

    struct WaveformBatchData {
        // Pre-flattened timeline arrays from client (pointers into ZMQ buffers)
        int batch_id;                    // Client-provided batch ID (for ordering)
        int num_timesteps;               // Number of timesteps in this batch
        int num_tones;                   // Actual number of tones used (can be < AOD_MAX_TONES)
        std::string trigger_type;        // "software" or "external"
        int32_t* h_timesteps;            // Size: num_timesteps
        uint8_t* h_do_generate;          // Size: num_timesteps - 1 (controls intervals)
        double* h_frequencies;           // Size: num_timesteps * num_channels * num_tones (flattened, float64)
        float* h_amplitudes;             // Size: num_timesteps * num_channels * num_tones
        float* h_offset_phases_user;     // Size: num_timesteps * num_channels * num_tones (user-provided)
    };

    // Shared pointers to command data
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

    // Shared memory accessors
    bool hasSharedMemory() const;
    std::string getSharedMemoryName() const;
    size_t getSharedMemorySize() const;
    void* getSharedMemoryPointer() const;

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
    bool initializeAWG(const std::vector<int32_t>& amplitudes_mv);

    // Stop AWG output and DMA (called from thread)
    bool stopAWG();

    // Finish streaming gracefully after all batches (called from thread)
    bool finishAWG();

    // Upload waveform batch directly to GPU (called from thread)
    bool uploadBatchToGPU(const WaveformCommand::WaveformBatchData& data);

    // FIFO streaming methods (called from thread)
    bool startStreaming();                              // Main streaming loop
    bool playBatch(int batch_id);                       // Stream single batch
    bool fifoStreamLoop(size_t total_bytes, size_t buffer_size);  // FIFO transfer loop
    bool waitForCompletion();                           // Wait for small waveform completion
    bool processBatchTransition();                      // Drain command queue between batches
    void idleStreamingLoop();                           // Wait loop after all batches complete
    void signalCommandComplete();                       // Signal command result ready

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

    // DMA parameters
    size_t notify_size_;  // Notification chunk size for FIFO transfers

    // GPU buffers (h_samples_pinned used for AWG DMA transfers)
    GPUBuffers gpu_buffers_;

    // Shared memory for zero-copy client communication (optional)
    std::unique_ptr<SharedMemoryManager> shared_memory_;

    // Batch metadata tracking (for appending and playback order)
    std::vector<int> batch_ids_;                              // Sorted list of batch IDs for playback order
    std::unordered_map<int, std::string> batch_trigger_types_; // batch_id -> trigger type
    std::unordered_map<int, int> batch_start_indices_;         // batch_id -> start index in GPU arrays
    std::unordered_map<int, int> batch_lengths_;               // batch_id -> number of timesteps (array length)
    std::unordered_map<int, int> batch_timestep_duration_;     // batch_id -> max timestep value (waveform duration)

    // Maximum index used in GPU timeline arrays (for appending)
    int max_timestep_index_;

    // FIFO streaming state
    std::atomic<bool> streaming_active_;   // True when in streaming loop
    std::atomic<bool> stop_requested_;     // Flag to exit streaming immediately
    std::atomic<bool> finish_requested_;   // Flag to exit after all batches complete
    size_t bytes_available_;               // Bytes generated and ready to stream
    size_t write_position_;                // Current write position in circular sw_buffer_
};

} // namespace aod

#endif // AOD_AWG_INTERFACE_H
