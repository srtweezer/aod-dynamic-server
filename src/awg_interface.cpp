#include "awg_interface.h"
#include "thread_safe_queue.h"
#include <config.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <unistd.h>  // For getpid()
#include <cuda_runtime.h>  // For cudaMemcpy

namespace aod {

AWGInterface::AWGInterface()
    : running_(false),
      shutdown_requested_(false),
      init_complete_(false),
      result_ready_(false),
      connected_(false),
      state_(AWGState::DISCONNECTED),
      card_handle_(nullptr),
      notify_size_(0),
      command_queue_(std::make_unique<ThreadSafeQueue<WaveformCommand>>()),
      max_timestep_index_(0),
      streaming_active_(false),
      stop_requested_(false),
      finish_requested_(false),
      bytes_available_(0),
      write_position_(0) {
}

AWGInterface::~AWGInterface() {
    stop();
}

bool AWGInterface::start() {
    if (running_) {
        std::cerr << "[AWG] Already running" << std::endl;
        return false;
    }

    std::cout << "[AWG] Starting AWG thread..." << std::endl;

    shutdown_requested_ = false;
    running_ = true;
    init_complete_ = false;

    // Start thread
    thread_ = std::make_unique<std::thread>(&AWGInterface::threadLoop, this);

    // Wait for initialization to complete
    {
        std::unique_lock<std::mutex> lock(init_mutex_);
        init_cv_.wait(lock, [this] { return init_complete_; });
    }

    if (!connected_) {
        std::cerr << "[AWG] Thread started but hardware connection failed" << std::endl;
        stop();
        return false;
    }

    std::cout << "[AWG] AWG thread started successfully" << std::endl;
    return true;
}

void AWGInterface::stop() {
    if (!running_) {
        return;
    }

    std::cout << "[AWG] Stopping AWG thread..." << std::endl;

    shutdown_requested_ = true;
    command_queue_->shutdown();

    // Wait for thread to finish
    if (thread_ && thread_->joinable()) {
        thread_->join();
    }

    running_ = false;
    std::cout << "[AWG] AWG thread stopped" << std::endl;
}

void AWGInterface::queueCommand(const WaveformCommand& cmd) {
    if (!running_) {
        std::cerr << "[AWG] Cannot queue command - thread not running" << std::endl;
        return;
    }

    command_queue_->push(cmd);
}

CommandResult AWGInterface::queueCommandAndWait(const WaveformCommand& cmd, int timeout_ms) {
    if (!running_) {
        return CommandResult{false, "AWG thread not running"};
    }

    // Reset result flag
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        result_ready_ = false;
    }

    // Queue command
    command_queue_->push(cmd);

    // Wait for result with timeout
    std::unique_lock<std::mutex> lock(result_mutex_);
    bool completed = result_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                          [this] { return result_ready_; });

    if (!completed) {
        return CommandResult{false, "Command timed out after " + std::to_string(timeout_ms) + "ms"};
    }

    return last_result_;
}

void AWGInterface::threadLoop() {
    std::cout << "[AWG Thread] Starting..." << std::endl;

    // Initialize hardware
    bool init_success = connectHardware();
    connected_ = init_success;

    if (init_success) {
        state_ = AWGState::CONNECTED;
    }

    // Signal initialization complete
    {
        std::lock_guard<std::mutex> lock(init_mutex_);
        init_complete_ = true;
    }
    init_cv_.notify_one();

    if (!init_success) {
        std::cerr << "[AWG Thread] Failed to connect to hardware" << std::endl;
        running_ = false;
        state_ = AWGState::DISCONNECTED;
        return;
    }

    std::cout << "[AWG Thread] Entering main loop..." << std::endl;

    // Main loop: process commands and handle FIFO streaming
    while (!shutdown_requested_) {
        // Try to get a command (with timeout via pop)
        auto cmd = command_queue_->pop();

        if (!cmd.has_value()) {
            // Shutdown or timeout
            continue;
        }

        // Process command
        processCommand(cmd.value());

        // TODO: Add FIFO loop here
        // - Monitor SPC_DATA_AVAIL_USER_LEN
        // - Generate waveforms
        // - Write to AWG buffer
        // - M2CMD_DATA_WAITDMA
    }

    std::cout << "[AWG Thread] Shutting down..." << std::endl;

    // Cleanup
    disconnectHardware();

    std::cout << "[AWG Thread] Stopped" << std::endl;
}

bool AWGInterface::connectHardware() {
    using namespace aod::config;

    std::cout << "[AWG] Scanning for Spectrum AWG devices..." << std::endl;
    std::cout << "[AWG]   Target serial number: " << AWG_SERIAL_NUMBER << std::endl;
    std::cout << "[AWG]   Sample rate: " << AWG_SAMPLE_RATE << " S/s" << std::endl;
    std::cout << "[AWG]   Channel mask: 0x" << std::hex << AWG_CHANNEL_MASK << std::dec << std::endl;
    std::cout << "[AWG]   Max amplitude: " << AWG_MAX_AMPLITUDE << " V" << std::endl;

    // Scan for devices /dev/spcm0 through /dev/spcm15
    const int MAX_CARDS = 16;
    bool found = false;

    for (int card_idx = 0; card_idx < MAX_CARDS && !found; card_idx++) {
        char device_path[64];
        snprintf(device_path, sizeof(device_path), "/dev/spcm%d", card_idx);

        // Try to open the device
        drv_handle hCard = spcm_hOpen(device_path);
        if (!hCard) {
            // Device doesn't exist or can't be opened - this is OK, continue scanning
            continue;
        }

        // Read card information
        int32 lCardType, lSerialNumber, lFncType;
        char szCardName[20] = {};

        spcm_dwGetParam_i32(hCard, SPC_PCITYP, &lCardType);
        spcm_dwGetParam_ptr(hCard, SPC_PCITYP, szCardName, sizeof(szCardName));
        spcm_dwGetParam_i32(hCard, SPC_PCISERIALNO, &lSerialNumber);
        spcm_dwGetParam_i32(hCard, SPC_FNCTYPE, &lFncType);

        std::cout << "[AWG] Found device " << device_path << ": "
                  << szCardName << " SN " << lSerialNumber;

        // Check if it's a generator card
        if (lFncType != SPCM_TYPE_AO) {
            std::cout << " (not a generator, skipping)" << std::endl;
            spcm_vClose(hCard);
            continue;
        }

        // Check serial number match
        if (AWG_SERIAL_NUMBER != 0 && lSerialNumber != AWG_SERIAL_NUMBER) {
            std::cout << " (serial mismatch, skipping)" << std::endl;
            spcm_vClose(hCard);
            continue;
        }

        // Found matching card!
        std::cout << " ✓" << std::endl;
        card_handle_ = hCard;
        found = true;

        std::cout << "[AWG] Connected to " << device_path
                  << " (" << szCardName << " SN " << lSerialNumber << ")" << std::endl;

        // Allocate GPU buffers after successful hardware connection
        if (!allocateGPU()) {
            std::cerr << "[AWG] Failed to allocate GPU buffers" << std::endl;
            spcm_vClose(card_handle_);
            card_handle_ = nullptr;
            return false;
        }

        // Initialize shared memory if enabled
        if (USE_SHARED_MEMORY) {
            // Generate unique name using process ID
            std::string shm_name = "/aod_server_" + std::to_string(getpid());

            // Calculate size
            int num_channels = __builtin_popcount(AWG_CHANNEL_MASK);
            size_t arrays_size = MAX_WAVEFORM_TIMESTEPS * num_channels * AOD_MAX_TONES;
            size_t shm_size =
                MAX_WAVEFORM_TIMESTEPS * sizeof(int32_t) +      // timesteps
                (MAX_WAVEFORM_TIMESTEPS - 1) * sizeof(uint8_t) +  // do_generate
                arrays_size * sizeof(double) +                   // frequencies (float64)
                2 * arrays_size * sizeof(float);                 // amp/phase (float32)

            shared_memory_ = std::make_unique<SharedMemoryManager>();
            if (!shared_memory_->create(shm_name, shm_size)) {
                std::cerr << "[AWG] Warning: Failed to create shared memory, will use ZMQ mode" << std::endl;
                shared_memory_.reset();
            }
        }
    }

    if (!found) {
        if (AWG_SERIAL_NUMBER != 0) {
            std::cerr << "[AWG] Error: No generator card found with serial number "
                      << AWG_SERIAL_NUMBER << std::endl;
        } else {
            std::cerr << "[AWG] Error: No generator cards found" << std::endl;
        }
        return false;
    }

    return true;
}

void AWGInterface::disconnectHardware() {
    if (connected_) {
        std::cout << "[AWG Thread] Disconnecting from AWG..." << std::endl;

        // Destroy shared memory if exists
        if (shared_memory_) {
            shared_memory_->destroy();
            shared_memory_.reset();
        }

        // Free GPU buffers
        freeGPU();

        if (card_handle_) {
            spcm_vClose(card_handle_);
            card_handle_ = nullptr;
        }

        connected_ = false;
        std::cout << "[AWG Thread] Disconnected" << std::endl;
    }
}

bool AWGInterface::allocateGPU() {
    std::cout << "[AWG] Allocating GPU buffers..." << std::endl;
    return allocateGPUBuffers(gpu_buffers_);
}

void AWGInterface::freeGPU() {
    freeGPUBuffers(gpu_buffers_);
}

bool AWGInterface::initializeAWG(const std::vector<int32>& amplitudes_mv) {
    using namespace aod::config;

    std::cout << "[AWG Thread] Initializing AWG..." << std::endl;

    // Validate amplitude count matches active channels
    int num_active_channels = __builtin_popcount(AWG_CHANNEL_MASK);
    if (amplitudes_mv.size() != static_cast<size_t>(num_active_channels)) {
        std::string error = "Expected " + std::to_string(num_active_channels) +
                           " amplitudes for active channels, got " + std::to_string(amplitudes_mv.size());
        std::cerr << "[AWG Thread] Error: " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    // If already initialized, stop DMA first
    if (state_ == AWGState::INITIALIZED || state_ == AWGState::STREAMING) {
        std::cout << "[AWG Thread] Already initialized, updating parameters..." << std::endl;
        spcm_dwSetParam_i32(card_handle_, SPC_M2CMD, M2CMD_DATA_STOPDMA);
    }

    spcm_dwSetParam_i32(card_handle_, SPC_M2CMD, M2CMD_CARD_STOP);

    // 1. Set card mode to FIFO single
    spcm_dwSetParam_i32(card_handle_, SPC_CARDMODE, SPC_REP_FIFO_SINGLE);
    spcm_dwSetParam_i32(card_handle_, SPC_CHENABLE, AWG_CHANNEL_MASK);
    spcm_dwSetParam_i32(card_handle_, SPC_LOOPS, 0);  // Continuous

    // 2. Setup clock
    spcm_dwSetParam_i32(card_handle_, SPC_CLOCKMODE, SPC_CM_INTPLL);
    spcm_dwSetParam_i64(card_handle_, SPC_SAMPLERATE, AWG_SAMPLE_RATE);
    spcm_dwSetParam_i32(card_handle_, SPC_CLOCKOUT, 0);

    // 3. Trigger will be configured per-batch in playBatch()

    // 4. Setup each active channel
    // Register offsets: CH1 = CH0 + 100, CH2 = CH0 + 200, etc.
    constexpr int32 CH_OFFSET = 100;

    int amp_idx = 0;
    for (int ch = 0; ch < 4; ch++) {
        if (AWG_CHANNEL_MASK & (1 << ch)) {
            int32 amplitude_mv = amplitudes_mv[amp_idx];

            // Set amplitude
            spcm_dwSetParam_i32(card_handle_, SPC_AMP0 + ch * CH_OFFSET, amplitude_mv);

            // Set filter to 0 (no filter)
            spcm_dwSetParam_i32(card_handle_, SPC_FILTER0 + ch * CH_OFFSET, 0);

            // Enable output
            spcm_dwSetParam_i32(card_handle_, SPC_ENABLEOUT0 + ch * CH_OFFSET, 1);

            std::cout << "[AWG Thread]   Channel " << ch << ": " << amplitude_mv << " mV" << std::endl;
            amp_idx++;
        }
    }

    // 5. Set hardware buffer size
    spcm_dwSetParam_i64(card_handle_, SPC_DATA_OUTBUFSIZE, AOD_HW_BUFFER_SIZE);

    // 6. Write setup to card
    int32 dwErr = spcm_dwSetParam_i32(card_handle_, SPC_M2CMD, M2CMD_CARD_WRITESETUP);
    if (dwErr != ERR_OK) {
        char szError[256];
        spcm_dwGetErrorInfo_i32(card_handle_, nullptr, nullptr, szError);
        std::string error = std::string("AWG setup failed: ") + szError;
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    // 7. Validate and setup DMA parameters
    notify_size_ = AOD_NOTIFY_SIZE;
    int num_channels = __builtin_popcount(AWG_CHANNEL_MASK);

    // Use GPU pinned buffer for DMA (already allocated in allocateGPU)
    size_t pinned_buffer_bytes = gpu_buffers_.total_samples * sizeof(int16_t);
    size_t pinned_total_samples = gpu_buffers_.total_samples;

    // Validate pinned buffer size
    if (pinned_buffer_bytes % 2 != 0) {
        std::string error = "Pinned buffer size must be even";
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    if (pinned_total_samples % num_channels != 0) {
        std::string error = "Pinned buffer samples (" + std::to_string(pinned_total_samples) +
                           ") must be multiple of num_channels (" + std::to_string(num_channels) + ")";
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    size_t pinned_samples_per_channel = pinned_total_samples / num_channels;
    if (pinned_samples_per_channel % WAVEFORM_TIMESTEP != 0) {
        std::string error = "Pinned buffer samples per channel (" + std::to_string(pinned_samples_per_channel) +
                           ") must be multiple of WAVEFORM_TIMESTEP (" + std::to_string(WAVEFORM_TIMESTEP) + ")";
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    // Validate notify size
    if (notify_size_ % 2 != 0) {
        std::string error = "AOD_NOTIFY_SIZE must be even (got " + std::to_string(notify_size_) + ")";
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    size_t notify_total_samples = notify_size_ / sizeof(int16_t);
    if (notify_total_samples % num_channels != 0) {
        std::string error = "Notify size samples (" + std::to_string(notify_total_samples) +
                           ") must be multiple of num_channels (" + std::to_string(num_channels) + ")";
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    size_t notify_samples_per_channel = notify_total_samples / num_channels;
    if (notify_samples_per_channel % WAVEFORM_TIMESTEP != 0) {
        std::string error = "Notify size samples per channel (" + std::to_string(notify_samples_per_channel) +
                           ") must be multiple of WAVEFORM_TIMESTEP (" + std::to_string(WAVEFORM_TIMESTEP) + ")";
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    // Calculate and log buffer characteristics
    size_t timesteps_per_channel = pinned_samples_per_channel / WAVEFORM_TIMESTEP;
    double buffer_duration_us = (double)pinned_samples_per_channel / AWG_SAMPLE_RATE * 1e6;
    size_t notify_timesteps = notify_samples_per_channel / WAVEFORM_TIMESTEP;
    double notify_duration_us = (double)notify_samples_per_channel / AWG_SAMPLE_RATE * 1e6;

    std::cout << "[AWG Thread] ═══════════════════════════════════════════════════" << std::endl;
    std::cout << "[AWG Thread] DMA Buffer (GPU Pinned):" << std::endl;
    std::cout << "[AWG Thread]   Size: " << pinned_buffer_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "[AWG Thread]   Total samples: " << pinned_total_samples << std::endl;
    std::cout << "[AWG Thread]   Samples per channel: " << pinned_samples_per_channel << std::endl;
    std::cout << "[AWG Thread]   Timesteps (per channel): " << timesteps_per_channel << std::endl;
    std::cout << "[AWG Thread]   Duration: " << buffer_duration_us / 1000.0 << " ms"
              << " (" << buffer_duration_us / 1e6 << " s)" << std::endl;
    std::cout << "[AWG Thread] ───────────────────────────────────────────────────" << std::endl;
    std::cout << "[AWG Thread] Notify Size:" << std::endl;
    std::cout << "[AWG Thread]   Size: " << notify_size_ / 1024 << " KB" << std::endl;
    std::cout << "[AWG Thread]   Total samples: " << notify_total_samples << std::endl;
    std::cout << "[AWG Thread]   Samples per channel: " << notify_samples_per_channel << std::endl;
    std::cout << "[AWG Thread]   Timesteps (per channel): " << notify_timesteps << std::endl;
    std::cout << "[AWG Thread]   Duration: " << notify_duration_us << " μs"
              << " (" << notify_duration_us / 1000.0 << " ms)" << std::endl;
    std::cout << "[AWG Thread] ═══════════════════════════════════════════════════" << std::endl;

    // 8. Zero the buffers
    zeroBuffer();

    std::cout << "[AWG Thread] AWG initialized successfully" << std::endl;
    state_ = AWGState::INITIALIZED;
    last_result_ = CommandResult{true, ""};
    return true;
}

bool AWGInterface::stopAWG() {
    using namespace aod::config;

    std::cout << "[AWG Thread] Stopping AWG..." << std::endl;

    // If not initialized, nothing to do - just return success
    if (state_ != AWGState::INITIALIZED && state_ != AWGState::STREAMING) {
        std::cout << "[AWG Thread] AWG not running, nothing to stop" << std::endl;
        last_result_ = CommandResult{true, ""};
        return true;
    }

    // If currently streaming, request stop and wait for streaming loop to exit
    if (streaming_active_) {
        std::cout << "[AWG Thread] Streaming active - requesting stop..." << std::endl;
        stop_requested_ = true;

        // Wait for streaming loop to exit (with timeout)
        int wait_count = 0;
        while (streaming_active_ && wait_count < 100) {  // Max 1 second wait
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
        }

        if (streaming_active_) {
            std::cerr << "[AWG Thread] Warning: Streaming loop did not exit cleanly" << std::endl;
        } else {
            std::cout << "[AWG Thread] Streaming loop exited cleanly" << std::endl;
        }
    }

    // Stop card and DMA
    int32 dwErr = spcm_dwSetParam_i32(card_handle_, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);
    if (dwErr != ERR_OK) {
        char szError[256];
        spcm_dwGetErrorInfo_i32(card_handle_, nullptr, nullptr, szError);
        std::string error = std::string("Failed to stop AWG: ") + szError;
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    // Zero buffer
    zeroBuffer();

    // Clear batch metadata
    batch_ids_.clear();
    batch_trigger_types_.clear();
    batch_start_indices_.clear();
    batch_lengths_.clear();
    batch_timestep_duration_.clear();
    max_timestep_index_ = 0;
    std::cout << "[AWG Thread] All batches cleared" << std::endl;

    // Set state to INITIALIZED (stops streaming but keeps initialization)
    state_ = AWGState::INITIALIZED;
    last_result_ = CommandResult{true, ""};

    std::cout << "[AWG Thread] AWG stopped successfully" << std::endl;
    return true;
}

bool AWGInterface::finishAWG() {
    std::cout << "[AWG Thread] Finish requested (graceful)..." << std::endl;

    if (state_ != AWGState::STREAMING) {
        std::cout << "[AWG Thread] Not streaming, treating FINISH as STOP" << std::endl;
        return stopAWG();
    }

    // Set finish flag - streaming loop will exit after all batches
    finish_requested_ = true;

    std::cout << "[AWG Thread] Finish flag set - waiting for all batches to complete..." << std::endl;

    // Wait for streaming loop to finish all batches and exit
    int wait_count = 0;
    while (streaming_active_ && wait_count < 60000) {  // Max 60 seconds (100us * 60000)
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        wait_count++;
    }

    if (streaming_active_) {
        std::cerr << "[AWG Thread] Warning: Streaming did not finish within timeout" << std::endl;
        last_result_ = CommandResult{false, "FINISH timeout - streaming did not complete"};
        return false;
    }

    std::cout << "[AWG Thread] FINISH complete - all batches played and cleaned up" << std::endl;
    last_result_ = CommandResult{true, ""};
    return true;
}

bool AWGInterface::uploadBatchToGPU(const WaveformCommand::WaveformBatchData& data) {
    using namespace aod::config;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::cout << "[AWG Thread] Uploading batch to GPU..." << std::endl;
    std::cout << "[AWG Thread]   Batch ID: " << data.batch_id << std::endl;
    std::cout << "[AWG Thread]   Timesteps: " << data.num_timesteps << std::endl;
    std::cout << "[AWG Thread]   Num tones: " << data.num_tones << std::endl;

    // Check for duplicate batch_id
    if (batch_trigger_types_.find(data.batch_id) != batch_trigger_types_.end()) {
        std::string error = "Duplicate batch_id: " + std::to_string(data.batch_id);
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    // Validate individual batch size
    if (data.num_timesteps <= 0) {
        std::string error = "Invalid num_timesteps: " + std::to_string(data.num_timesteps);
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    // Check if appending would exceed total capacity
    int new_max_index = max_timestep_index_ + data.num_timesteps;
    if (new_max_index > MAX_WAVEFORM_TIMESTEPS) {
        std::string error = "Total timeline would exceed MAX_WAVEFORM_TIMESTEPS (" +
                           std::to_string(MAX_WAVEFORM_TIMESTEPS) + "). Current: " +
                           std::to_string(max_timestep_index_) + ", Adding: " +
                           std::to_string(data.num_timesteps);
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    // Validate num_tones
    int num_channels = __builtin_popcount(AWG_CHANNEL_MASK);
    if (data.num_tones > AOD_MAX_TONES || data.num_tones <= 0) {
        std::string error = "Invalid num_tones: " + std::to_string(data.num_tones) +
                           " (must be 1-" + std::to_string(AOD_MAX_TONES) + ")";
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    auto t_validation_done = std::chrono::high_resolution_clock::now();
    auto validation_us = std::chrono::duration_cast<std::chrono::microseconds>(t_validation_done - t_start).count();

    // Append batch data to GPU arrays (with strided copy for num_tones padding)
    auto t_gpu_start = std::chrono::high_resolution_clock::now();
    uploadBatchDataToGPU(gpu_buffers_,
                         data.h_timesteps,
                         data.h_do_generate,
                         data.h_frequencies,
                         data.h_amplitudes,
                         data.h_offset_phases,
                         data.num_timesteps,
                         num_channels,
                         data.num_tones,
                         max_timestep_index_);  // Append at current end
    auto t_gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(t_gpu_end - t_gpu_start).count();

    // Validate timesteps are strictly ascending (on GPU)
    auto t_validate_start = std::chrono::high_resolution_clock::now();
    bool is_ascending = validateTimestepsAscending(
        gpu_buffers_.d_batch_timesteps + max_timestep_index_,
        data.num_timesteps);

    if (!is_ascending) {
        std::string error = "Timesteps must be strictly ascending (each timestep < next timestep)";
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    // Get max timestep (last element in the array)
    int32_t max_timestep_value;
    cudaMemcpy(&max_timestep_value,
              gpu_buffers_.d_batch_timesteps + max_timestep_index_ + data.num_timesteps - 1,
              sizeof(int32_t),
              cudaMemcpyDeviceToHost);

    auto t_validate_end = std::chrono::high_resolution_clock::now();
    auto validate_us = std::chrono::duration_cast<std::chrono::microseconds>(t_validate_end - t_validate_start).count();

    std::cout << "[AWG Thread]   Timestep validation: " << validate_us << " μs" << std::endl;
    std::cout << "[AWG Thread]   Max timestep value: " << max_timestep_value << std::endl;

    // Store batch metadata
    batch_trigger_types_[data.batch_id] = data.trigger_type;
    batch_start_indices_[data.batch_id] = max_timestep_index_;
    batch_lengths_[data.batch_id] = data.num_timesteps;
    batch_timestep_duration_[data.batch_id] = max_timestep_value;

    // Add to sorted batch_ids list (insert in sorted position)
    auto insert_pos = std::lower_bound(batch_ids_.begin(), batch_ids_.end(), data.batch_id);
    batch_ids_.insert(insert_pos, data.batch_id);

    // Update max index
    max_timestep_index_ = new_max_index;

    auto t_metadata_start = std::chrono::high_resolution_clock::now();
    auto metadata_us = std::chrono::duration_cast<std::chrono::microseconds>(t_metadata_start - t_gpu_end).count();

    auto t_total_end = std::chrono::high_resolution_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t_total_end - t_start).count();

    last_result_ = CommandResult{true, ""};

    std::cout << "[AWG Thread] Batch " << data.batch_id << " uploaded successfully" << std::endl;
    std::cout << "[AWG Thread]   Start index: " << batch_start_indices_[data.batch_id] << std::endl;
    std::cout << "[AWG Thread]   Total batches: " << batch_ids_.size() << std::endl;
    std::cout << "[AWG Thread]   Total timeline length: " << max_timestep_index_ << std::endl;
    std::cout << "[AWG Thread] ───────────────────────────────────────" << std::endl;
    std::cout << "[AWG Thread] AWG Thread Timing:" << std::endl;
    std::cout << "[AWG Thread]   Validation:        " << validation_us << " μs" << std::endl;
    std::cout << "[AWG Thread]   GPU copy:          " << gpu_us << " μs" << std::endl;
    std::cout << "[AWG Thread]   Metadata update:   " << metadata_us << " μs" << std::endl;
    std::cout << "[AWG Thread]   Total (AWG side):  " << total_us << " μs" << std::endl;
    std::cout << "[AWG Thread] ───────────────────────────────────────" << std::endl;
    return true;
}

void AWGInterface::zeroBuffer() {
    // Zero GPU buffers (includes h_samples_pinned used for DMA)
    zeroGPUBuffers(gpu_buffers_);

    std::cout << "[AWG Thread] Buffers zeroed (GPU + pinned)" << std::endl;
}

void AWGInterface::processCommand(const WaveformCommand& cmd) {
    switch (cmd.type) {
        case AWGCommandType::INITIALIZE:
            if (cmd.initialize_data) {
                initializeAWG(cmd.initialize_data->amplitudes_mv);
                // initializeAWG sets last_result_
            } else {
                std::cerr << "[AWG Thread] Initialize command missing data" << std::endl;
                last_result_ = CommandResult{false, "Initialize command missing data"};
            }
            break;

        case AWGCommandType::START:
            startStreaming();
            // startStreaming signals completion immediately, then blocks
            // Don't signal again at end of processCommand
            return;  // Early return - already signaled

        case AWGCommandType::STOP:
            stopAWG();
            // stopAWG sets last_result_
            break;

        case AWGCommandType::FINISH:
            finishAWG();
            // finishAWG sets last_result_
            break;

        case AWGCommandType::WAVEFORM_BATCH:
            if (cmd.waveform_batch_data) {
                uploadBatchToGPU(*cmd.waveform_batch_data);
                // uploadBatchToGPU sets last_result_
            } else {
                std::cerr << "[AWG Thread] WaveformBatch command missing data" << std::endl;
                last_result_ = CommandResult{false, "WaveformBatch command missing data"};
            }
            break;

        default:
            std::cerr << "[AWG Thread] Unknown command type" << std::endl;
            last_result_ = CommandResult{false, "Unknown command type"};
            break;
    }

    // Signal command completion
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        result_ready_ = true;
    }
    result_cv_.notify_one();
}

// Shared memory accessors
bool AWGInterface::hasSharedMemory() const {
    return shared_memory_ && shared_memory_->isCreated();
}

std::string AWGInterface::getSharedMemoryName() const {
    if (shared_memory_) {
        return shared_memory_->getName();
    }
    return "";
}

size_t AWGInterface::getSharedMemorySize() const {
    if (shared_memory_) {
        return shared_memory_->getSize();
    }
    return 0;
}

void* AWGInterface::getSharedMemoryPointer() const {
    if (shared_memory_) {
        return shared_memory_->getPointer();
    }
    return nullptr;
}

// Signal command result ready
void AWGInterface::signalCommandComplete() {
    std::lock_guard<std::mutex> lock(result_mutex_);
    result_ready_ = true;
    result_cv_.notify_one();
}

// FIFO Streaming Implementation
bool AWGInterface::startStreaming() {
    using namespace aod::config;

    std::cout << "[AWG Thread] Starting FIFO streaming..." << std::endl;

    // Validate state
    if (state_ != AWGState::INITIALIZED) {
        std::string error = "Must be INITIALIZED to start (current state: " +
                           std::to_string(static_cast<int>(state_.load())) + ")";
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        signalCommandComplete();  // Signal failure
        return false;
    }

    if (batch_ids_.empty()) {
        std::string error = "No batches loaded";
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        signalCommandComplete();  // Signal failure
        return false;
    }

    // Set streaming state
    state_ = AWGState::STREAMING;
    streaming_active_ = true;
    stop_requested_ = false;
    finish_requested_ = false;
    bytes_available_ = 0;
    write_position_ = 0;

    std::cout << "[AWG Thread] Will play " << batch_ids_.size() << " batches in order: ";
    for (int id : batch_ids_) std::cout << id << " ";
    std::cout << std::endl;

    // Signal success immediately - streaming has started
    // Client shouldn't wait for streaming to finish
    last_result_ = CommandResult{true, ""};
    signalCommandComplete();

    std::cout << "[AWG Thread] START command signaled as successful, entering streaming loop" << std::endl;

    // Main batch loop - use while loop to handle dynamic batch insertion
    size_t next_batch_to_play = 0;
    while (!stop_requested_) {
        // Between-batch command processing (may insert new batches)
        if (!processBatchTransition()) {
            break;  // Stop requested
        }

        // Re-check batch count after processing commands
        if (next_batch_to_play >= batch_ids_.size()) {
            std::cout << "[AWG Thread] All batches played" << std::endl;
            break;  // No more batches
        }

        // Play current batch
        int batch_id = batch_ids_[next_batch_to_play];
        std::cout << "[AWG Thread] Playing batch index " << next_batch_to_play
                  << " (batch_id=" << batch_id << ")" << std::endl;

        if (!playBatch(batch_id)) {
            streaming_active_ = false;
            state_ = AWGState::INITIALIZED;
            return false;  // Error during batch
        }

        next_batch_to_play++;
    }

    // After all batches: check if we should enter idle or exit
    if (!stop_requested_ && !finish_requested_) {
        // No stop/finish requested - enter idle loop
        idleStreamingLoop();
    } else if (finish_requested_) {
        std::cout << "[AWG Thread] FINISH complete - all batches played, exiting streaming" << std::endl;
    }

    // Stop card when exiting streaming (for both STOP and FINISH)
    std::cout << "[AWG Thread] Stopping card..." << std::endl;
    spcm_dwSetParam_i32(card_handle_, SPC_M2CMD, M2CMD_CARD_STOP);

    // If FINISH was requested, clear batches and zero GPU arrays
    if (finish_requested_) {
        std::cout << "[AWG Thread] FINISH - clearing batches and zeroing GPU arrays..." << std::endl;
        zeroBuffer();
        batch_ids_.clear();
        batch_trigger_types_.clear();
        batch_start_indices_.clear();
        batch_lengths_.clear();
        batch_timestep_duration_.clear();
        max_timestep_index_ = 0;
        std::cout << "[AWG Thread] All batches cleared" << std::endl;
    }

    streaming_active_ = false;
    state_ = AWGState::INITIALIZED;
    last_result_ = CommandResult{true, ""};

    std::cout << "[AWG Thread] Streaming stopped" << std::endl;
    return true;
}

bool AWGInterface::processBatchTransition() {
    std::cout << "[AWG Thread] Between batches - draining command queue..." << std::endl;

    int commands_processed = 0;

    // Drain all queued commands
    while (true) {
        auto opt_cmd = command_queue_->try_pop();
        if (!opt_cmd.has_value()) {
            break;  // No more commands
        }

        WaveformCommand& cmd = opt_cmd.value();
        commands_processed++;

        // Process command
        if (cmd.type == AWGCommandType::WAVEFORM_BATCH) {
            std::cout << "[AWG Thread] Processing queued WAVEFORM_BATCH during streaming" << std::endl;
            uploadBatchToGPU(*cmd.waveform_batch_data);
            signalCommandComplete();

        } else if (cmd.type == AWGCommandType::STOP) {
            std::cout << "[AWG Thread] STOP command received - will exit after current batch" << std::endl;
            stop_requested_ = true;
            signalCommandComplete();
            return false;  // Exit streaming immediately

        } else if (cmd.type == AWGCommandType::FINISH) {
            std::cout << "[AWG Thread] FINISH command received - will exit after all batches complete" << std::endl;
            finish_requested_ = true;
            signalCommandComplete();
            // Don't return false - continue playing remaining batches

        } else {
            std::cerr << "[AWG Thread] Ignoring command type "
                      << static_cast<int>(cmd.type) << " during streaming" << std::endl;
            last_result_ = CommandResult{false, "Command not supported during streaming"};
            signalCommandComplete();
        }
    }

    if (commands_processed > 0) {
        std::cout << "[AWG Thread] Processed " << commands_processed << " commands" << std::endl;
    }

    return true;  // Continue streaming
}

bool AWGInterface::playBatch(int batch_id) {
    using namespace aod::config;

    std::cout << "[AWG Thread] ═══════════════════════════════════" << std::endl;
    std::cout << "[AWG Thread] Playing batch " << batch_id << std::endl;

    // Get batch info
    int start_idx = batch_start_indices_[batch_id];
    int num_timesteps = batch_lengths_[batch_id];                 // Number of timesteps (array length)
    int max_timestep_value = batch_timestep_duration_[batch_id];   // Max timestep value (duration)
    std::string trigger_type = batch_trigger_types_[batch_id];

    std::cout << "[AWG Thread]   Start index: " << start_idx << std::endl;
    std::cout << "[AWG Thread]   Num timesteps (array length): " << num_timesteps << std::endl;
    std::cout << "[AWG Thread]   Max timestep value (duration): " << max_timestep_value << std::endl;
    std::cout << "[AWG Thread]   Trigger: " << trigger_type << std::endl;

    // Calculate total waveform size using max timestep value (NOT num_timesteps!)
    int num_channels = __builtin_popcount(AWG_CHANNEL_MASK);
    size_t samples_per_channel = max_timestep_value * WAVEFORM_TIMESTEP;
    size_t total_samples = samples_per_channel * num_channels;
    size_t waveform_size_bytes = total_samples * sizeof(int16_t);

    std::cout << "[AWG Thread]   Samples per channel: " << samples_per_channel << std::endl;
    std::cout << "[AWG Thread]   Total samples: " << total_samples << std::endl;
    std::cout << "[AWG Thread]   Total size: " << waveform_size_bytes / 1024 << " KB" << std::endl;
    std::cout << "[AWG Thread]   Duration: "
              << (double)samples_per_channel / AWG_SAMPLE_RATE * 1000.0 << " ms" << std::endl;

    // 1. Configure trigger based on batch trigger_type
    if (trigger_type == "software") {
        std::cout << "[AWG Thread]   Configuring software trigger..." << std::endl;
        spcm_dwSetParam_i32(card_handle_, SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE);
        spcm_dwSetParam_i32(card_handle_, SPC_TRIG_ANDMASK, 0);
    } else if (trigger_type == "external") {
        std::cout << "[AWG Thread]   Configuring external trigger (EXT0)..." << std::endl;
        spcm_dwSetParam_i32(card_handle_, SPC_TRIG_ORMASK, SPC_TMASK_EXT0);
        spcm_dwSetParam_i32(card_handle_, SPC_TRIG_ANDMASK, 0);
        // Configure external trigger mode: positive edge
        spcm_dwSetParam_i32(card_handle_, SPC_TRIG_EXT0_MODE, SPC_TM_POS);
        // Configure external trigger parameters
        spcm_dwSetParam_i32(card_handle_, SPC_TRIG_TERM, 0);  // High-impedance termination
        spcm_dwSetParam_i32(card_handle_, SPC_TRIG_EXT0_ACDC, COUPLING_DC);  // DC coupling
        spcm_dwSetParam_i32(card_handle_, SPC_TRIG_EXT0_LEVEL0, 2000);  // Trigger level: 2000 mV
        std::cout << "[AWG Thread]     Mode: Positive edge" << std::endl;
        std::cout << "[AWG Thread]     Termination: High-impedance" << std::endl;
        std::cout << "[AWG Thread]     Coupling: DC" << std::endl;
        std::cout << "[AWG Thread]     Level: 2000 mV" << std::endl;
    } else {
        std::string error = "Invalid trigger_type: " + trigger_type;
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    // 2. Use GPU pinned buffer for DMA
    size_t buffer_bytes = gpu_buffers_.total_samples * sizeof(int16_t);
    size_t initial_fill = std::min(waveform_size_bytes, buffer_bytes);

    std::cout << "[AWG Thread]   DMA buffer size: " << buffer_bytes / 1024 << " KB (GPU pinned)" << std::endl;
    std::cout << "[AWG Thread]   Initial fill: " << initial_fill / 1024 << " KB" << std::endl;

    // 3. Copy initial buffer from GPU samples to pinned memory
    std::cout << "[AWG Thread]   Copying initial buffer from GPU..." << std::endl;

    // Copy from beginning of d_samples to beginning of h_samples_pinned
    cudaError_t cuda_err = cudaMemcpy(
        gpu_buffers_.h_samples_pinned,
        gpu_buffers_.d_samples,
        initial_fill,
        cudaMemcpyDeviceToHost
    );

    if (cuda_err != cudaSuccess) {
        std::string error = std::string("Initial GPU copy failed: ") + cudaGetErrorString(cuda_err);
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    std::cout << "[AWG Thread]   Copied " << initial_fill << " bytes from GPU" << std::endl;

    // 4. Set timeout for DMA operations
    spcm_dwSetParam_i32(card_handle_, SPC_TIMEOUT, 5000);  // 5 second timeout

    // 5. Define DMA transfer for this waveform (use actual waveform size, not buffer size)
    size_t dma_buffer_size = std::min(waveform_size_bytes, buffer_bytes);
    spcm_dwDefTransfer_i64(card_handle_, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD,
                          notify_size_,
                          (void*) gpu_buffers_.h_samples_pinned,
                          0,
                          dma_buffer_size);

    // 6. Mark initial data available to card
    spcm_dwSetParam_i64(card_handle_, SPC_DATA_AVAIL_CARD_LEN, initial_fill);

    // 7. Start DMA transfer and wait for first notify
    std::cout << "[AWG Thread]   Starting DMA transfer..." << std::endl;
    int32 dwErr = spcm_dwSetParam_i32(card_handle_, SPC_M2CMD,
                                      M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA);
    if (dwErr != ERR_OK) {
        char szError[256];
        spcm_dwGetErrorInfo_i32(card_handle_, nullptr, nullptr, szError);
        std::string error = std::string("Failed to start DMA: ") + szError;
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    std::cout << "[AWG Thread]   DMA started, first notify complete" << std::endl;

    // 8. Wait for hardware buffer to fill before starting card
    bool hw_ready = false;
    int wait_iterations = 0;
    while (!hw_ready && wait_iterations < 10000) {  // Max 1 second (100us * 10000)
        int64 fill_promille;
        spcm_dwGetParam_i64(card_handle_, SPC_FILLSIZEPROMILLE, &fill_promille);

        if (fill_promille >= 1000) {  // 100% full
            hw_ready = true;
            std::cout << "[AWG Thread]   Hardware buffer full (100%)" << std::endl;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            wait_iterations++;
        }
    }

    if (!hw_ready) {
        std::cerr << "[AWG Thread] Warning: Hardware buffer did not fill completely, starting anyway" << std::endl;
    }

    // 9. Start card output
    dwErr = spcm_dwSetParam_i32(card_handle_, SPC_M2CMD,
                                M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);
    if (dwErr != ERR_OK) {
        char szError[256];
        spcm_dwGetErrorInfo_i32(card_handle_, nullptr, nullptr, szError);
        std::string error = std::string("Failed to start card: ") + szError;
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    //std::cout << "[AWG Thread]   Card started, playback active" << std::endl;

    // 8. FIFO streaming loop or wait for completion
    bool success;
    if (waveform_size_bytes > buffer_bytes) {
        // Large waveform - need FIFO streaming
        std::cout << "[AWG Thread]   Large waveform - entering FIFO loop" << std::endl;
        success = fifoStreamLoop(waveform_size_bytes, buffer_bytes);
    } else {
        // Small waveform - just wait for completion
        std::cout << "[AWG Thread]   Small waveform - waiting for completion" << std::endl;
        success = waitForCompletion();
    }

    std::cout << "[AWG Thread] Batch " << batch_id
              << (success ? " completed" : " failed") << std::endl;
    std::cout << "[AWG Thread] ═══════════════════════════════════" << std::endl;

    return success;
}

bool AWGInterface::fifoStreamLoop(size_t total_waveform_bytes, size_t buffer_size) {
    // Follow Spectrum FIFO example exactly
    size_t bytes_sent_to_card = buffer_size;  // Already sent initial fill!

    // std::cout << "[AWG Thread] FIFO loop started" << std::endl;
    // std::cout << "[AWG Thread]   Already sent: " << bytes_sent_to_card << " bytes (initial fill)" << std::endl;
    // std::cout << "[AWG Thread]   Total waveform: " << total_waveform_bytes << " bytes" << std::endl;
    // std::cout << "[AWG Thread]   Remaining: " << (total_waveform_bytes - bytes_sent_to_card) << " bytes" << std::endl;

    int32 dwError = 0;
    while (!dwError && bytes_sent_to_card < total_waveform_bytes) {
        // Read available bytes from card (how much it consumed, now free)
        int64 avail_bytes;
        spcm_dwGetParam_i64(card_handle_, SPC_DATA_AVAIL_USER_LEN, &avail_bytes);

        // Only process if we have at least notify_size available (like Spectrum example)
        if (avail_bytes >= static_cast<int64>(notify_size_)) {
            int64 user_pos;
            spcm_dwGetParam_i64(card_handle_, SPC_DATA_AVAIL_USER_POS, &user_pos);

            // Verify position consistency
            if (user_pos != static_cast<int64>(write_position_)) {
                std::cerr << "[AWG Thread] WARNING: Position mismatch! Expected " << write_position_
                          << ", card reports " << user_pos << std::endl;
            }
            write_position_ = user_pos;

            // Process exactly notify_size bytes (like Spectrum example)
            size_t bytes_to_mark = notify_size_;

            // Don't exceed remaining waveform
            size_t bytes_remaining = total_waveform_bytes - bytes_sent_to_card;
            if (bytes_to_mark > bytes_remaining) {
                bytes_to_mark = bytes_remaining;
            }

            // Copy from GPU samples array to pinned memory at same circular position
            size_t buffer_offset = write_position_ / sizeof(int16_t);

            cudaError_t cuda_err = cudaMemcpy(
                gpu_buffers_.h_samples_pinned + buffer_offset,
                gpu_buffers_.d_samples + buffer_offset,
                bytes_to_mark,
                cudaMemcpyDeviceToHost
            );

            if (cuda_err != cudaSuccess) {
                std::cerr << "[AWG Thread] GPU copy failed: " << cudaGetErrorString(cuda_err) << std::endl;
                return false;
            }

            // Mark data available to card
            spcm_dwSetParam_i64(card_handle_, SPC_DATA_AVAIL_CARD_LEN, bytes_to_mark);

            bytes_sent_to_card += bytes_to_mark;
            write_position_ = (write_position_ + bytes_to_mark) % buffer_size;
        }

        // Wait for next notification
        if (!dwError)
            dwError = spcm_dwSetParam_i32(card_handle_, SPC_M2CMD, M2CMD_DATA_WAITDMA);
    }

    // Check for errors
    if (dwError) {
        if (dwError == ERR_FIFOHWOVERRUN && bytes_sent_to_card >= total_waveform_bytes) {
            // Overrun is expected if we've sent all the data
            std::cout << "[AWG Thread] FIFO overrun (expected - all data sent)" << std::endl;
        } else {
            // Unexpected error or premature overrun
            char szError[256];
            spcm_dwGetErrorInfo_i32(card_handle_, nullptr, nullptr, szError);
            std::cerr << "[AWG Thread] FIFO error: " << szError << std::endl;
            std::cerr << "[AWG Thread] Sent " << bytes_sent_to_card << " / "
                      << total_waveform_bytes << " bytes before error" << std::endl;
            return false;
        }
    }

    std::cout << "[AWG Thread] FIFO loop complete - sent all " << bytes_sent_to_card << " bytes" << std::endl;

    // Wait for final playback completion
    return waitForCompletion();
}

bool AWGInterface::waitForCompletion() {
    std::cout << "[AWG Thread] Waiting for playback completion..." << std::endl;

    bool finished = false;
    while (!finished) {
        int32 status;
        spcm_dwGetParam_i32(card_handle_, SPC_M2STATUS, &status);

        if (status & M2STAT_CARD_READY) {
            std::cout << "[AWG Thread] Card reports READY - playback complete" << std::endl;
            finished = true;
        }

        // Check for FIFO overrun (expected at end of single replay)
        if (status & ERR_FIFOHWOVERRUN) {
            std::cout << "[AWG Thread] FIFO overrun (expected - batch complete)" << std::endl;
            // Stop DMA
            spcm_dwSetParam_i32(card_handle_, SPC_M2CMD, M2CMD_DATA_STOPDMA);
            finished = true;
        }

        if (!finished) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    std::cout << "[AWG Thread] Playback completion confirmed" << std::endl;
    return true;
}

void AWGInterface::idleStreamingLoop() {
    std::cout << "[AWG Thread] All batches complete - entering idle streaming..." << std::endl;
    std::cout << "[AWG Thread] Staying in STREAMING state, waiting for STOP/FINISH or new batches" << std::endl;

    while (!stop_requested_ && !finish_requested_) {
        // Check command queue with timeout
        auto opt_cmd = command_queue_->try_pop();
        if (opt_cmd.has_value()) {
            WaveformCommand& cmd = opt_cmd.value();

            if (cmd.type == AWGCommandType::STOP) {
                std::cout << "[AWG Thread] STOP received in idle loop" << std::endl;
                stop_requested_ = true;
                signalCommandComplete();
                break;

            } else if (cmd.type == AWGCommandType::FINISH) {
                std::cout << "[AWG Thread] FINISH received in idle loop" << std::endl;
                finish_requested_ = true;
                signalCommandComplete();
                break;

            } else if (cmd.type == AWGCommandType::WAVEFORM_BATCH) {
                std::cout << "[AWG Thread] New batch added during idle streaming" << std::endl;
                uploadBatchToGPU(*cmd.waveform_batch_data);
                signalCommandComplete();
                // Stay in idle loop for now (future: could restart batch playback)

            } else {
                std::cerr << "[AWG Thread] Ignoring command during idle streaming" << std::endl;
                last_result_ = CommandResult{false, "Command not supported during idle streaming"};
                signalCommandComplete();
            }
        } else {
            // No commands - sleep briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    std::cout << "[AWG Thread] Exiting idle streaming loop" << std::endl;
}

} // namespace aod
