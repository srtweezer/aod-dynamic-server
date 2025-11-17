#include "awg_interface.h"
#include "thread_safe_queue.h"
#include <config.h>
#include <iostream>
#include <cstring>

namespace aod {

AWGInterface::AWGInterface()
    : running_(false),
      shutdown_requested_(false),
      init_complete_(false),
      result_ready_(false),
      connected_(false),
      state_(AWGState::DISCONNECTED),
      card_handle_(nullptr),
      sw_buffer_(nullptr),
      sw_buffer_size_(0),
      notify_size_(0),
      command_queue_(std::make_unique<ThreadSafeQueue<WaveformCommand>>()),
      num_batches_(0),
      next_batch_id_(1) {
    using namespace aod::config;
    max_batches_ = MAX_WAVEFORM_BATCHES;
    batch_queue_ = std::make_unique<WaveformBatch[]>(max_batches_);
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

        // Free GPU buffers first
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

    // 1. Set card mode to FIFO single
    spcm_dwSetParam_i32(card_handle_, SPC_CARDMODE, SPC_REP_FIFO_SINGLE);
    spcm_dwSetParam_i32(card_handle_, SPC_CHENABLE, AWG_CHANNEL_MASK);
    spcm_dwSetParam_i32(card_handle_, SPC_LOOPS, 0);  // Continuous

    // 2. Setup clock
    spcm_dwSetParam_i32(card_handle_, SPC_CLOCKMODE, SPC_CM_INTPLL);
    spcm_dwSetParam_i64(card_handle_, SPC_SAMPLERATE, AWG_SAMPLE_RATE);
    spcm_dwSetParam_i32(card_handle_, SPC_CLOCKOUT, 0);

    // 3. Setup trigger (software trigger, no AND mask)
    spcm_dwSetParam_i32(card_handle_, SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE);
    spcm_dwSetParam_i32(card_handle_, SPC_TRIG_ANDMASK, 0);

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

    // 7. Allocate software buffer if not already allocated
    if (!sw_buffer_) {
        sw_buffer_size_ = AOD_DMA_BUFFER_SIZE;
        notify_size_ = AOD_NOTIFY_SIZE;

        // Allocate page-aligned memory for DMA
        sw_buffer_ = malloc(sw_buffer_size_);  // TODO: Use pvAllocMemPageAligned when available
        if (!sw_buffer_) {
            std::string error = "Failed to allocate " + std::to_string(sw_buffer_size_ / (1024*1024)) + " MB DMA buffer";
            std::cerr << "[AWG Thread] " << error << std::endl;
            last_result_ = CommandResult{false, error};
            return false;
        }

        std::cout << "[AWG Thread] Allocated " << sw_buffer_size_ / (1024*1024)
                  << " MB DMA buffer" << std::endl;
    }

    // 8. Zero the buffer
    zeroBuffer();

    // 9. Define DMA transfer
    spcm_dwDefTransfer_i64(card_handle_, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD,
                           notify_size_, sw_buffer_, 0, sw_buffer_size_);

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

    // Clear waveform batch queue
    num_batches_ = 0;
    std::cout << "[AWG Thread] Waveform batch queue cleared" << std::endl;

    // Set state to INITIALIZED (stops streaming but keeps initialization)
    state_ = AWGState::INITIALIZED;
    last_result_ = CommandResult{true, ""};

    std::cout << "[AWG Thread] AWG stopped successfully" << std::endl;
    return true;
}

bool AWGInterface::storeBatch(const WaveformBatch& batch) {
    using namespace aod::config;

    std::cout << "[AWG Thread] Storing waveform batch..." << std::endl;

    // Check if queue is full
    int current_batches = num_batches_.load();
    if (current_batches >= max_batches_) {
        std::string error = "Batch queue full (" + std::to_string(max_batches_) + " batches)";
        std::cerr << "[AWG Thread] " << error << std::endl;
        last_result_ = CommandResult{false, error};
        return false;
    }

    // Validate batch data
    int num_channels = __builtin_popcount(AWG_CHANNEL_MASK);

    for (size_t i = 0; i < batch.waveforms.size(); i++) {
        const auto& wf = batch.waveforms[i];

        // Calculate expected array size
        size_t expected_size = wf.num_steps * num_channels * wf.num_tones;

        // Validate array sizes
        if (wf.time_steps.size() != static_cast<size_t>(wf.num_steps)) {
            std::string error = "Waveform " + std::to_string(i) + ": time_steps size mismatch";
            last_result_ = CommandResult{false, error};
            return false;
        }

        if (wf.frequencies.size() != expected_size) {
            std::string error = "Waveform " + std::to_string(i) + ": frequencies size mismatch. "
                               "Expected " + std::to_string(expected_size) + ", got " + std::to_string(wf.frequencies.size());
            last_result_ = CommandResult{false, error};
            return false;
        }

        if (wf.amplitudes.size() != expected_size) {
            std::string error = "Waveform " + std::to_string(i) + ": amplitudes size mismatch";
            last_result_ = CommandResult{false, error};
            return false;
        }

        if (wf.offset_phases.size() != expected_size) {
            std::string error = "Waveform " + std::to_string(i) + ": offset_phases size mismatch";
            last_result_ = CommandResult{false, error};
            return false;
        }
    }

    // Store batch in queue
    batch_queue_[current_batches] = batch;
    num_batches_++;

    std::cout << "[AWG Thread] Batch stored successfully (ID: " << batch.batch_id << ")" << std::endl;
    std::cout << "[AWG Thread]   Waveforms: " << batch.waveforms.size() << std::endl;
    std::cout << "[AWG Thread]   Queue: " << num_batches_ << "/" << max_batches_ << std::endl;

    last_result_ = CommandResult{true, ""};
    return true;
}

void AWGInterface::zeroBuffer() {
    // Zero software DMA buffer
    if (sw_buffer_ && sw_buffer_size_ > 0) {
        std::memset(sw_buffer_, 0, sw_buffer_size_);
    }

    // Zero GPU buffers
    zeroGPUBuffers(gpu_buffers_);

    std::cout << "[AWG Thread] Buffers zeroed (CPU + GPU)" << std::endl;
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
            std::cout << "[AWG Thread] START command (not yet implemented)" << std::endl;
            last_result_ = CommandResult{false, "START not yet implemented"};
            break;

        case AWGCommandType::STOP:
            stopAWG();
            // stopAWG sets last_result_
            break;

        case AWGCommandType::WAVEFORM_BATCH:
            if (cmd.waveform_batch_data) {
                storeBatch(cmd.waveform_batch_data->batch);
                // storeBatch sets last_result_
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

} // namespace aod
