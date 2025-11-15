#include "awg_interface.h"
#include "thread_safe_queue.h"
#include <config.h>
#include <iostream>

namespace aod {

AWGInterface::AWGInterface()
    : running_(false),
      shutdown_requested_(false),
      init_complete_(false),
      connected_(false),
      card_handle_(nullptr),
      command_queue_(std::make_unique<ThreadSafeQueue<WaveformCommand>>()) {
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

void AWGInterface::threadLoop() {
    std::cout << "[AWG Thread] Starting..." << std::endl;

    // Initialize hardware
    bool init_success = connectHardware();
    connected_ = init_success;

    // Signal initialization complete
    {
        std::lock_guard<std::mutex> lock(init_mutex_);
        init_complete_ = true;
    }
    init_cv_.notify_one();

    if (!init_success) {
        std::cerr << "[AWG Thread] Failed to connect to hardware" << std::endl;
        running_ = false;
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

        if (card_handle_) {
            spcm_vClose(card_handle_);
            card_handle_ = nullptr;
        }

        connected_ = false;
        std::cout << "[AWG Thread] Disconnected" << std::endl;
    }
}

void AWGInterface::processCommand(const WaveformCommand& cmd) {
    // Placeholder for command processing
    std::cout << "[AWG Thread] Processing command " << cmd.command_id << std::endl;

    // TODO: Implement actual command handling
    // - Parse waveform parameters
    // - Generate waveform data (call GPU kernel)
    // - Queue for FIFO output
}

} // namespace aod
