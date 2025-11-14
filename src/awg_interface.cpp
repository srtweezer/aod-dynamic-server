#include "awg_interface.h"
#include <config.h>
#include <iostream>

namespace aod {

AWGInterface::AWGInterface()
    : connected_(false), card_handle_(nullptr) {
}

AWGInterface::~AWGInterface() {
    disconnect();
}

bool AWGInterface::connect() {
    using namespace aod::config;

    std::cout << "[AWG] Scanning for Spectrum AWG devices..." << std::endl;
    std::cout << "[AWG]   Target serial number: " << AWG_SERIAL_NUMBER << std::endl;
    std::cout << "[AWG]   Sample rate: " << AWG_SAMPLE_RATE << " S/s" << std::endl;
    std::cout << "[AWG]   Channels: " << AWG_NUM_CHANNELS << std::endl;
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
        connected_ = true;
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

void AWGInterface::disconnect() {
    if (connected_) {
        std::cout << "[AWG] Disconnecting from AWG..." << std::endl;

        if (card_handle_) {
            spcm_vClose(card_handle_);
            card_handle_ = nullptr;
        }

        connected_ = false;
        std::cout << "[AWG] Disconnected" << std::endl;
    }
}

} // namespace aod
