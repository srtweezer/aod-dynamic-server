#include "gpu_buffers.h"
#include <config.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

namespace aod {

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA] Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while(0)

#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA] Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
        } \
    } while(0)

bool allocateGPUBuffers(GPUBuffers& buffers) {
    using namespace aod::config;

    std::cout << "[GPU] Allocating GPU buffers..." << std::endl;

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(GPU_DEVICE_ID));

    // Calculate dimensions
    buffers.num_channels = __builtin_popcount(AWG_CHANNEL_MASK);
    buffers.num_tones = AOD_MAX_TONES;
    buffers.timestep = WAVEFORM_TIMESTEP;
    buffers.num_chunks = AOD_DMA_BUFFER_SIZE / (2 * WAVEFORM_TIMESTEP);

    // Calculate sizes
    buffers.total_samples = buffers.num_chunks * buffers.num_channels * buffers.timestep;
    buffers.tone_params_size = buffers.num_chunks * buffers.num_channels * buffers.num_tones;

    std::cout << "[GPU] Buffer dimensions:" << std::endl;
    std::cout << "[GPU]   Chunks: " << buffers.num_chunks << std::endl;
    std::cout << "[GPU]   Channels: " << buffers.num_channels << std::endl;
    std::cout << "[GPU]   Tones per channel: " << buffers.num_tones << std::endl;
    std::cout << "[GPU]   Timestep: " << buffers.timestep << " samples" << std::endl;

    // Calculate memory sizes
    size_t samples_bytes = buffers.total_samples * sizeof(int16_t);
    size_t tone_params_bytes = buffers.tone_params_size * sizeof(float);

    std::cout << "[GPU] Memory allocation:" << std::endl;
    std::cout << "[GPU]   Output samples: " << samples_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "[GPU]   Amplitudes: " << tone_params_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "[GPU]   Phases: " << tone_params_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "[GPU]   Frequencies: " << tone_params_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "[GPU]   Total device: " << (samples_bytes + 3 * tone_params_bytes) / (1024*1024) << " MB" << std::endl;
    std::cout << "[GPU]   Pinned host: " << samples_bytes / (1024*1024) << " MB" << std::endl;

    // Allocate device memory
    std::cout << "[GPU] Allocating device memory..." << std::endl;
    CUDA_CHECK(cudaMalloc(&buffers.d_samples, samples_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_amplitudes, tone_params_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_phases, tone_params_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_frequencies, tone_params_bytes));

    // Allocate pinned host memory
    std::cout << "[GPU] Allocating pinned host memory..." << std::endl;
    CUDA_CHECK(cudaMallocHost(&buffers.h_samples_pinned, samples_bytes));

    // Zero all buffers initially
    std::cout << "[GPU] Zeroing buffers..." << std::endl;
    CUDA_CHECK(cudaMemset(buffers.d_samples, 0, samples_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_amplitudes, 0, tone_params_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_phases, 0, tone_params_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_frequencies, 0, tone_params_bytes));
    std::memset(buffers.h_samples_pinned, 0, samples_bytes);

    std::cout << "[GPU] GPU buffers allocated successfully" << std::endl;
    return true;
}

void freeGPUBuffers(GPUBuffers& buffers) {
    std::cout << "[GPU] Freeing GPU buffers..." << std::endl;

    if (buffers.d_samples) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_samples));
        buffers.d_samples = nullptr;
    }

    if (buffers.d_amplitudes) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_amplitudes));
        buffers.d_amplitudes = nullptr;
    }

    if (buffers.d_phases) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_phases));
        buffers.d_phases = nullptr;
    }

    if (buffers.d_frequencies) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_frequencies));
        buffers.d_frequencies = nullptr;
    }

    if (buffers.h_samples_pinned) {
        CUDA_CHECK_VOID(cudaFreeHost(buffers.h_samples_pinned));
        buffers.h_samples_pinned = nullptr;
    }

    std::cout << "[GPU] GPU buffers freed" << std::endl;
}

void zeroGPUBuffers(GPUBuffers& buffers) {
    if (!buffers.d_samples) {
        return;  // Not allocated yet
    }

    size_t samples_bytes = buffers.total_samples * sizeof(int16_t);
    size_t tone_params_bytes = buffers.tone_params_size * sizeof(float);

    CUDA_CHECK_VOID(cudaMemset(buffers.d_samples, 0, samples_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_amplitudes, 0, tone_params_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_phases, 0, tone_params_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_frequencies, 0, tone_params_bytes));

    if (buffers.h_samples_pinned) {
        std::memset(buffers.h_samples_pinned, 0, samples_bytes);
    }

    std::cout << "[GPU] GPU buffers zeroed" << std::endl;
}

} // namespace aod
