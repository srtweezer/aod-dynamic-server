#include "gpu_buffers.h"
#include <config.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <vector>

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
    buffers.max_timesteps = MAX_WAVEFORM_TIMESTEPS;

    // Calculate sizes
    buffers.total_samples = buffers.num_chunks * buffers.num_channels * buffers.timestep;
    buffers.tone_params_size = buffers.num_chunks * buffers.num_channels * buffers.num_tones;
    buffers.batch_arrays_size = buffers.max_timesteps * buffers.num_channels * buffers.num_tones;

    std::cout << "[GPU] Buffer dimensions:" << std::endl;
    std::cout << "[GPU]   Chunks: " << buffers.num_chunks << std::endl;
    std::cout << "[GPU]   Channels: " << buffers.num_channels << std::endl;
    std::cout << "[GPU]   Tones per channel: " << buffers.num_tones << std::endl;
    std::cout << "[GPU]   Timestep: " << buffers.timestep << " samples" << std::endl;
    std::cout << "[GPU]   Max batch timesteps: " << buffers.max_timesteps << std::endl;

    // Calculate memory sizes
    size_t samples_bytes = buffers.total_samples * sizeof(int16_t);
    size_t tone_params_bytes = buffers.tone_params_size * sizeof(float);
    size_t batch_timesteps_bytes = buffers.max_timesteps * sizeof(int32_t);
    size_t batch_flags_bytes = buffers.max_timesteps * sizeof(bool);
    size_t batch_arrays_bytes = buffers.batch_arrays_size * sizeof(float);

    std::cout << "[GPU] Memory allocation:" << std::endl;
    std::cout << "[GPU]   Output samples: " << samples_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "[GPU]   Amplitudes: " << tone_params_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "[GPU]   Phases: " << tone_params_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "[GPU]   Frequencies: " << tone_params_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "[GPU]   Batch timesteps: " << batch_timesteps_bytes / 1024 << " KB" << std::endl;
    std::cout << "[GPU]   Batch flags: " << batch_flags_bytes / 1024 << " KB" << std::endl;
    std::cout << "[GPU]   Batch freq/amp/phase (each): " << batch_arrays_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "[GPU]   Total device: " << (samples_bytes + 3 * tone_params_bytes +
                                                batch_timesteps_bytes + batch_flags_bytes +
                                                3 * batch_arrays_bytes) / (1024*1024) << " MB" << std::endl;
    std::cout << "[GPU]   Pinned host: " << samples_bytes / (1024*1024) << " MB" << std::endl;

    // Allocate device memory
    std::cout << "[GPU] Allocating device memory..." << std::endl;
    CUDA_CHECK(cudaMalloc(&buffers.d_samples, samples_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_amplitudes, tone_params_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_phases, tone_params_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_frequencies, tone_params_bytes));

    // Allocate batch data arrays
    CUDA_CHECK(cudaMalloc(&buffers.d_batch_timesteps, batch_timesteps_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_batch_do_generate, batch_flags_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_batch_frequencies, batch_arrays_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_batch_amplitudes, batch_arrays_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_batch_offset_phases, batch_arrays_bytes));

    // Allocate pinned host memory
    std::cout << "[GPU] Allocating pinned host memory..." << std::endl;
    CUDA_CHECK(cudaMallocHost(&buffers.h_samples_pinned, samples_bytes));

    // Zero all buffers initially
    std::cout << "[GPU] Zeroing buffers..." << std::endl;
    CUDA_CHECK(cudaMemset(buffers.d_samples, 0, samples_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_amplitudes, 0, tone_params_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_phases, 0, tone_params_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_frequencies, 0, tone_params_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_batch_timesteps, 0, batch_timesteps_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_batch_do_generate, 0, batch_flags_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_batch_frequencies, 0, batch_arrays_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_batch_amplitudes, 0, batch_arrays_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_batch_offset_phases, 0, batch_arrays_bytes));
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

    if (buffers.d_batch_timesteps) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_batch_timesteps));
        buffers.d_batch_timesteps = nullptr;
    }

    if (buffers.d_batch_do_generate) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_batch_do_generate));
        buffers.d_batch_do_generate = nullptr;
    }

    if (buffers.d_batch_frequencies) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_batch_frequencies));
        buffers.d_batch_frequencies = nullptr;
    }

    if (buffers.d_batch_amplitudes) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_batch_amplitudes));
        buffers.d_batch_amplitudes = nullptr;
    }

    if (buffers.d_batch_offset_phases) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_batch_offset_phases));
        buffers.d_batch_offset_phases = nullptr;
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
    size_t batch_timesteps_bytes = buffers.max_timesteps * sizeof(int32_t);
    size_t batch_flags_bytes = buffers.max_timesteps * sizeof(bool);
    size_t batch_arrays_bytes = buffers.batch_arrays_size * sizeof(float);

    CUDA_CHECK_VOID(cudaMemset(buffers.d_samples, 0, samples_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_amplitudes, 0, tone_params_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_phases, 0, tone_params_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_frequencies, 0, tone_params_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_timesteps, 0, batch_timesteps_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_do_generate, 0, batch_flags_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_frequencies, 0, batch_arrays_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_amplitudes, 0, batch_arrays_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_offset_phases, 0, batch_arrays_bytes));

    if (buffers.h_samples_pinned) {
        std::memset(buffers.h_samples_pinned, 0, samples_bytes);
    }

    std::cout << "[GPU] GPU buffers zeroed" << std::endl;
}

void uploadBatchDataToGPU(GPUBuffers& buffers,
                          const int32_t* h_timesteps,
                          const uint8_t* h_do_generate,
                          const float* h_frequencies,
                          const float* h_amplitudes,
                          const float* h_offset_phases,
                          int num_timesteps,
                          int num_channels,
                          int num_tones,
                          int target_offset) {
    if (!buffers.d_batch_timesteps) {
        std::cerr << "[GPU] Warning: Batch arrays not allocated" << std::endl;
        return;
    }

    int max_tones = buffers.num_tones;  // AOD_MAX_TONES

    // Validate parameters
    if (num_timesteps <= 0 || target_offset + num_timesteps > static_cast<int>(buffers.max_timesteps)) {
        std::cerr << "[GPU] Error: Invalid parameters. num_timesteps=" << num_timesteps
                  << ", target_offset=" << target_offset
                  << ", max=" << buffers.max_timesteps << std::endl;
        return;
    }

    // Copy timesteps and do_generate directly (no padding needed)
    size_t timesteps_bytes = num_timesteps * sizeof(int32_t);
    size_t flags_bytes = num_timesteps * sizeof(uint8_t);

    CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_timesteps + target_offset,
                               h_timesteps,
                               timesteps_bytes,
                               cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_do_generate + target_offset,
                               h_do_generate,
                               flags_bytes,
                               cudaMemcpyHostToDevice));

    // For frequency/amplitude/phase arrays: do strided copy to pad num_tones → AOD_MAX_TONES
    // Client data layout: [timestep][channel][num_tones]
    // GPU data layout: [timestep][channel][AOD_MAX_TONES]

    if (num_tones == max_tones) {
        // No padding needed - direct copy
        size_t arrays_elements = num_timesteps * num_channels * num_tones;
        size_t arrays_bytes = arrays_elements * sizeof(float);
        int offset_elements = target_offset * num_channels * max_tones;

        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_frequencies + offset_elements,
                                   h_frequencies, arrays_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_amplitudes + offset_elements,
                                   h_amplitudes, arrays_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_offset_phases + offset_elements,
                                   h_offset_phases, arrays_bytes, cudaMemcpyHostToDevice));
    } else {
        // Need strided copy: expand num_tones → max_tones with zero-padding
        // Allocate temporary buffers on CPU for expanded data
        size_t expanded_size = num_timesteps * num_channels * max_tones;
        std::vector<float> expanded_freq(expanded_size, 0.0f);
        std::vector<float> expanded_amp(expanded_size, 0.0f);
        std::vector<float> expanded_phase(expanded_size, 0.0f);

        // Copy and expand
        for (int t = 0; t < num_timesteps; t++) {
            for (int ch = 0; ch < num_channels; ch++) {
                for (int tone = 0; tone < num_tones; tone++) {
                    // Source index: [t][ch][tone] with num_tones
                    int src_idx = t * num_channels * num_tones + ch * num_tones + tone;
                    // Dest index: [t][ch][tone] with max_tones
                    int dst_idx = t * num_channels * max_tones + ch * max_tones + tone;

                    expanded_freq[dst_idx] = h_frequencies[src_idx];
                    expanded_amp[dst_idx] = h_amplitudes[src_idx];
                    expanded_phase[dst_idx] = h_offset_phases[src_idx];
                }
                // Tones beyond num_tones are already zero-initialized
            }
        }

        // Copy expanded data to GPU at target offset
        size_t expanded_bytes = expanded_size * sizeof(float);
        int offset_elements = target_offset * num_channels * max_tones;

        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_frequencies + offset_elements,
                                   expanded_freq.data(), expanded_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_amplitudes + offset_elements,
                                   expanded_amp.data(), expanded_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_offset_phases + offset_elements,
                                   expanded_phase.data(), expanded_bytes, cudaMemcpyHostToDevice));
    }

    std::cout << "[GPU] Uploaded " << num_timesteps << " timesteps to GPU at offset "
              << target_offset << " (tones: " << num_tones << " → " << max_tones << ")" << std::endl;
}

} // namespace aod
