#include "gpu_buffers.h"
#include <config.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <chrono>

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
    buffers.max_timesteps = MAX_WAVEFORM_TIMESTEPS;

    // Calculate sizes
    // AOD_DMA_BUFFER_SIZE is total bytes for ALL channels combined
    buffers.total_samples = AOD_DMA_BUFFER_SIZE / sizeof(int16_t);  // Total int16 samples across all channels
    buffers.num_chunks = buffers.total_samples / (buffers.num_channels * buffers.timestep);  // Chunks across all channels
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
    std::cout << "[GPU]   Temp buffers (each): " << batch_arrays_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "[GPU]   Total device: " << (samples_bytes + 3 * tone_params_bytes +
                                                batch_timesteps_bytes + batch_flags_bytes +
                                                6 * batch_arrays_bytes) / (1024*1024) << " MB" << std::endl;
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

    // Allocate temporary buffers for strided copy (same size as batch arrays)
    CUDA_CHECK(cudaMalloc(&buffers.d_temp_frequencies, batch_arrays_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_temp_amplitudes, batch_arrays_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_temp_offset_phases, batch_arrays_bytes));

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

    if (buffers.d_temp_frequencies) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_temp_frequencies));
        buffers.d_temp_frequencies = nullptr;
    }

    if (buffers.d_temp_amplitudes) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_temp_amplitudes));
        buffers.d_temp_amplitudes = nullptr;
    }

    if (buffers.d_temp_offset_phases) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_temp_offset_phases));
        buffers.d_temp_offset_phases = nullptr;
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

// CUDA kernel to validate timesteps are strictly ascending
// Each thread checks one pair (timesteps[i] < timesteps[i+1])
// Sets result to 0 if any pair violates ascending order
__global__ void validateAscendingKernel(const int32_t* timesteps,
                                        int num_timesteps,
                                        int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_timesteps - 1) {
        // Check if timesteps[idx] < timesteps[idx + 1]
        if (timesteps[idx] >= timesteps[idx + 1]) {
            atomicMin(result, 0);  // Mark as invalid
        }
    }
}

// CUDA kernel to expand tone arrays from num_tones to max_tones with zero-padding
// Each thread handles one [timestep][channel] slice
__global__ void expandTonesKernel(
    const float* __restrict__ src,  // Compact: [timestep][channel][num_tones]
    float* __restrict__ dst,        // Expanded: [timestep][channel][max_tones]
    int num_timesteps,
    int num_channels,
    int num_tones,
    int max_tones,
    int dst_offset) {

    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_slices = num_timesteps * num_channels;

    if (idx < total_slices) {
        int t = idx / num_channels;
        int ch = idx % num_channels;

        // Source base index in compact array: [t][ch][tone]
        int src_base = t * num_channels * num_tones + ch * num_tones;

        // Destination base index in expanded array: [(dst_offset + t)][ch][tone]
        int dst_base = (dst_offset + t) * num_channels * max_tones + ch * max_tones;

        // Copy actual tones
        for (int tone = 0; tone < num_tones; tone++) {
            dst[dst_base + tone] = src[src_base + tone];
        }

        // Zero-pad remaining tones
        for (int tone = num_tones; tone < max_tones; tone++) {
            dst[dst_base + tone] = 0.0f;
        }
    }
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
    auto t_start = std::chrono::high_resolution_clock::now();

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

    auto t_simple_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_timesteps + target_offset,
                               h_timesteps,
                               timesteps_bytes,
                               cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_do_generate + target_offset,
                               h_do_generate,
                               flags_bytes,
                               cudaMemcpyHostToDevice));
    auto t_simple_end = std::chrono::high_resolution_clock::now();
    auto simple_us = std::chrono::duration_cast<std::chrono::microseconds>(t_simple_end - t_simple_start).count();

    // For frequency/amplitude/phase arrays: do strided copy to pad num_tones → AOD_MAX_TONES
    // Client data layout: [timestep][channel][num_tones]
    // GPU data layout: [timestep][channel][AOD_MAX_TONES]

    auto t_tones_start = std::chrono::high_resolution_clock::now();

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
        // GPU-accelerated striding: copy compact data then expand on GPU

        // 1. Copy compact data to temp buffers (smaller transfer)
        size_t compact_elements = num_timesteps * num_channels * num_tones;
        size_t compact_bytes = compact_elements * sizeof(float);

        auto t_compact_copy_start = std::chrono::high_resolution_clock::now();
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_temp_frequencies, h_frequencies,
                                   compact_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_temp_amplitudes, h_amplitudes,
                                   compact_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_temp_offset_phases, h_offset_phases,
                                   compact_bytes, cudaMemcpyHostToDevice));
        auto t_compact_copy_end = std::chrono::high_resolution_clock::now();
        auto compact_copy_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_compact_copy_end - t_compact_copy_start).count();

        // 2. Launch kernel to expand in parallel
        int total_slices = num_timesteps * num_channels;
        int block_size = 256;
        int grid_size = (total_slices + block_size - 1) / block_size;

        auto t_kernel_start = std::chrono::high_resolution_clock::now();
        expandTonesKernel<<<grid_size, block_size>>>(
            buffers.d_temp_frequencies, buffers.d_batch_frequencies,
            num_timesteps, num_channels, num_tones, max_tones, target_offset);
        expandTonesKernel<<<grid_size, block_size>>>(
            buffers.d_temp_amplitudes, buffers.d_batch_amplitudes,
            num_timesteps, num_channels, num_tones, max_tones, target_offset);
        expandTonesKernel<<<grid_size, block_size>>>(
            buffers.d_temp_offset_phases, buffers.d_batch_offset_phases,
            num_timesteps, num_channels, num_tones, max_tones, target_offset);

        CUDA_CHECK_VOID(cudaDeviceSynchronize());
        auto t_kernel_end = std::chrono::high_resolution_clock::now();
        auto kernel_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_kernel_end - t_kernel_start).count();

        std::cout << "[GPU]   ├─ Compact copy: " << compact_copy_us << " μs" << std::endl;
        std::cout << "[GPU]   └─ Kernel expand: " << kernel_us << " μs" << std::endl;
    }

    auto t_tones_end = std::chrono::high_resolution_clock::now();
    auto tones_us = std::chrono::duration_cast<std::chrono::microseconds>(t_tones_end - t_tones_start).count();

    auto t_total_end = std::chrono::high_resolution_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t_total_end - t_start).count();

    std::cout << "[GPU] Uploaded " << num_timesteps << " timesteps to GPU at offset "
              << target_offset << " (tones: " << num_tones << " → " << max_tones << ")" << std::endl;
    std::cout << "[GPU] ─── GPU Copy Timing ───" << std::endl;
    std::cout << "[GPU]   Timesteps/flags:   " << simple_us << " μs" << std::endl;
    std::cout << "[GPU]   Tone arrays:       " << tones_us << " μs "
              << (num_tones == max_tones ? "(direct)" : "(strided+padded)") << std::endl;
    std::cout << "[GPU]   Total GPU copy:    " << total_us << " μs" << std::endl;
    std::cout << "[GPU] ─────────────────────────" << std::endl;
}

bool validateTimestepsAscending(const int32_t* d_timesteps, int num_timesteps) {
    if (num_timesteps <= 1) {
        return true;  // Single or empty is trivially valid
    }

    // Allocate device memory for result flag
    int* d_result;
    CUDA_CHECK_VOID(cudaMalloc(&d_result, sizeof(int)));

    // Initialize result to 1 (valid)
    int init_result = 1;
    CUDA_CHECK_VOID(cudaMemcpy(d_result, &init_result, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel - each thread checks one pair
    int num_pairs = num_timesteps - 1;
    int block_size = 256;
    int grid_size = (num_pairs + block_size - 1) / block_size;

    validateAscendingKernel<<<grid_size, block_size>>>(d_timesteps, num_timesteps, d_result);

    // Get result
    int result;
    CUDA_CHECK_VOID(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK_VOID(cudaFree(d_result));

    return (result == 1);
}

} // namespace aod
