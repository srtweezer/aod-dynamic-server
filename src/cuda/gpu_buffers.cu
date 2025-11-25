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
    size_t batch_flags_bytes = (buffers.max_timesteps - 1) * sizeof(bool);
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
    CUDA_CHECK(cudaMalloc(&buffers.d_batch_offset_phases_user, batch_arrays_bytes));

    // Allocate temporary buffers for strided copy (same size as batch arrays)
    CUDA_CHECK(cudaMalloc(&buffers.d_temp_frequencies, batch_arrays_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_temp_amplitudes, batch_arrays_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_temp_offset_phases_user, batch_arrays_bytes));

    // Allocate temporary buffer for float64 frequency data
    size_t batch_arrays_bytes_fp64 = buffers.batch_arrays_size * sizeof(double);
    CUDA_CHECK(cudaMalloc(&buffers.d_temp_frequencies_fp64, batch_arrays_bytes_fp64));

    // Allocate computed phase arrays
    CUDA_CHECK(cudaMalloc(&buffers.d_batch_offset_phases, batch_arrays_bytes));
    size_t phase_corrections_bytes = (buffers.max_timesteps - 1) *
                                      buffers.num_channels *
                                      buffers.num_tones * sizeof(float);
    CUDA_CHECK(cudaMalloc(&buffers.d_batch_phase_corrections, phase_corrections_bytes));

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
    CUDA_CHECK(cudaMemset(buffers.d_batch_offset_phases_user, 0, batch_arrays_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_batch_offset_phases, 0, batch_arrays_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_batch_phase_corrections, 0, phase_corrections_bytes));
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

    if (buffers.d_batch_offset_phases_user) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_batch_offset_phases_user));
        buffers.d_batch_offset_phases_user = nullptr;
    }

    if (buffers.d_temp_frequencies) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_temp_frequencies));
        buffers.d_temp_frequencies = nullptr;
    }

    if (buffers.d_temp_amplitudes) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_temp_amplitudes));
        buffers.d_temp_amplitudes = nullptr;
    }

    if (buffers.d_temp_offset_phases_user) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_temp_offset_phases_user));
        buffers.d_temp_offset_phases_user = nullptr;
    }

    if (buffers.d_temp_frequencies_fp64) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_temp_frequencies_fp64));
        buffers.d_temp_frequencies_fp64 = nullptr;
    }

    if (buffers.d_batch_offset_phases) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_batch_offset_phases));
        buffers.d_batch_offset_phases = nullptr;
    }

    if (buffers.d_batch_phase_corrections) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_batch_phase_corrections));
        buffers.d_batch_phase_corrections = nullptr;
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
    size_t batch_flags_bytes = (buffers.max_timesteps - 1) * sizeof(bool);
    size_t batch_arrays_bytes = buffers.batch_arrays_size * sizeof(float);

    CUDA_CHECK_VOID(cudaMemset(buffers.d_samples, 0, samples_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_amplitudes, 0, tone_params_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_phases, 0, tone_params_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_frequencies, 0, tone_params_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_timesteps, 0, batch_timesteps_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_do_generate, 0, batch_flags_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_frequencies, 0, batch_arrays_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_amplitudes, 0, batch_arrays_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_offset_phases_user, 0, batch_arrays_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_offset_phases, 0, batch_arrays_bytes));
    size_t phase_corrections_bytes = (buffers.max_timesteps - 1) *
                                      buffers.num_channels *
                                      buffers.num_tones * sizeof(float);
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_phase_corrections, 0, phase_corrections_bytes));

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

// Simple double→float conversion kernel (no striding)
__global__ void convertFp64ToFp32Kernel(
    const double* __restrict__ src,
    float* __restrict__ dst,
    int num_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

// CUDA kernel to expand tones from double (compact) to float (padded)
// Performs double→float conversion AND strided copy in single pass
__global__ void expandTonesKernelFp64(
    const double* __restrict__ src,  // Compact: [timestep][channel][num_tones], float64
    float* __restrict__ dst,         // Expanded: [timestep][channel][max_tones], float32
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

        // Copy with conversion
        for (int tone = 0; tone < num_tones; tone++) {
            dst[dst_base + tone] = static_cast<float>(src[src_base + tone]);
        }

        // Zero-pad
        for (int tone = num_tones; tone < max_tones; tone++) {
            dst[dst_base + tone] = 0.0f;
        }
    }
}

// Compute offset phases with high-precision modulo operation
// Formula: (2π * f * t) % (2π) + offset_phase_user
// Uses double precision for phase calculation to avoid errors at high frequencies
__global__ void computeOffsetPhasesKernel(
    const float* __restrict__ d_frequencies,        // [timestep][channel][tone]
    const float* __restrict__ d_offset_phases_user, // [timestep][channel][tone]
    const int32_t* __restrict__ d_timesteps,        // [timestep]
    float* __restrict__ d_offset_phases_out,        // [timestep][channel][tone]
    int num_timesteps,
    int num_channels,
    int max_tones,
    int target_offset,
    double waveform_timestep,                       // WAVEFORM_TIMESTEP (samples)
    double sample_rate) {                           // AWG_SAMPLE_RATE (Hz)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_timesteps * num_channels * max_tones;

    if (idx < total_elements) {
        int t = idx / (num_channels * max_tones);
        int remainder = idx % (num_channels * max_tones);
        int ch = remainder / max_tones;
        int tone = remainder % max_tones;

        // Get timestep value from the array
        int32_t timestep_value = d_timesteps[target_offset + t];

        // Convert to physical time (seconds from batch start)
        // Use double precision for accuracy
        double t_seconds = static_cast<double>(timestep_value) *
                           waveform_timestep / sample_rate;

        // Get frequency (Hz) - convert to double for phase calculation
        int freq_idx = (target_offset + t) * num_channels * max_tones +
                       ch * max_tones + tone;
        double f = static_cast<double>(d_frequencies[freq_idx]);

        // Compute phase modulo term with double precision
        const double TWO_PI = 6.283185307179586;
        double phase_term = 2.0 * M_PI * f * t_seconds;
        double phase_mod = fmod(phase_term, TWO_PI);  // High-precision modulo

        // Get user-provided offset phase
        float user_phase = d_offset_phases_user[freq_idx];

        // Compute final phase (cast to float32 for storage)
        float final_phase = static_cast<float>(phase_mod) + user_phase;

        // Store in output
        d_offset_phases_out[freq_idx] = final_phase;
    }
}

// Compute phase corrections for frequency sweeps between timesteps
// Formula: (π + delta_phi - delta_phi0) % (2π) - π
// where delta_phi = π * (f2 - f1) * (t2 - t1), delta_phi0 is reference (tone 0)
__global__ void computePhaseCorrectionKernel(
    const float* __restrict__ d_frequencies,        // [timestep][channel][tone]
    const int32_t* __restrict__ d_timesteps,        // [timestep]
    float* __restrict__ d_phase_corrections,        // [timestep-1][channel][tone]
    int num_timesteps,
    int num_channels,
    int max_tones,
    int target_offset,
    float waveform_timestep,                        // WAVEFORM_TIMESTEP (samples)
    float sample_rate) {                            // AWG_SAMPLE_RATE (Hz)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_intervals = (num_timesteps - 1) * num_channels;

    if (idx < total_intervals) {
        int interval = idx / num_channels;  // Which timestep pair
        int ch = idx % num_channels;

        // Get consecutive timestep values
        int32_t t1_value = d_timesteps[target_offset + interval];
        int32_t t2_value = d_timesteps[target_offset + interval + 1];

        // Convert to physical time difference (seconds)
        float dt = static_cast<float>(t2_value - t1_value) *
                   waveform_timestep / sample_rate;

        // Base indices for frequencies at t1 and t2
        int freq_base_t1 = (target_offset + interval) * num_channels * max_tones +
                           ch * max_tones;
        int freq_base_t2 = (target_offset + interval + 1) * num_channels * max_tones +
                           ch * max_tones;

        // Get reference (tone 0) delta_phi
        float f1_tone0 = d_frequencies[freq_base_t1];
        float f2_tone0 = d_frequencies[freq_base_t2];
        float delta_phi0 = M_PI * (f2_tone0 - f1_tone0) * dt;

        // Compute correction for all tones
        for (int tone = 0; tone < max_tones; tone++) {
            float f1 = d_frequencies[freq_base_t1 + tone];
            float f2 = d_frequencies[freq_base_t2 + tone];

            float delta_phi = M_PI * (f2 - f1) * dt;

            // Phase correction formula: (π + delta_phi - delta_phi0) % (2π) - π
            float correction = M_PI + delta_phi - delta_phi0;
            correction = fmodf(correction, 2.0f * M_PI);  // Modulo 2π
            correction -= M_PI;

            // Store in output array [interval][channel][tone]
            int out_idx = interval * num_channels * max_tones + ch * max_tones + tone;
            d_phase_corrections[out_idx] = correction;
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
                          const double* h_frequencies,
                          const float* h_amplitudes,
                          const float* h_offset_phases_user,
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
    size_t flags_bytes = (num_timesteps - 1) * sizeof(uint8_t);

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
        // No padding needed - but still need double→float conversion for frequencies
        size_t arrays_elements = num_timesteps * num_channels * num_tones;
        size_t arrays_bytes_fp32 = arrays_elements * sizeof(float);
        size_t arrays_bytes_fp64 = arrays_elements * sizeof(double);
        int offset_elements = target_offset * num_channels * max_tones;

        // Copy double frequencies to temp buffer, then convert to float32
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_temp_frequencies_fp64, h_frequencies,
                                   arrays_bytes_fp64, cudaMemcpyHostToDevice));

        // Launch conversion kernel
        int block_size = 256;
        int grid_size = (arrays_elements + block_size - 1) / block_size;
        convertFp64ToFp32Kernel<<<grid_size, block_size>>>(
            buffers.d_temp_frequencies_fp64,
            buffers.d_batch_frequencies + offset_elements,
            arrays_elements);
        CUDA_CHECK_VOID(cudaDeviceSynchronize());

        // Amplitudes and phases remain float32 (direct copy)
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_amplitudes + offset_elements,
                                   h_amplitudes, arrays_bytes_fp32, cudaMemcpyHostToDevice));
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_offset_phases_user + offset_elements,
                                   h_offset_phases_user, arrays_bytes_fp32, cudaMemcpyHostToDevice));
    } else {
        // GPU-accelerated striding: copy compact data then expand on GPU

        // 1. Copy compact data to temp buffers (smaller transfer)
        size_t compact_elements = num_timesteps * num_channels * num_tones;
        size_t compact_bytes_fp32 = compact_elements * sizeof(float);
        size_t compact_bytes_fp64 = compact_elements * sizeof(double);

        auto t_compact_copy_start = std::chrono::high_resolution_clock::now();
        // Frequencies: copy as double to fp64 temp buffer
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_temp_frequencies_fp64, h_frequencies,
                                   compact_bytes_fp64, cudaMemcpyHostToDevice));
        // Amplitudes and phases: copy as float to temp buffers
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_temp_amplitudes, h_amplitudes,
                                   compact_bytes_fp32, cudaMemcpyHostToDevice));
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_temp_offset_phases_user, h_offset_phases_user,
                                   compact_bytes_fp32, cudaMemcpyHostToDevice));
        auto t_compact_copy_end = std::chrono::high_resolution_clock::now();
        auto compact_copy_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_compact_copy_end - t_compact_copy_start).count();

        // 2. Launch kernels to expand in parallel
        int total_slices = num_timesteps * num_channels;
        int block_size = 256;
        int grid_size = (total_slices + block_size - 1) / block_size;

        auto t_kernel_start = std::chrono::high_resolution_clock::now();
        // Frequencies: use fp64 kernel (converts double→float AND expands)
        expandTonesKernelFp64<<<grid_size, block_size>>>(
            buffers.d_temp_frequencies_fp64, buffers.d_batch_frequencies,
            num_timesteps, num_channels, num_tones, max_tones, target_offset);
        // Amplitudes and phases: use existing float32 kernel
        expandTonesKernel<<<grid_size, block_size>>>(
            buffers.d_temp_amplitudes, buffers.d_batch_amplitudes,
            num_timesteps, num_channels, num_tones, max_tones, target_offset);
        expandTonesKernel<<<grid_size, block_size>>>(
            buffers.d_temp_offset_phases_user, buffers.d_batch_offset_phases_user,
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

    // ===== PHASE COMPUTATION =====
    auto t_phase_start = std::chrono::high_resolution_clock::now();

    // Get configuration constants
    double waveform_timestep_d = static_cast<double>(config::WAVEFORM_TIMESTEP);
    double sample_rate_d = static_cast<double>(config::AWG_SAMPLE_RATE);
    float waveform_timestep_f = static_cast<float>(config::WAVEFORM_TIMESTEP);
    float sample_rate_f = static_cast<float>(config::AWG_SAMPLE_RATE);

    // Kernel 1: Compute offset phases
    int total_phase_elements = num_timesteps * num_channels * max_tones;
    int phase_block_size = 256;
    int phase_grid_size = (total_phase_elements + phase_block_size - 1) / phase_block_size;

    computeOffsetPhasesKernel<<<phase_grid_size, phase_block_size>>>(
        buffers.d_batch_frequencies,
        buffers.d_batch_offset_phases_user,
        buffers.d_batch_timesteps,
        buffers.d_batch_offset_phases,
        num_timesteps,
        num_channels,
        max_tones,
        target_offset,
        waveform_timestep_d,
        sample_rate_d);

    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    auto t_phase_mid = std::chrono::high_resolution_clock::now();

    // Kernel 2: Compute phase corrections
    int total_corrections = (num_timesteps - 1) * num_channels;
    int corr_block_size = 256;
    int corr_grid_size = (total_corrections + corr_block_size - 1) / corr_block_size;

    computePhaseCorrectionKernel<<<corr_grid_size, corr_block_size>>>(
        buffers.d_batch_frequencies,
        buffers.d_batch_timesteps,
        buffers.d_batch_phase_corrections,
        num_timesteps,
        num_channels,
        max_tones,
        target_offset,
        waveform_timestep_f,
        sample_rate_f);

    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    auto t_phase_end = std::chrono::high_resolution_clock::now();
    auto phase_compute_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_phase_mid - t_phase_start).count();
    auto correction_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_phase_end - t_phase_mid).count();
    auto phase_total_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_phase_end - t_phase_start).count();

    auto t_total_end = std::chrono::high_resolution_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t_total_end - t_start).count();

    std::cout << "[GPU] Uploaded " << num_timesteps << " timesteps to GPU at offset "
              << target_offset << " (tones: " << num_tones << " → " << max_tones << ")" << std::endl;
    std::cout << "[GPU] ─── GPU Copy Timing ───" << std::endl;
    std::cout << "[GPU]   Timesteps/flags:   " << simple_us << " μs" << std::endl;
    std::cout << "[GPU]   Tone arrays:       " << tones_us << " μs "
              << (num_tones == max_tones ? "(direct)" : "(strided+padded)") << std::endl;
    std::cout << "[GPU]   Phase computation: " << phase_total_us << " μs" << std::endl;
    std::cout << "[GPU]     ├─ Offset phases:   " << phase_compute_us << " μs" << std::endl;
    std::cout << "[GPU]     └─ Corrections:     " << correction_us << " μs" << std::endl;
    std::cout << "[GPU]   Total GPU upload:  " << total_us << " μs" << std::endl;
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
