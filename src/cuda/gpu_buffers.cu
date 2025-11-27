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
    CUDA_CHECK(cudaMalloc(&buffers.d_batch_amplitudes, batch_arrays_bytes));

    // Allocate coefficient arrays [interval][channel][tone]
    size_t coef_bytes = (buffers.max_timesteps - 1) *
                        buffers.num_channels *
                        buffers.num_tones * sizeof(float);
    CUDA_CHECK(cudaMalloc(&buffers.d_batch_coef0, coef_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_batch_coef1, coef_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_batch_coef2, coef_bytes));

    // Allocate temporary buffers
    CUDA_CHECK(cudaMalloc(&buffers.d_temp_amplitudes, batch_arrays_bytes));
    size_t temp_freq_bytes = buffers.batch_arrays_size * sizeof(double);
    size_t temp_phase_bytes = buffers.batch_arrays_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&buffers.d_temp_frequencies, temp_freq_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.d_temp_offset_phases, temp_phase_bytes));

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
    CUDA_CHECK(cudaMemset(buffers.d_batch_amplitudes, 0, batch_arrays_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_batch_coef0, 0, coef_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_batch_coef1, 0, coef_bytes));
    CUDA_CHECK(cudaMemset(buffers.d_batch_coef2, 0, coef_bytes));
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

    if (buffers.d_batch_amplitudes) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_batch_amplitudes));
        buffers.d_batch_amplitudes = nullptr;
    }

    if (buffers.d_batch_coef0) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_batch_coef0));
        buffers.d_batch_coef0 = nullptr;
    }

    if (buffers.d_batch_coef1) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_batch_coef1));
        buffers.d_batch_coef1 = nullptr;
    }

    if (buffers.d_batch_coef2) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_batch_coef2));
        buffers.d_batch_coef2 = nullptr;
    }

    if (buffers.d_temp_amplitudes) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_temp_amplitudes));
        buffers.d_temp_amplitudes = nullptr;
    }

    if (buffers.d_temp_frequencies) {
        CUDA_CHECK_VOID(cudaFree(buffers.d_temp_frequencies));
        buffers.d_temp_frequencies = nullptr;
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
    size_t batch_flags_bytes = (buffers.max_timesteps - 1) * sizeof(bool);
    size_t batch_arrays_bytes = buffers.batch_arrays_size * sizeof(float);

    size_t coef_bytes = (buffers.max_timesteps - 1) *
                        buffers.num_channels *
                        buffers.num_tones * sizeof(float);

    CUDA_CHECK_VOID(cudaMemset(buffers.d_samples, 0, samples_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_amplitudes, 0, tone_params_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_phases, 0, tone_params_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_frequencies, 0, tone_params_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_timesteps, 0, batch_timesteps_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_do_generate, 0, batch_flags_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_amplitudes, 0, batch_arrays_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_coef0, 0, coef_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_coef1, 0, coef_bytes));
    CUDA_CHECK_VOID(cudaMemset(buffers.d_batch_coef2, 0, coef_bytes));

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

// Kernel to expand tone arrays from num_tones to max_tones with zero-padding
// Used only for amplitudes now (frequencies/phases handled via coefficients)
__global__ void expandTonesKernel(
    const float* __restrict__ src,  // Compact: [timestep][channel][num_tones]
    float* __restrict__ dst,        // Expanded: [timestep][channel][max_tones]
    int num_timesteps,
    int num_channels,
    int num_tones,
    int max_tones,
    int dst_offset) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_slices = num_timesteps * num_channels;

    if (idx < total_slices) {
        int t = idx / num_channels;
        int ch = idx % num_channels;

        int src_base = t * num_channels * num_tones + ch * num_tones;
        int dst_base = (dst_offset + t) * num_channels * max_tones + ch * max_tones;

        // Copy actual tones
        for (int tone = 0; tone < num_tones; tone++) {
            dst[dst_base + tone] = src[src_base + tone];
        }

        // Zero-pad
        for (int tone = num_tones; tone < max_tones; tone++) {
            dst[dst_base + tone] = 0.0f;
        }
    }
}

// Compute polynomial coefficients for waveform generation
// Coefficients encode frequency chirps and phase evolution for efficient interpolation
__global__ void computeCoefficientsKernel(
    const double* __restrict__ d_frequencies,       // [timestep][channel][tone] COMPACT
    const float* __restrict__ d_offset_phases,      // [timestep][channel][tone] COMPACT
    const int32_t* __restrict__ d_timesteps,        // [timestep]
    float* __restrict__ d_coef0,                    // [interval][channel][tone]
    float* __restrict__ d_coef1,                    // [interval][channel][tone]
    float* __restrict__ d_coef2,                    // [interval][channel][tone]
    int num_intervals,                              // num_timesteps - 1
    int num_channels,
    int max_tones,                                  // Output stride
    int num_tones_actual,                           // Input stride (compact)
    int target_offset,
    double waveform_timestep,
    double sample_rate) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = num_intervals * num_channels;

    if (idx < total_work) {
        int interval = idx / num_channels;
        int ch = idx % num_channels;

        // Get timestep values
        int32_t t1_val = d_timesteps[target_offset + interval];
        int32_t t2_val = d_timesteps[target_offset + interval + 1];

        // Convert to physical times (double precision)
        double t1 = static_cast<double>(t1_val) * waveform_timestep / sample_rate;
        double t2 = static_cast<double>(t2_val) * waveform_timestep / sample_rate;
        double dt = t2 - t1;
        double t_sum = t1 + t2;

        // Base indices in GPU arrays [timestep][channel][tone]
        int base_t1 = (target_offset + interval) * num_channels * max_tones + ch * max_tones;
        int base_t2 = (target_offset + interval + 1) * num_channels * max_tones + ch * max_tones;

        // Get reference (tone 0) values - double precision
        double f1_ref = d_frequencies[base_t1];
        double f2_ref = d_frequencies[base_t2];
        double phi1_ref = static_cast<double>(d_offset_phases[base_t1]);
        double phi2_ref = static_cast<double>(d_offset_phases[base_t2]);

        // Compute reference psi(0)
        const double TWO_PI = 6.283185307179586;
        double psi_ref = M_PI * (f2_ref - f1_ref) * t_sum + phi2_ref - phi1_ref;

        // Compute coefficients for all tones
        for (int tone = 0; tone < max_tones; tone++) {
            double f1 = d_frequencies[base_t1 + tone];
            double f2 = d_frequencies[base_t2 + tone];
            double phi1 = static_cast<double>(d_offset_phases[base_t1 + tone]);
            double phi2 = static_cast<double>(d_offset_phases[base_t2 + tone]);

            // Compute psi and delta_psi
            double psi = M_PI * (f2 - f1) * t_sum + phi2 - phi1;
            double delta_psi = psi - psi_ref;

            // Coef0: (2π*f1*t1 + phi1) % (2π)
            double coef0_d = 2.0 * M_PI * f1 * t1 + phi1;
            coef0_d = fmod(coef0_d, TWO_PI);

            // Coef1: (delta_psi+π)%(2π) - π + 2π*f1*dt
            double term1 = fmod(delta_psi + M_PI, TWO_PI) - M_PI;
            double coef1_d = term1 + 2.0 * M_PI * f1 * dt;

            // Coef2: π*(f2-f1)*dt
            double coef2_d = M_PI * (f2 - f1) * dt;

            // Cast to float32 and store [interval][channel][tone]
            int out_idx = interval * num_channels * max_tones + ch * max_tones + tone;
            d_coef0[out_idx] = static_cast<float>(coef0_d);
            d_coef1[out_idx] = static_cast<float>(coef1_d);
            d_coef2[out_idx] = static_cast<float>(coef2_d);
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

    // Step 1: Upload timesteps and do_generate
    size_t timesteps_bytes = num_timesteps * sizeof(int32_t);
    size_t flags_bytes = (num_timesteps - 1) * sizeof(uint8_t);

    auto t_simple_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_timesteps + target_offset,
                               h_timesteps, timesteps_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_do_generate + target_offset,
                               h_do_generate, flags_bytes, cudaMemcpyHostToDevice));
    auto t_simple_end = std::chrono::high_resolution_clock::now();
    auto simple_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_simple_end - t_simple_start).count();

    // Step 2: Upload amplitudes (with striding if needed)
    auto t_amp_start = std::chrono::high_resolution_clock::now();
    int offset_elements = target_offset * num_channels * max_tones;

    if (num_tones == max_tones) {
        // Direct copy
        size_t arrays_bytes = num_timesteps * num_channels * num_tones * sizeof(float);
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_batch_amplitudes + offset_elements,
                                   h_amplitudes, arrays_bytes, cudaMemcpyHostToDevice));
    } else {
        // Strided copy via temp buffer + expansion kernel
        size_t compact_bytes = num_timesteps * num_channels * num_tones * sizeof(float);
        CUDA_CHECK_VOID(cudaMemcpy(buffers.d_temp_amplitudes, h_amplitudes,
                                   compact_bytes, cudaMemcpyHostToDevice));

        int total_slices = num_timesteps * num_channels;
        int block_size = 256;
        int grid_size = (total_slices + block_size - 1) / block_size;
        expandTonesKernel<<<grid_size, block_size>>>(
            buffers.d_temp_amplitudes, buffers.d_batch_amplitudes,
            num_timesteps, num_channels, num_tones, max_tones, target_offset);
        CUDA_CHECK_VOID(cudaDeviceSynchronize());
    }

    auto t_amp_end = std::chrono::high_resolution_clock::now();
    auto amp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_amp_end - t_amp_start).count();

    // Step 3: Upload freq/phase to temp buffers (compact layout)
    auto t_temp_start = std::chrono::high_resolution_clock::now();

    // Upload in compact form - coefficient kernel handles padding internally
    size_t compact_elements = num_timesteps * num_channels * num_tones;
    size_t compact_freq_bytes = compact_elements * sizeof(double);
    size_t compact_phase_bytes = compact_elements * sizeof(float);

    CUDA_CHECK_VOID(cudaMemcpy(buffers.d_temp_frequencies, h_frequencies,
                               compact_freq_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(buffers.d_temp_offset_phases, h_offset_phases_user,
                               compact_phase_bytes, cudaMemcpyHostToDevice));

    auto t_temp_end = std::chrono::high_resolution_clock::now();
    auto temp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_temp_end - t_temp_start).count();

    // Step 4: Compute polynomial coefficients
    auto t_coef_start = std::chrono::high_resolution_clock::now();

    double waveform_timestep_d = static_cast<double>(config::WAVEFORM_TIMESTEP);
    double sample_rate_d = static_cast<double>(config::AWG_SAMPLE_RATE);

    int num_intervals = num_timesteps - 1;
    int total_work = num_intervals * num_channels;
    int block_size = 256;
    int grid_size = (total_work + block_size - 1) / block_size;

    computeCoefficientsKernel<<<grid_size, block_size>>>(
        buffers.d_temp_frequencies,
        buffers.d_temp_offset_phases,
        buffers.d_batch_timesteps,
        buffers.d_batch_coef0,
        buffers.d_batch_coef1,
        buffers.d_batch_coef2,
        num_intervals,
        num_channels,
        max_tones,
        num_tones,  // Pass actual num_tones for compact indexing
        target_offset,
        waveform_timestep_d,
        sample_rate_d);

    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    auto t_coef_end = std::chrono::high_resolution_clock::now();
    auto coef_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_coef_end - t_coef_start).count();

    auto t_total_end = std::chrono::high_resolution_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_total_end - t_start).count();

    std::cout << "[GPU] Uploaded " << num_timesteps << " timesteps to GPU at offset "
              << target_offset << " (tones: " << num_tones << " → " << max_tones << ")" << std::endl;
    std::cout << "[GPU] ─── GPU Upload Timing ───" << std::endl;
    std::cout << "[GPU]   Timesteps/flags:     " << simple_us << " μs" << std::endl;
    std::cout << "[GPU]   Amplitudes:          " << amp_us << " μs" << std::endl;
    std::cout << "[GPU]   Temp freq/phase:     " << temp_us << " μs" << std::endl;
    std::cout << "[GPU]   Coefficient compute: " << coef_us << " μs" << std::endl;
    std::cout << "[GPU]   Total GPU upload:    " << total_us << " μs" << std::endl;
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
