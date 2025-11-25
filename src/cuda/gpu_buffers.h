#ifndef AOD_GPU_BUFFERS_H
#define AOD_GPU_BUFFERS_H

#include <cstddef>
#include <cstdint>

namespace aod {

// GPU buffer management for waveform generation
// Memory layout: [chunk][channel][tone] for tone parameters
//                [chunk][channel][timestep] for output samples
struct GPUBuffers {
    // Device memory (GPU)
    int16_t* d_samples;           // Output waveform samples
    float* d_amplitudes;          // Tone amplitudes [chunk][channel][tone]
    float* d_phases;              // Tone phases [chunk][channel][tone]
    float* d_frequencies;         // Tone frequencies [chunk][channel][tone]

    // Batch data arrays (global timeline combining all batches)
    int32_t* d_batch_timesteps;   // Timesteps array [timestep]
    bool* d_batch_do_generate;    // Generation flags [timestep-1] (interval control)
    float* d_batch_frequencies;   // Frequencies [timestep][channel][tone]
    float* d_batch_amplitudes;    // Amplitudes [timestep][channel][tone]
    float* d_batch_offset_phases_user; // User-provided offset phases [timestep][channel][tone]

    // Temporary buffers for compact data (strided copy optimization)
    // Used when client sends num_tones < AOD_MAX_TONES
    float* d_temp_frequencies;    // Same size as batch arrays
    float* d_temp_amplitudes;
    float* d_temp_offset_phases_user;

    // Temporary buffer for float64 frequency data before conversion
    double* d_temp_frequencies_fp64;  // For receiving float64 before conversion

    // Computed phase arrays (generated from user phases + frequencies)
    float* d_batch_offset_phases;       // Computed offset phases [timestep][channel][tone]
    float* d_batch_phase_corrections;   // Phase corrections [timestep-1][channel][tone]

    // Pinned host memory (CPU, page-locked for fast transfer)
    int16_t* h_samples_pinned;    // For DMA transfers to AWG

    // Buffer dimensions
    size_t num_chunks;            // Number of waveform chunks
    size_t num_channels;          // Number of active AWG channels
    size_t num_tones;             // Maximum tones per channel
    size_t timestep;              // Samples per chunk
    size_t max_timesteps;         // Maximum timesteps in batch arrays

    // Computed sizes
    size_t total_samples;         // Total int16 samples
    size_t tone_params_size;      // Size of each tone parameter array (amp/phase/freq)
    size_t batch_arrays_size;     // Size of batch frequency/amplitude/phase arrays

    // Constructor
    GPUBuffers()
        : d_samples(nullptr),
          d_amplitudes(nullptr),
          d_phases(nullptr),
          d_frequencies(nullptr),
          d_batch_timesteps(nullptr),
          d_batch_do_generate(nullptr),
          d_batch_frequencies(nullptr),
          d_batch_amplitudes(nullptr),
          d_batch_offset_phases_user(nullptr),
          d_temp_frequencies(nullptr),
          d_temp_amplitudes(nullptr),
          d_temp_offset_phases_user(nullptr),
          d_temp_frequencies_fp64(nullptr),
          d_batch_offset_phases(nullptr),
          d_batch_phase_corrections(nullptr),
          h_samples_pinned(nullptr),
          num_chunks(0),
          num_channels(0),
          num_tones(0),
          timestep(0),
          max_timesteps(0),
          total_samples(0),
          tone_params_size(0),
          batch_arrays_size(0) {}
};

// Allocate all GPU buffers based on compile-time configuration
// Returns true on success, false on allocation failure
bool allocateGPUBuffers(GPUBuffers& buffers);

// Free all allocated GPU buffers
void freeGPUBuffers(GPUBuffers& buffers);

// Zero all GPU and pinned host buffers
void zeroGPUBuffers(GPUBuffers& buffers);

// Upload batch data from CPU to GPU with strided copy for tone padding
// Copies data from client arrays (with num_tones) to GPU arrays (with AOD_MAX_TONES)
// Appends at target_offset in GPU arrays, zero-fills unused tone slots
void uploadBatchDataToGPU(GPUBuffers& buffers,
                          const int32_t* h_timesteps,
                          const uint8_t* h_do_generate,
                          const double* h_frequencies,
                          const float* h_amplitudes,
                          const float* h_offset_phases_user,
                          int num_timesteps,
                          int num_channels,
                          int num_tones,
                          int target_offset);

// Validate that timesteps are strictly ascending (each < next)
// Returns true if valid, false otherwise
bool validateTimestepsAscending(const int32_t* d_timesteps, int num_timesteps);

} // namespace aod

#endif // AOD_GPU_BUFFERS_H
