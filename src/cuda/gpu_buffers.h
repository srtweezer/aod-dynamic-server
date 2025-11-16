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

    // Pinned host memory (CPU, page-locked for fast transfer)
    int16_t* h_samples_pinned;    // For DMA transfers to AWG

    // Buffer dimensions
    size_t num_chunks;            // Number of waveform chunks
    size_t num_channels;          // Number of active AWG channels
    size_t num_tones;             // Maximum tones per channel
    size_t timestep;              // Samples per chunk

    // Computed sizes
    size_t total_samples;         // Total int16 samples
    size_t tone_params_size;      // Size of each tone parameter array (amp/phase/freq)

    // Constructor
    GPUBuffers()
        : d_samples(nullptr),
          d_amplitudes(nullptr),
          d_phases(nullptr),
          d_frequencies(nullptr),
          h_samples_pinned(nullptr),
          num_chunks(0),
          num_channels(0),
          num_tones(0),
          timestep(0),
          total_samples(0),
          tone_params_size(0) {}
};

// Allocate all GPU buffers based on compile-time configuration
// Returns true on success, false on allocation failure
bool allocateGPUBuffers(GPUBuffers& buffers);

// Free all allocated GPU buffers
void freeGPUBuffers(GPUBuffers& buffers);

// Zero all GPU and pinned host buffers
void zeroGPUBuffers(GPUBuffers& buffers);

} // namespace aod

#endif // AOD_GPU_BUFFERS_H
