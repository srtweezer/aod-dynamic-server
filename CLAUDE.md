# AOD Dynamic Server

## Project Overview

This project implements a high-performance real-time server for generating multi-tone acoustic waveforms to control acousto-optic deflectors (AODs) used in optical tweezer systems for quantum computing and atomic physics experiments.

### Purpose

The server enables precise control of individual atoms trapped in optical tweezers by generating complex waveforms that drive AODs. These waveforms allow for dynamic manipulation of atom positions, including transport, pickup, and dropoff operations.

### Key Requirements

1. **Real-time Performance**: Generate waveforms with minimal latency to stream continuously to AWG hardware
2. **GPU Acceleration**: Leverage CUDA for highly optimized parallel waveform generation
3. **Protobuf API**: Provide a clean interface for users to specify multi-tone action sequences
4. **AWG Integration**: Stream data to Spectrum Instrumentation AWG operating in FIFO mode
5. **Multi-tone Synthesis**: Generate complex waveforms with multiple frequency components for precise spatial control

## Technical Architecture

### Core Components

- **C++ Server**: Main application handling client requests and orchestrating waveform generation
- **CUDA Kernels**: GPU-accelerated waveform synthesis for real-time performance
- **Protobuf Interface**: API definition for client communication
- **AWG Driver**: Interface to Spectrum Instrumentation hardware (FIFO mode)
- **Configuration System**: Parameters defining hardware specifications and constraints

### Hardware Context

- **AWG**: Spectrum Instrumentation arbitrary waveform generators
  - Multi-channel output capability
  - FIFO mode for continuous streaming
  - High sample rates (100+ MSample/s typical)

- **AODs**: Acousto-optic deflectors with RF frequency control
  - Frequency range determines spatial positioning
  - Hardware parameters will be defined via configuration system

### Physics Context

The system controls optical tweezers created by acousto-optic deflectors. Multi-tone RF signals driving the AODs create multiple diffraction orders, each producing an optical trap. By precisely controlling:
- Frequencies: spatial position of tweezers
- Amplitudes: trap intensity
- Phases: reduces intermodulation distortion

The system can dynamically transport atoms between positions, crucial for quantum computing operations.

## Analysis of Existing Code

### `load_waveform.py`

Python implementation for AWG control using the `spcm` library:

- **AWG initialization**: Configures channels, clock, triggers, and digital outputs
- **Two modes**:
  - `load_awg()`: Single waveform playback with loop control
  - `load_awg_sweep()`: Multi-segment waveform sequences
- **Features**:
  - External/internal clock support
  - Amplitude normalization and scaling
  - Digital channel output via bit 15
  - 16-bit sample resolution
  - Enforces 32-sample alignment requirement

**Key insights**: The AWG requires careful buffer management, proper amplitude scaling to stay within limits, and 32-sample alignment for DMA transfers.

### `tweezer_generation.py`

Python/JAX implementation of waveform generation logic:

- **Chunk-based architecture**: Sequences composed of modular chunks
  - `StaticTweezers`: Fixed position tweezers
  - `Translation`: Moving tweezers with various trajectories (constant jerk, adiabatic sine, linear)
  - `PickupOrDropoff`: Amplitude ramping for atom loading/unloading
  - `VariableAmplitudeTranslation`: Combined motion and amplitude control

- **Phase tracking**: Maintains phase continuity across chunks to prevent discontinuities
- **Multi-tone synthesis**: Sums multiple frequency components for each AOD axis (X/Y)
- **Optimal intermodulation**: Implements phase optimization algorithm to minimize peak-to-RMS ratio

**Key insights**: The waveform generation is mathematically complex with careful phase management. The chunk-based design is clean and extensible. JAX provides array operations but we'll need equivalent GPU kernels.

### `test_gpu.cu`

CUDA performance testing of waveform generation:

- **Kernel**: `generate_sum_waveform()`
  - Uses shared memory for efficient summation
  - Parallel reduction pattern
  - One block per time sample, threads per atom/tone
  - Cosine generation using `cospif()`

- **Performance**: Tests with ~64 tones, ~524k samples, measuring computation latency
- **Memory management**: Pre-allocates buffers, uses device-to-device copies

**Key insights**: The GPU approach is promising for real-time generation. The shared memory reduction is efficient for summing many tones. Need to optimize for streaming/pipelined operation rather than batch processing.

### `Makefile`

Simple build setup for CUDA compilation targeting sm_86 (Ada Lovelace architecture).

## High-Level Course of Action

### Phase 1: Project Setup & Architecture (Current Phase)
- [x] Document project requirements and existing work
- [ ] Define project structure (directories, build system)
- [ ] Set up CMake build system for C++/CUDA hybrid project
- [ ] Define core abstractions and class hierarchy
- [ ] Design Protobuf API schema for client-server communication
- [ ] Design configuration system for hardware parameters

### Phase 2: Core Infrastructure
- [ ] Implement basic C++ server framework
  - Socket/network communication layer
  - Protobuf message handling
  - Request queuing and scheduling

- [ ] Create waveform generation abstraction layer
  - Chunk interface design
  - Sequence management
  - Buffer management for streaming

- [ ] Implement configuration loading and validation

### Phase 3: GPU Acceleration
- [ ] Port waveform generation algorithms to CUDA
  - Static tone generation
  - Frequency chirps for translation
  - Phase tracking across chunks

- [ ] Optimize GPU kernels for real-time performance
  - Stream-based processing
  - Overlapped compute and transfer
  - Double-buffering for continuous output

- [ ] Implement GPU memory management
  - Pinned memory for fast transfers
  - Buffer pools to avoid allocation overhead

### Phase 4: AWG Integration
- [ ] Create C++ wrapper for Spectrum Instrumentation API
- [ ] Implement FIFO mode streaming
- [ ] Handle buffer underrun detection and recovery
- [ ] Add synchronization and triggering support

### Phase 5: Testing & Optimization
- [ ] Unit tests for waveform generation accuracy
- [ ] Integration tests with AWG hardware
- [ ] Latency benchmarking and profiling
- [ ] Real-world physics validation
- [ ] Performance tuning for sustained throughput

### Phase 6: API & Usability
- [ ] Complete Protobuf API implementation
- [ ] Python client library
- [ ] Documentation and examples
- [ ] Error handling and diagnostics

## Design Decisions to Consider

1. **Threading model**: Single-threaded with async I/O vs. multi-threaded request handling
2. **Memory layout**: AoS vs. SoA for tone parameters, impact on coalesced memory access
3. **Precision**: float vs. double for phase accumulation (drift over long sequences)
4. **Buffering strategy**: How many buffers, what size, precomputation pipeline depth
5. **API granularity**: Low-level per-chunk control vs. high-level "move atom A to B" commands
6. **Error handling**: How to handle underruns, GPU errors, invalid sequences
7. **Configurability**: Runtime vs. compile-time parameters for optimization

## Key Challenges

1. **Latency**: Must generate waveforms faster than real-time to prevent AWG buffer underruns
2. **Phase continuity**: Smooth transitions between chunks to avoid spurious atom heating
3. **Amplitude optimization**: Stay within AWG limits while maximizing signal quality
4. **Scalability**: Handle varying numbers of tweezers (1-100+) efficiently
5. **Reliability**: Robust error handling for 24/7 operation in experiments

## Resources & Dependencies

- CUDA Toolkit (11.0+)
- CMake (3.20+)
- Protobuf (3.0+)
- Spectrum Instrumentation drivers and SDK
- C++17 or later
- (Optional) Python bindings: pybind11

## Notes

- AWG sample alignment requirements must be enforced throughout (hardware-specific)
- Amplitude constraints must be validated before streaming (configurable limits)
- External clock support needed for synchronization with other experiment hardware
- Digital output channels can be used for triggering downstream equipment
