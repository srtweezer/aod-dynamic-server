# AOD Dynamic Server

A high-performance real-time server for generating multi-tone acoustic waveforms to control acousto-optic deflectors (AODs) in optical tweezer systems.

## Overview

This server enables precise control of individual atoms trapped in optical tweezers by generating complex waveforms that drive AODs. It uses GPU acceleration (CUDA) for real-time waveform generation and communicates with clients via ZeroMQ with Protocol Buffers.

## Features

- **Real-time waveform generation** with GPU acceleration
- **ZeroMQ REQ-REP** communication pattern
- **Protocol Buffers** for structured API
- **YAML configuration** for hardware parameters
- **Modular design** for easy extension

## Dependencies

- **CMake** >= 3.20
- **C++17** compiler (g++, clang++)
- **CUDA Toolkit** >= 11.0 (optional, for GPU acceleration)
- **Protocol Buffers** >= 3.0
- **ZeroMQ** (libzmq)
- **Spectrum Instrumentation SDK** (required, for AWG hardware)

### Installing Dependencies (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    cmake \
    g++ \
    libzmq3-dev \
    libprotobuf-dev \
    protobuf-compiler
```

For CUDA (optional), follow the [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

For Spectrum Instrumentation SDK, install from [Spectrum's website](https://www.spectrum-instrumentation.com/en/downloads).

## Building

### 1. Clone the repository

```bash
cd /path/to/aod-dynamic-server
```

### 2. Configure the server

Copy the configuration template and customize it:

```bash
cp config.cmake.template config.cmake
```

Edit `config.cmake` with your hardware parameters:
- `AWG_SERIAL_NUMBER`: Your AWG card serial number (or 0 for auto-detect)
- `SERVER_PORT`: ZMQ server port
- `AWG_SAMPLE_RATE`, `AWG_NUM_CHANNELS`, etc.

**IMPORTANT**: All parameters in `config.cmake` are **compile-time constants**. Changes require rebuilding the project.

### 3. Build the project

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

The executable `aod-server` will be created in the `build/` directory.

### 4. Rebuilding after configuration changes

If you modify `config.cmake`:

```bash
cd build
rm -rf *
cmake ..
make -j$(nproc)
```

## Running the Server

### Start the server

```bash
./build/aod-server
```

The server will:
1. Display compiled configuration
2. Scan for Spectrum AWG devices (/dev/spcm0-15)
3. Connect to AWG matching the configured serial number
4. Start ZMQ server on configured port

### Command-line options

```bash
./build/aod-server --help
```

Options:
- `--help`: Show help message

**Note**: Configuration is compiled in from `config.cmake`. There are no runtime configuration files.

### Stop the server

Press `Ctrl+C` to gracefully shut down the server.

## Configuration

All configuration is done via `config.cmake` at **compile-time**. This approach:
- ✅ Allows array sizes and constants to be known at compile time
- ✅ Enables compiler optimizations
- ✅ Perfect for GPU kernel parameters and buffer sizes
- ✅ No runtime config file parsing overhead

See `config.cmake.template` for all available options:

- **Server**: Port, bind address
- **AWG**: Serial number, channels, sample rate, amplitude limits
- **Waveform**: Max tones, buffer sizes
- **GPU**: Device selection, kernel parameters
- **Logging**: Level and output file

To change any setting, edit `config.cmake` and rebuild.

## Testing the Server

### Using Python (with zmq and protobuf)

```python
import zmq
import aod_server_pb2  # Generated from proto file

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Send ping request
request = aod_server_pb2.Request()
request.ping.CopyFrom(aod_server_pb2.PingRequest())

socket.send(request.SerializeToString())

# Receive response
response_data = socket.recv()
response = aod_server_pb2.Response()
response.ParseFromString(response_data)

print(f"Server timestamp: {response.ping.timestamp_ns}")
```

## Project Structure

```
aod-dynamic-server/
├── CMakeLists.txt              # Root build configuration
├── config.cmake.template       # Configuration template (committed)
├── config.cmake                # Your configuration (not committed)
├── proto/                      # Protocol Buffer definitions
│   ├── aod_server.proto
│   └── CMakeLists.txt
├── src/                        # Source code
│   ├── config.h.in             # Configuration header template
│   ├── main.cpp                # Entry point
│   ├── server.{h,cpp}          # ZMQ server
│   ├── awg_interface.{h,cpp}   # AWG interface
│   └── cuda/                   # CUDA kernels (future)
├── external/                   # External dependencies
│   └── spectrum/               # Spectrum SDK headers
├── build/                      # Build directory (not committed)
│   └── config.h                # Generated configuration
├── CLAUDE.md                   # Development notes
└── API.md                      # API documentation
```

## API Documentation

See [API.md](API.md) for complete API documentation.

## Development

### Adding new commands

1. Add message definitions to `proto/aod_server.proto`
2. Add handler method in `src/server.cpp`
3. Update API documentation in `API.md`

### CUDA kernels

Place CUDA kernel files in `src/cuda/` and update `src/CMakeLists.txt`.

## Troubleshooting

### "Configuration file not found"

Make sure you've created `config.cmake` from the template:
```bash
cp config.cmake.template config.cmake
```

### AWG Device Not Found

The server scans `/dev/spcm0` through `/dev/spcm15` for Spectrum cards.

- Check device permissions: `ls -l /dev/spcm*`
- Verify card is detected: `lspci | grep Spectrum`
- Check serial number in `config.cmake` matches your hardware
- Set `AWG_SERIAL_NUMBER = 0` to connect to first available generator card

### ZMQ binding errors

- Check if the port is already in use: `netstat -tulpn | grep 5555`
- Change `SERVER_PORT` in `config.cmake` and rebuild

### CUDA errors

- Verify CUDA installation: `nvcc --version`
- Check GPU availability: `nvidia-smi`

## License

[To be determined]

## Contributing

[To be determined]

## Contact

[To be determined]
