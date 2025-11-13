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
- **CUDA Toolkit** >= 11.0
- **Protocol Buffers** >= 3.0
- **ZeroMQ** (libzmq)
- **yaml-cpp**
- **Spectrum Instrumentation SDK** (optional, for AWG integration)

### Installing Dependencies (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    cmake \
    g++ \
    libzmq3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libyaml-cpp-dev
```

For CUDA, follow the [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

## Building

### 1. Clone the repository

```bash
cd /path/to/aod-dynamic-server
```

### 2. Configure the server

Copy the configuration template and customize it:

```bash
cp config.yml.template config.yml
```

Edit `config.yml` with your hardware parameters (port, AWG settings, etc.).

### 3. Build the project

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

The executable `aod-server` will be created in the `build/` directory.

## Running the Server

### Start the server

```bash
./build/aod-server
```

The server will:
1. Load configuration from `config.yml`
2. Validate parameters
3. Initialize AWG interface (placeholder)
4. Start ZMQ server on configured port (default: 5555)

### Command-line options

```bash
./build/aod-server --help
```

Options:
- `--config PATH`: Specify configuration file (default: `config.yml`)
- `--help`: Show help message

### Stop the server

Press `Ctrl+C` to gracefully shut down the server.

## Configuration

See `config.yml.template` for all available configuration options:

- **server**: Port and bind address
- **awg**: AWG device parameters
- **logging**: Log level and output
- **gpu**: CUDA device selection

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
├── config.yml.template         # Configuration template
├── proto/                      # Protocol Buffer definitions
│   ├── aod_server.proto
│   └── CMakeLists.txt
├── src/                        # Source code
│   ├── main.cpp                # Entry point
│   ├── server.{h,cpp}          # ZMQ server
│   ├── config.{h,cpp}          # Configuration loader
│   ├── awg_interface.{h,cpp}   # AWG interface
│   └── cuda/                   # CUDA kernels (future)
├── old/                        # Reference implementations
├── docs/                       # Documentation
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

Make sure you've created `config.yml` from the template:
```bash
cp config.yml.template config.yml
```

### ZMQ binding errors

- Check if the port is already in use: `netstat -tulpn | grep 5555`
- Try changing the port in `config.yml`

### CUDA errors

- Verify CUDA installation: `nvcc --version`
- Check GPU availability: `nvidia-smi`

## License

[To be determined]

## Contributing

[To be determined]

## Contact

[To be determined]
