# AOD Dynamic Server API Documentation

This document describes the Protocol Buffer API for communicating with the AOD Dynamic Server.

## Communication Protocol

- **Transport**: ZeroMQ REQ-REP pattern
- **Serialization**: Protocol Buffers (proto3)
- **Default Port**: 5555 (configurable in `config.yml`)

## Message Structure

All communication uses two top-level messages:

### Request (Client → Server)

```protobuf
message Request {
  oneof command {
    PingRequest ping = 1;
    // Future commands...
  }
}
```

### Response (Server → Client)

```protobuf
message Response {
  oneof result {
    PingResponse ping = 1;
    // Future responses...
  }
}
```

## Available Commands

### Ping

Health check command to verify server connectivity and measure round-trip time.

#### Request

```protobuf
message PingRequest {
  // Empty - no parameters required
}
```

#### Response

```protobuf
message PingResponse {
  int64 timestamp_ns = 1;  // Server timestamp in nanoseconds since epoch
}
```

#### Example Usage (Python)

```python
import zmq
import aod_server_pb2

# Connect to server
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Create ping request
request = aod_server_pb2.Request()
request.ping.CopyFrom(aod_server_pb2.PingRequest())

# Send request
socket.send(request.SerializeToString())

# Receive response
response_data = socket.recv()
response = aod_server_pb2.Response()
response.ParseFromString(response_data)

# Extract timestamp
timestamp_ns = response.ping.timestamp_ns
print(f"Server timestamp: {timestamp_ns} ns")
```

## Future Commands (Planned)

The following commands are planned for future implementation:

### Generate Waveform

Generate and stream waveforms to the AWG.

```protobuf
message GenerateWaveformRequest {
  // To be defined
  // - Sequence of tweezer operations
  // - Timing parameters
  // - Frequency/amplitude specifications
}

message GenerateWaveformResponse {
  bool success = 1;
  string error_message = 2;
}
```

### Stop/Abort

Stop ongoing waveform generation.

```protobuf
message StopRequest {
  // Empty or abort reason
}

message StopResponse {
  bool success = 1;
}
```

### Get Status

Query server and AWG status.

```protobuf
message GetStatusRequest {
  // Empty
}

message GetStatusResponse {
  bool awg_connected = 1;
  bool gpu_available = 2;
  int32 buffer_fill_percent = 3;
  // More status fields...
}
```

## Error Handling

Currently, errors are logged to the server console. Future versions will include:

```protobuf
message ErrorResponse {
  int32 error_code = 1;
  string error_message = 2;
  string stack_trace = 3;  // Optional, for debugging
}
```

Common error codes (planned):
- `1`: Invalid request format
- `2`: AWG not connected
- `3`: GPU error
- `4`: Configuration error
- `5`: Buffer underrun

## Client Libraries

### Python

Generate Python bindings:

```bash
protoc --python_out=. proto/aod_server.proto
```

Install dependencies:

```bash
pip install pyzmq protobuf
```

### C++

The generated C++ code is automatically compiled as part of the server build. To use it in a separate C++ client:

1. Link against `aod_proto` library
2. Include `aod_server.pb.h`
3. Use `zmq::socket_t` with `zmq::socket_type::req`

### Other Languages

Protocol Buffers and ZeroMQ are available for many languages. Generate bindings using:

```bash
protoc --[language]_out=. proto/aod_server.proto
```

Supported languages include: Java, Go, Rust, JavaScript, Ruby, PHP, C#, etc.

## Connection Examples

### Python Client Template

```python
import zmq
import aod_server_pb2

class AODClient:
    def __init__(self, host="localhost", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")

    def ping(self):
        request = aod_server_pb2.Request()
        request.ping.CopyFrom(aod_server_pb2.PingRequest())
        self.socket.send(request.SerializeToString())

        response_data = self.socket.recv()
        response = aod_server_pb2.Response()
        response.ParseFromString(response_data)

        return response.ping.timestamp_ns

    def close(self):
        self.socket.close()
        self.context.term()

# Usage
client = AODClient()
timestamp = client.ping()
print(f"Server responded with timestamp: {timestamp}")
client.close()
```

## Network Configuration

The server bind address is configured in `config.yml`:

```yaml
server:
  port: 5555
  bind_address: "tcp://*"  # Listen on all interfaces
```

Client connection strings:
- Local: `tcp://localhost:5555`
- Remote: `tcp://192.168.1.100:5555`
- Unix socket (future): `ipc:///tmp/aod-server.sock`

## Performance Considerations

- **Latency**: REQ-REP pattern is synchronous. Typical round-trip time < 1 ms on localhost.
- **Throughput**: For high-frequency updates, consider batching commands (future enhancement).
- **Timeouts**: Set appropriate ZMQ timeouts to handle server unavailability.

### ZMQ Timeout Example (Python)

```python
socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
try:
    response_data = socket.recv()
except zmq.Again:
    print("Server did not respond in time")
```

## Versioning

API version: **0.1.0** (initial development)

Breaking changes will increment the major version. The server will eventually include version negotiation in the protocol.

---

**Note**: This API is under active development. Commands and message formats may change before version 1.0.0.
