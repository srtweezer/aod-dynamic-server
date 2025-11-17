# AOD Dynamic Server API Documentation

This document describes the Protocol Buffer API for communicating with the AOD Dynamic Server.

## Communication Protocol

- **Transport**: ZeroMQ REQ-REP pattern
- **Serialization**: Protocol Buffers (proto3)
- **Default Port**: Configured at compile-time in `config.cmake` (default: 5555)
- **Response Time**: Commands are synchronous with 1-second timeout

## Message Structure

All communication uses two top-level messages:

### Request (Client → Server)

```protobuf
message Request {
  oneof command {
    PingRequest ping = 1;
    InitializeRequest initialize = 2;
    StopRequest stop = 3;
    WaveformBatchRequest waveform_batch = 4;
  }
}
```

### Response (Server → Client)

```protobuf
message Response {
  oneof result {
    PingResponse ping = 1;
    InitializeResponse initialize = 2;
    StopResponse stop = 3;
    WaveformBatchResponse waveform_batch = 4;
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

---

### Initialize

Configure AWG hardware with channel amplitudes and prepare for waveform generation.

#### Request

```protobuf
message InitializeRequest {
  repeated int32 channel_amplitudes_mv = 1;  // Amplitude in mV for each active channel
}
```

**Parameters:**
- `channel_amplitudes_mv`: Array of amplitudes in millivolts
  - Size must match the number of active channels in `AWG_CHANNEL_MASK` (compile-time config)
  - Example: If `AWG_CHANNEL_MASK = 0b0011` (2 channels), provide 2 amplitudes
  - Values are applied in channel order (e.g., [amp_ch0, amp_ch1])

#### Response

```protobuf
message InitializeResponse {
  bool success = 1;
  string error_message = 2;  // Empty if success=true, detailed error otherwise
}
```

#### What It Does

1. Sets AWG mode to FIFO single replay (`SPC_REP_FIFO_SINGLE`)
2. Enables channels based on compile-time `AWG_CHANNEL_MASK`
3. Configures each active channel:
   - Sets amplitude (in mV)
   - Sets filter to 0 (no filter)
   - Enables output
4. Configures clock (internal PLL) and sample rate
5. Sets software trigger
6. Allocates and zeros 64 MB DMA buffer
7. Defines DMA transfer with notify size

**State Transition:** CONNECTED → INITIALIZED

**Re-initialization:** If already initialized, updates amplitudes and zeros buffer without reallocating.

#### Example Usage (Python)

```python
import zmq
import aod_server_pb2

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:8037")

# Initialize with 4 channel amplitudes (for mask 0b1111)
request = aod_server_pb2.Request()
init_req = request.initialize
init_req.channel_amplitudes_mv.extend([500, 800, 1000, 750])  # mV

socket.send(request.SerializeToString())
response_data = socket.recv()

response = aod_server_pb2.Response()
response.ParseFromString(response_data)

if response.initialize.success:
    print("AWG initialized successfully")
else:
    print(f"Error: {response.initialize.error_message}")
```

#### Common Errors

- **"Expected N amplitudes, got M"**: Amplitude count doesn't match active channels
- **"AWG setup failed: ..."**: Spectrum SDK configuration error (check hardware)
- **"Failed to allocate buffer"**: Insufficient memory for DMA buffer
- **"Command timed out"**: Initialization took longer than 1 second

---

### Stop

Stop AWG output, halt DMA transfer, and zero the software buffer.

#### Request

```protobuf
message StopRequest {
  // Empty - no parameters required
}
```

#### Response

```protobuf
message StopResponse {
  bool success = 1;
  string error_message = 2;
}
```

#### What It Does

1. Stops card output (`M2CMD_CARD_STOP`)
2. Stops DMA transfer (`M2CMD_DATA_STOPDMA`)
3. Zeros the software and GPU buffers
4. **Clears the waveform batch queue** - all queued batches are discarded
5. **Does NOT de-initialize** - AWG stays in INITIALIZED state

**Note:** If AWG is not running (CONNECTED state), Stop is a no-op and returns success. This makes Stop idempotent and safe to call multiple times.

#### Example Usage (Python)

```python
# Send Stop command
request = aod_server_pb2.Request()
request.stop.CopyFrom(aod_server_pb2.StopRequest())

socket.send(request.SerializeToString())
response_data = socket.recv()

response = aod_server_pb2.Response()
response.ParseFromString(response_data)

if response.stop.success:
    print("AWG stopped successfully")
else:
    print(f"Error: {response.stop.error_message}")
```

#### Common Errors

- **"Failed to stop AWG: ..."**: Spectrum SDK error during stop (rare)

---

### WaveformBatch

Queue a batch of waveforms with interpolated tone parameters for execution.

#### Request

```protobuf
enum TriggerType {
  TRIGGER_SOFTWARE = 0;
  TRIGGER_EXTERNAL = 1;
}

message Waveform {
  int32 duration = 1;              // In WAVEFORM_TIMESTEP units
  int32 num_tones = 2;
  int32 num_steps = 3;             // Number of interpolation points
  repeated int32 time_steps = 4;   // Interpolation x-coordinates (in WAVEFORM_TIMESTEP units)
  repeated float frequencies = 5;   // Flattened: [step][channel][tone]
  repeated float amplitudes = 6;    // Flattened: [step][channel][tone]
  repeated float offset_phases = 7; // Flattened: [step][channel][tone]
}

message WaveformBatchRequest {
  TriggerType trigger_type = 1;
  int32 delay = 2;                 // Delay from trigger (in WAVEFORM_TIMESTEP units)
  repeated Waveform waveforms = 3;
}
```

#### Response

```protobuf
message WaveformBatchResponse {
  bool success = 1;
  string error_message = 2;
  int32 batch_id = 3;  // Unique ID for this batch
}
```

#### Parameters

**Batch-level:**
- `trigger_type`: SOFTWARE (0) or EXTERNAL (1)
- `delay`: Delay from trigger in WAVEFORM_TIMESTEP units
- `waveforms`: Array of waveforms to execute sequentially

**Per-waveform:**
- `duration`: Waveform length in WAVEFORM_TIMESTEP units
- `num_tones`: Number of frequency tones
- `num_steps`: Number of interpolation points for tone parameters
- `time_steps`: Interpolation time coordinates (size: num_steps)
  - Values in WAVEFORM_TIMESTEP units
  - Defines when each interpolation point occurs

**Tone parameter arrays:**
- All have size: `num_steps × num_channels × num_tones`
- Indexing: `[step × num_channels × num_tones + channel × num_tones + tone]`
- `frequencies`: In Hz (actual frequency, e.g., 80.5e6 for 80.5 MHz)
- `amplitudes`: Normalized amplitude (typically 0.0 to 1.0)
- `offset_phases`: Phase offset in radians

#### What It Does

1. Validates all array sizes match declared dimensions
2. Stores batch in fixed-size queue (max: MAX_WAVEFORM_BATCHES from config)
3. Appends to existing queue (batches execute in order)
4. Returns unique batch_id for tracking
5. **Does NOT execute** - batches are stored for future Start command

**State**: No state change (stays INITIALIZED)

**Queue behavior**: Batches are appended, not replaced

#### Example Usage (Python)

```python
import aod_server_pb2

# Create batch request
request = aod_server_pb2.Request()
batch_req = request.waveform_batch

batch_req.trigger_type = aod_server_pb2.TRIGGER_SOFTWARE
batch_req.delay = 10  # 10 timesteps after trigger

# Add a waveform
wf = batch_req.waveforms.add()
wf.duration = 100
wf.num_tones = 2
wf.num_steps = 3

# Define interpolation times
wf.time_steps.extend([0, 50, 100])

# Define tone parameters (flattened)
# For 4 channels, 2 tones, 3 steps: size = 3*4*2 = 24
num_channels = 4
size = wf.num_steps * num_channels * wf.num_tones  # 24

# Example: linearly varying frequency (in Hz)
frequencies = []
for step in range(wf.num_steps):
    for ch in range(num_channels):
        for tone in range(wf.num_tones):
            # AOD frequencies typically 54-82 MHz
            freq_hz = 70e6 + 1e6 * tone + 0.1e6 * step  # Hz
            frequencies.append(freq_hz)

wf.frequencies.extend(frequencies)
wf.amplitudes.extend([1.0] * size)  # Constant amplitude
wf.offset_phases.extend([0.0] * size)  # Zero phase

# Send request
socket.send(request.SerializeToString())
response_data = socket.recv()

response = aod_server_pb2.Response()
response.ParseFromString(response_data)

if response.waveform_batch.success:
    print(f"Batch queued with ID: {response.waveform_batch.batch_id}")
else:
    print(f"Error: {response.waveform_batch.error_message}")
```

#### Common Errors

- **"Batch queue full (16 batches)"**: Too many batches queued, need to execute/clear them
- **"Waveform N: frequencies size mismatch"**: Array size doesn't match num_steps × num_channels × num_tones
- **"Waveform N: time_steps size mismatch"**: time_steps array size doesn't match num_steps

#### Notes

- Waveform durations and delays are in **WAVEFORM_TIMESTEP units** (compile-time config, typically 512 samples)
- Consecutive waveforms must be separated by at least 1 timestep
- Frequencies are in **Hz** (e.g., 80.5e6 for 80.5 MHz AOD frequency)
- Time resolution: 1 WAVEFORM_TIMESTEP = WAVEFORM_TIMESTEP / sample_rate seconds
- Maximum batches in queue: MAX_WAVEFORM_BATCHES (compile-time config, default: 16)

---

## Future Commands (Planned)

The following commands are planned for future implementation:

### Start

Execute queued waveform batches and begin FIFO streaming.

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

### Start

Start AWG output and begin FIFO streaming.

```protobuf
message StartRequest {
  // To be defined
}

message StartResponse {
  bool success = 1;
  string error_message = 2;
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

## AWG State Machine

The AWG follows this state machine:

```
DISCONNECTED → CONNECTED → INITIALIZED → STREAMING
       ↑            ↑            ↓
       └────────────┴───────── STOP
```

**States:**
- **DISCONNECTED**: No hardware connection
- **CONNECTED**: Hardware connected, not configured
- **INITIALIZED**: Configured with buffers allocated, ready to stream
- **STREAMING**: Actively outputting waveforms (future)

**Transitions:**
- `start()` (server startup): DISCONNECTED → CONNECTED
- `Initialize`: CONNECTED → INITIALIZED (or INITIALIZED → INITIALIZED on re-init)
- `Stop`: Stops output but keeps INITIALIZED state (idempotent)
- `Start` (future): INITIALIZED → STREAMING
- `disconnectHardware()`: Any → DISCONNECTED

## Error Handling

All commands return synchronous responses with detailed error messages.

**Response Pattern:**
```protobuf
message SomeResponse {
  bool success = 1;
  string error_message = 2;  // Populated on failure
}
```

**Error Sources:**
- **Validation errors**: Invalid parameters (e.g., wrong amplitude count)
- **Spectrum SDK errors**: Hardware configuration failures (includes SDK error text)
- **State errors**: Command called in wrong state (e.g., Stop when not initialized)
- **Timeout errors**: Command took longer than 1 second to complete
- **Allocation errors**: Failed to allocate buffers

**Example Error Messages:**
- `"Expected 4 amplitudes for active channels, got 2"`
- `"AWG setup failed: Call: (SPC_CARDMODE, ...) -> invalid mode"`
- `"AWG not initialized or streaming (current state: 1)"`
- `"Command timed out after 1000ms"`

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
