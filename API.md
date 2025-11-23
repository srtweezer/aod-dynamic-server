# AOD Dynamic Server API Documentation

This document describes the JSON-based API for communicating with the AOD Dynamic Server using zero-copy ZeroMQ messaging.

## Communication Protocol

- **Transport**: ZeroMQ REQ-REP pattern
- **Serialization**: JSON (metadata) + raw binary arrays (waveform data)
- **Default Port**: Configured at compile-time in `config.cmake` (default: 5555)
- **Response Time**: Commands are synchronous with 1-second timeout
- **Zero-Copy**: Waveform data transferred as raw NumPy arrays without serialization overhead

## Message Structure

### Simple Commands (INITIALIZE, START, STOP)

**Single-part ZMQ message:**
- Part 0: JSON metadata

### WAVEFORM_BATCH Command

**Multi-part ZMQ message:**
- Part 0: JSON metadata
- Part 1: timesteps array (int32, binary)
- Part 2: do_generate array (uint8, binary)
- Part 3: frequencies array (float32, binary, flattened)
- Part 4: amplitudes array (float32, binary, flattened)
- Part 5: offset_phases array (float32, binary, flattened)

### Response Format

All responses are single-part JSON messages:

```json
{
  "success": true,
  "error_message": "",
  "batch_id": 123  // Only for WAVEFORM_BATCH
}
```

---

## Available Commands

### INITIALIZE

Configure AWG hardware with channel amplitudes and prepare for waveform generation.

#### Request

```json
{
  "command": "INITIALIZE",
  "amplitudes_mv": [1000, 1000, 1000, 1000]
}
```

**Parameters:**
- `amplitudes_mv`: Array of amplitudes in millivolts (integers)
  - Size must match the number of active channels in `AWG_CHANNEL_MASK` (compile-time config)
  - Example: If `AWG_CHANNEL_MASK = 0b1111` (4 channels), provide 4 amplitudes
  - Values are applied in channel order

#### Response

```json
{
  "success": true,
  "error_message": ""
}
```

#### What It Does

1. Sets AWG mode to FIFO single replay
2. Enables channels based on compile-time `AWG_CHANNEL_MASK`
3. Configures each active channel with amplitude, filter, and enables output
4. Configures clock (internal PLL) and sample rate
5. Sets software trigger
6. Allocates DMA buffer and initializes GPU arrays

**State Transition:** CONNECTED → INITIALIZED

**Re-initialization:** If already initialized, updates amplitudes and zeros buffers without reallocating.

#### Example Usage (Python)

```python
import zmq
import json

# Connect to server
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:8037")

# Send Initialize command
request = {
    "command": "INITIALIZE",
    "amplitudes_mv": [500, 800, 1000, 750]
}
socket.send_json(request)

# Receive response
response = socket.recv_json()

if response["success"]:
    print("AWG initialized successfully")
else:
    print(f"Error: {response['error_message']}")
```

#### Common Errors

- **"Expected N amplitudes, got M"**: Amplitude count doesn't match active channels
- **"AWG setup failed: ..."**: Spectrum SDK configuration error
- **"Failed to allocate buffer"**: Insufficient memory

---

### START

Start AWG output and begin FIFO streaming (future implementation).

#### Request

```json
{
  "command": "START"
}
```

#### Response

```json
{
  "success": false,
  "error_message": "START not yet implemented"
}
```

**Note:** Currently returns not implemented. Future implementation will begin streaming waveforms from GPU to AWG hardware.

---

### STOP

Stop AWG output, halt DMA transfer, and clear waveform data.

#### Request

```json
{
  "command": "STOP"
}
```

#### Response

```json
{
  "success": true,
  "error_message": ""
}
```

#### What It Does

1. Stops card output (`M2CMD_CARD_STOP`)
2. Stops DMA transfer (`M2CMD_DATA_STOPDMA`)
3. Zeros software and GPU buffers
4. **Resets timeline** - clears all waveform data from GPU
5. **Does NOT de-initialize** - AWG stays in INITIALIZED state

**Note:** Stop is idempotent - safe to call multiple times even if not running.

#### Example Usage (Python)

```python
request = {"command": "STOP"}
socket.send_json(request)
response = socket.recv_json()

if response["success"]:
    print("AWG stopped successfully")
```

---

### WAVEFORM_BATCH

Upload a pre-flattened timeline of waveform data directly to GPU.

#### Message Format

**Part 0 - JSON Metadata:**

```json
{
  "command": "WAVEFORM_BATCH",
  "batch_id": 100,
  "trigger_type": "software",
  "num_timesteps": 1024,
  "num_tones": 64
}
```

**Note:** `num_channels` is NOT sent - server uses compile-time `AWG_CHANNEL_MASK` configuration

**Part 1 - timesteps (int32):**
- Shape: `[num_timesteps]`
- Units: WAVEFORM_TIMESTEP (compile-time config, typically 512 samples)
- Description: Time coordinate for each timestep

**Part 2 - do_generate (uint8):**
- Shape: `[num_timesteps]`
- Values: 0 (delay period) or 1 (generate waveform)
- Description: Flag indicating whether to generate waveform at this timestep

**Part 3 - frequencies (float32):**
- Shape: `[num_timesteps, num_channels, num_tones]` (flattened)
- Units: Hz
- Description: Frequency for each tone at each timestep
- Note: `num_tones` is actual tones used (can be < AOD_MAX_TONES). Server pads to AOD_MAX_TONES with zeros.

**Part 4 - amplitudes (float32):**
- Shape: `[num_timesteps, num_channels, num_tones]` (flattened)
- Units: Normalized (0.0 to 1.0 typical)
- Description: Amplitude for each tone at each timestep

**Part 5 - offset_phases (float32):**
- Shape: `[num_timesteps, num_channels, num_tones]` (flattened)
- Units: Radians
- Description: Phase offset for each tone at each timestep

#### Parameters

**Metadata:**
- `batch_id`: Integer identifier for this batch (client-provided, for ordering)
- `trigger_type`: "software" or "external"
- `num_timesteps`: Number of timesteps in this batch
- `num_tones`: Actual number of tones used (1 to AOD_MAX_TONES)

**Arrays:**
- All arrays must be **contiguous C-order** NumPy arrays
- Flattened arrays use row-major indexing: `[timestep][channel][tone]`
- Index formula: `timestep * num_channels * num_tones + channel * num_tones + tone`

#### Array Flattening Details

The client must flatten waveforms into a **global timeline** that accounts for:

1. **Delays**: Timesteps where `do_generate = 0`
2. **Waveform data**: Timesteps where `do_generate = 1`
3. **Zero-padding**: Unused tone slots (if actual tones < num_tones)

**Timeline Structure:**
```
[delay 0...delay_0] → [waveform_0 data] → [delay 1] → [waveform_1 data] → ...
```

Each waveform's data is interpolated across its duration, with one timestep per interpolation point.

#### Response

```json
{
  "success": true,
  "error_message": "",
  "batch_id": 123
}
```

The `batch_id` is a unique sequential integer for tracking.

#### What It Does

1. Receives and validates array sizes based on client's `num_tones`
2. Checks for duplicate `batch_id` (returns error if duplicate)
3. **Appends** batch data to GPU timeline at current end position
4. **Strided copy**: Pads client's `num_tones` to `AOD_MAX_TONES` with zeros in GPU
5. **Zero-copy transfer**: Data goes directly from ZMQ buffers to GPU memory
6. Stores batch metadata for playback ordering
7. Keeps batch_ids in sorted order for sequential playback

**State**: Stays INITIALIZED (does not start streaming)

**Batch Ordering**: Batches are played in **batch_id order** (numerically sorted), not order received

**Capacity**: Total timeline length across all batches cannot exceed MAX_WAVEFORM_TIMESTEPS

**Performance**: Zero-copy design + direct GPU upload + server-side tone padding optimization

#### Batch Management Details

**Appending Behavior:**
- Each `WAVEFORM_BATCH` command **appends** data to GPU timeline
- Batches sent with `batch_id` 100, 200, 300 will play in that order
- Can send batches out of order: 300, 100, 200 → plays as 100, 200, 300
- Timeline is cumulative: batch at index 0, next at index (prev_length), etc.

**Duplicate Handling:**
- Sending same `batch_id` twice returns error
- Use `STOP` to clear all batches and start fresh

**Capacity:**
- MAX_WAVEFORM_TIMESTEPS applies to **total** of all batches
- Example: 3 batches of 1000 timesteps each = 3000 total (OK if limit is 16384)

#### Example Usage (Python)

```python
import numpy as np
import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:8037")

# Client-side timeline flattening (example with 2 waveforms)
# Waveform 1: 10 timestep delay, then 100 timesteps of data
# Waveform 2: 5 timestep delay, then 50 timesteps of data
num_timesteps = 10 + 100 + 5 + 50  # 165 total
num_channels = 4  # Must match server's AWG_CHANNEL_MASK config
num_tones = 2     # Actual tones used (can be < AOD_MAX_TONES = 128)

# Pre-allocate arrays
timesteps = np.zeros(num_timesteps, dtype=np.int32)
do_generate = np.zeros(num_timesteps, dtype=np.uint8)
frequencies = np.zeros((num_timesteps, num_channels, num_tones), dtype=np.float32)
amplitudes = np.zeros((num_timesteps, num_channels, num_tones), dtype=np.float32)
phases = np.zeros((num_timesteps, num_channels, num_tones), dtype=np.float32)

# Fill in waveform 1 delay (timesteps 0-9)
idx = 0
for d in range(10):
    timesteps[idx] = d
    do_generate[idx] = 0  # Delay, no generation
    idx += 1

# Fill in waveform 1 data (timesteps 10-109)
for step in range(100):
    timesteps[idx] = step
    do_generate[idx] = 1  # Generate waveform
    # Fill frequencies, amplitudes, phases for all channels/tones
    for ch in range(num_channels):
        for tone in range(num_tones):  # Use actual num_tones
            frequencies[idx, ch, tone] = 70e6 + 1e6 * tone  # Hz
            amplitudes[idx, ch, tone] = 1.0
            phases[idx, ch, tone] = 0.0
    idx += 1

# Fill in waveform 2 delay (timesteps 110-114)
for d in range(5):
    timesteps[idx] = d
    do_generate[idx] = 0
    idx += 1

# Fill in waveform 2 data (timesteps 115-164)
for step in range(50):
    timesteps[idx] = step
    do_generate[idx] = 1
    for ch in range(num_channels):
        for tone in range(num_tones):
            frequencies[idx, ch, tone] = 72e6 + 1e6 * tone
            amplitudes[idx, ch, tone] = 0.8
            phases[idx, ch, tone] = 0.0
    idx += 1

# Flatten 3D arrays for transmission
freq_flat = frequencies.ravel()
amp_flat = amplitudes.ravel()
phase_flat = phases.ravel()

# Send metadata (client provides batch_id for ordering)
metadata = {
    "command": "WAVEFORM_BATCH",
    "batch_id": 100,  # Client controls batch order
    "trigger_type": "software",
    "num_timesteps": num_timesteps,
    "num_tones": num_tones  # Actual tones used (not num_channels!)
}
socket.send_json(metadata, zmq.SNDMORE)

# Send arrays (zero-copy)
socket.send(timesteps, copy=False, zmq.SNDMORE)
socket.send(do_generate, copy=False, zmq.SNDMORE)
socket.send(freq_flat, copy=False, zmq.SNDMORE)
socket.send(amp_flat, copy=False, zmq.SNDMORE)
socket.send(phase_flat, copy=False)

# Receive response
response = socket.recv_json()

if response["success"]:
    print(f"Waveform uploaded with batch_id: {response['batch_id']}")
else:
    print(f"Error: {response['error_message']}")
```

#### Common Errors

- **"Duplicate batch_id: N"**: Batch with this ID already uploaded (use different ID)
- **"Total timeline would exceed MAX_WAVEFORM_TIMESTEPS"**: All batches together exceed limit
- **"Array size mismatch"**: Array dimensions don't match metadata
- **"Invalid num_tones"**: Must be between 1 and AOD_MAX_TONES
- **"Failed to receive array part N"**: Network error receiving array data

#### Performance Notes

- **Zero-copy**: Arrays sent via `socket.send(array, copy=False)` avoid Python serialization
- **Direct GPU transfer**: Data copied from ZMQ buffers straight to GPU memory
- **No intermediate storage**: No CPU arrays stored server-side
- **Throughput**: Limited by network bandwidth, not serialization overhead
- **Typical latency**: < 10ms for small batches on localhost, scales with array size

---

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
- `INITIALIZE`: CONNECTED → INITIALIZED
- `STOP`: Clears waveforms but stays INITIALIZED
- `START` (future): INITIALIZED → STREAMING

---

## Error Handling

All commands return JSON responses with `success` and `error_message` fields.

**Example Error Response:**
```json
{
  "success": false,
  "error_message": "Array size mismatch: expected 262144 floats, got 131072"
}
```

**Error Categories:**
- **Validation errors**: Invalid parameters or array sizes
- **Spectrum SDK errors**: Hardware configuration failures
- **State errors**: Command called in wrong state
- **Timeout errors**: Command took longer than 1 second
- **Allocation errors**: Failed to allocate GPU memory

---

## Client Implementation Guide

### Python Client Template

```python
import zmq
import numpy as np
import json

class AODClient:
    def __init__(self, host="localhost", port=8037):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        self.num_channels = 4  # From server config
        self.max_tones = 128   # From server config (AOD_MAX_TONES)

    def _send_command(self, metadata, arrays=None):
        """Send command with optional arrays (zero-copy)"""
        if arrays:
            self.socket.send_json(metadata, zmq.SNDMORE)
            for i, arr in enumerate(arrays):
                arr_contig = np.ascontiguousarray(arr)
                flags = 0 if i == len(arrays)-1 else zmq.SNDMORE
                self.socket.send(arr_contig, copy=False, flags=flags)
        else:
            self.socket.send_json(metadata)

        return self.socket.recv_json()

    def initialize(self, amplitudes_mv):
        """Initialize AWG with channel amplitudes"""
        response = self._send_command({
            'command': 'INITIALIZE',
            'amplitudes_mv': list(amplitudes_mv)
        })
        if not response['success']:
            raise RuntimeError(response['error_message'])

    def start(self):
        """Start AWG streaming (future implementation)"""
        response = self._send_command({'command': 'START'})
        if not response['success']:
            raise RuntimeError(response['error_message'])

    def stop(self):
        """Stop AWG output"""
        response = self._send_command({'command': 'STOP'})
        if not response['success']:
            raise RuntimeError(response['error_message'])

    def send_waveform_batch(self, batch_id, timesteps, do_generate,
                           frequencies, amplitudes, phases,
                           trigger_type='software'):
        """
        Send pre-flattened waveform timeline.

        Args:
            batch_id: Integer identifier for playback ordering
            timesteps: int32 array [num_timesteps]
            do_generate: uint8 array [num_timesteps]
            frequencies: float32 array [num_timesteps, num_channels, num_tones]
            amplitudes: float32 array [num_timesteps, num_channels, num_tones]
            phases: float32 array [num_timesteps, num_channels, num_tones]
            trigger_type: 'software' or 'external'

        Returns:
            batch_id: Echo of provided batch_id (for confirmation)

        Note: num_tones inferred from frequencies.shape[2] (actual tones used)
              Server pads to AOD_MAX_TONES automatically
        """
        num_timesteps = len(timesteps)
        num_tones = frequencies.shape[2]  # Actual tones from array shape

        # Flatten 3D arrays
        freq_flat = frequencies.ravel()
        amp_flat = amplitudes.ravel()
        phase_flat = phases.ravel()

        response = self._send_command(
            {
                'command': 'WAVEFORM_BATCH',
                'batch_id': batch_id,
                'trigger_type': trigger_type,
                'num_timesteps': num_timesteps,
                'num_tones': num_tones
            },
            arrays=[timesteps, do_generate, freq_flat, amp_flat, phase_flat]
        )

        if not response['success']:
            raise RuntimeError(response['error_message'])

        return response['batch_id']

    def close(self):
        """Close socket and context"""
        self.socket.close()
        self.context.term()

# Usage example
client = AODClient()
client.initialize([1000, 1000, 1000, 1000])

# Send multiple waveform batches (they'll play in batch_id order)
# ... (create timeline arrays for batch 100) ...
client.send_waveform_batch(100, timesteps1, do_generate1,
                           frequencies1, amplitudes1, phases1)

# ... (create timeline arrays for batch 200) ...
client.send_waveform_batch(200, timesteps2, do_generate2,
                           frequencies2, amplitudes2, phases2)

print("Uploaded 2 batches - will play in order: 100, 200")

client.stop()  # Clears all batches
client.close()
```

### Required Dependencies

```bash
pip install pyzmq numpy
```

**No protobuf required!**

---

## Network Configuration

Server bind address configured in `config.cmake`:

```cmake
set(SERVER_PORT 8037 CACHE STRING "ZMQ REQ-REP server port")
set(SERVER_BIND_ADDRESS "tcp://*" CACHE STRING "ZMQ bind address")
```

**Client connection strings:**
- Local: `tcp://localhost:8037`
- Remote: `tcp://192.168.1.100:8037`

---

## Performance Characteristics

### Latency

- Simple commands (INITIALIZE, START, STOP): < 1 ms on localhost
- WAVEFORM_BATCH: Depends on array size
  - 1000 timesteps, 4 channels, 64 tones: ~5-10 ms on localhost
  - Includes network transfer + GPU copy

### Throughput

- **Zero-copy benefits**: No serialization overhead for array data
- **Direct GPU path**: Network → ZMQ buffer → GPU (single copy)
- **Bottleneck**: Usually network bandwidth, not CPU
- **Typical**: 100-500 MB/s on localhost, limited by ZMQ/kernel

### Memory Efficiency

- **Server**: No intermediate CPU storage of waveform arrays
- **Client**: NumPy arrays sent directly from Python memory
- **GPU**: Arrays copied once from ZMQ buffers to device memory

---

## Configuration Reference

Key compile-time configuration parameters from `config.cmake`:

```cmake
# Maximum timesteps in global waveform batch arrays
set(MAX_WAVEFORM_TIMESTEPS 16384)

# Maximum number of simultaneous tones per channel
set(AOD_MAX_TONES 128)

# Waveform timestep (samples)
set(WAVEFORM_TIMESTEP 512)

# AWG channel mask (bitwise: bit 0 = CH0, bit 1 = CH1, etc.)
set(AWG_CHANNEL_MASK 0b1111)  # All 4 channels

# AWG sample rate (Hz)
set(AWG_SAMPLE_RATE 625000000)  # 625 MHz
```

### Time Resolution

Time resolution = `WAVEFORM_TIMESTEP / AWG_SAMPLE_RATE`

Example with defaults:
- 512 samples / 625 MHz = 0.8192 μs per timestep

---

## Versioning

API version: **0.2.0** (zero-copy JSON implementation)

**Breaking changes from 0.1.0:**
- Removed Protocol Buffers entirely
- Waveform data sent as pre-flattened arrays (client-side flattening)
- New multi-part message format for WAVEFORM_BATCH
- Simplified command structure (JSON-based)

---

## Migration from Protobuf API (v0.1.0)

Key differences:

1. **No Protobuf**: Use JSON and NumPy arrays instead
2. **Client flattens timeline**: Server no longer builds global timeline from per-waveform data
3. **Zero-copy arrays**: Use `socket.send(array, copy=False)` for performance
4. **No batch queue**: Each WAVEFORM_BATCH command overwrites previous data
5. **Direct GPU upload**: Data goes straight to GPU, no CPU storage

**Migration steps:**
1. Remove protobuf dependency
2. Convert per-waveform data to global timeline on client
3. Flatten arrays with delays and waveform data
4. Send via multi-part ZMQ message

---

**Note**: This API is under active development. The WAVEFORM_BATCH command is implemented; START command streaming is planned.
