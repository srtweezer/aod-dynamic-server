#!/usr/bin/env python3
"""
Simple test client for AOD Dynamic Server
Tests the Ping command
"""

import zmq
import sys
import os

# Add build directory to path for protobuf imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build', 'proto'))

try:
    import aod_server_pb2
except ImportError:
    print("Error: Could not import aod_server_pb2")
    print("Please generate Python protobuf files:")
    print("  cd proto && protoc --python_out=../build/proto aod_server.proto")
    sys.exit(1)

def test_ping(host="localhost", port=5555):
    """Test the ping command"""
    print(f"Connecting to AOD server at {host}:{port}...")

    # Create ZMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{host}:{port}")

    # Set timeout
    socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout

    try:
        # Create ping request
        request = aod_server_pb2.Request()
        request.ping.CopyFrom(aod_server_pb2.PingRequest())

        print("Sending Ping request...")
        socket.send(request.SerializeToString())

        # Receive response
        response_data = socket.recv()
        response = aod_server_pb2.Response()
        response.ParseFromString(response_data)

        # Extract timestamp
        if response.HasField('ping'):
            timestamp_ns = response.ping.timestamp_ns
            timestamp_s = timestamp_ns / 1e9
            print(f"✓ Ping successful!")
            print(f"  Server timestamp: {timestamp_ns} ns ({timestamp_s:.6f} s)")
            return True
        else:
            print("✗ Unexpected response type")
            return False

    except zmq.Again:
        print("✗ Timeout: Server did not respond")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    success = test_ping()
    sys.exit(0 if success else 1)
