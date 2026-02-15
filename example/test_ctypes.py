from ctypes import CDLL, Structure, c_float, c_uint, POINTER
import numpy as np
import os

# Constants (must match C)
MAX_NEURONS = 8      # adjust to match C code
MAX_CONNECTIONS = 6     # adjust to match C code
INPUT_SIZE = 6        # adjust to match C code

# Define C struct in Python
class Neuron(Structure):
    _fields_ = [
        ("state", c_float),
        ("output", c_float),
        ("num_connections", c_uint),
        ("layer_id", c_uint)
    ]

# Load shared library
lib_path = os.path.join(os.getcwd(), "neural_web.so")
lib = CDLL(lib_path)

# Set argument types and return type
lib.initializeNeurons.argtypes = [
    POINTER(Neuron),       # neurons
    POINTER(c_uint),       # connections
    POINTER(c_float),      # weights
    POINTER(c_float)       # input_tensor
]
lib.initializeNeurons.restype = None

# Allocate arrays
neurons = (Neuron * MAX_NEURONS)()
connections = (c_uint * (MAX_NEURONS * MAX_CONNECTIONS))()
weights = (c_float * (MAX_NEURONS * MAX_CONNECTIONS))()

# Input tensor (example: random floats)
input_tensor_np = np.random.rand(INPUT_SIZE).astype(np.float32)
input_tensor = (c_float * INPUT_SIZE)(*input_tensor_np)

# Call the C function
lib.initializeNeurons(neurons, connections, weights, input_tensor)

print("First 5 neurons:")
for i in range(5):
    print(f"Neuron {i}: state={neurons[i].state}, output={neurons[i].output}, "
          f"num_connections={neurons[i].num_connections}, layer_id={neurons[i].layer_id}")

print("\nFirst 10 connections:", list(connections[:10]))
print("\nFirst 10 weights:", list(weights[:10]))
