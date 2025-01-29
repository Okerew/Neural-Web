#include <metal_stdlib>
using namespace metal;

// Constants for neural network parameters
constant float DECAY_RATE = 0.8f;
constant float INPUT_WEIGHT = 0.1f;
constant float CONNECTION_WEIGHT = 0.2f;
constant float ACTIVATION_SCALE = 1.5f;
constant float ACTIVATION_BIAS = 0.1f;
constant float MIN_ACTIVATION = -1.0f;
constant float MAX_ACTIVATION = 1.0f;
constant float LEARNING_RATE = 0.01f;
constant float WEIGHT_DECAY = 0.1f;
constant float MIN_WEIGHT = -1.0f;
constant float MAX_WEIGHT = 1.0f;
constant uint MAX_NEURONS = 1024;     
constant uint MAX_CONNECTIONS = 16;   

struct MemoryEntry {
    float vector[MAX_NEURONS * 2]; // Stores neuron states and outputs
    float importance;              // Importance of the memory
    uint timestamp;                // Timestamp of memory creation
};

struct Neuron {
    float state;
    float output;
    uint num_connections;
    uint layer_id;
};

// Fast approximation of tanh for better performance
static inline float fast_tanh(float x) {
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return clamp(a / b, MIN_ACTIVATION, MAX_ACTIVATION);
}

// Activation function with configurable response curve
static inline float activation_function(float x, float scale, float bias) {
    // Apply scale and bias
    float scaled = x * scale + bias;
    
    // Use fast tanh approximation
    float base_activation = fast_tanh(scaled);
    
    // Add nonlinearity for more dynamic response
    float sign_val = metal::sign(base_activation);
    float abs_val = metal::abs(base_activation);
    return sign_val * metal::pow(abs_val, 1.1f);
}

kernel void update_neurons(device Neuron* neurons [[buffer(0)]],
                         device const float* weights [[buffer(1)]],
                         device const uint* connections [[buffer(2)]],
                         device const uint& max_neurons [[buffer(3)]],
                         device const uint& max_connections [[buffer(4)]],
                         device const float* input_tensor [[buffer(5)]],
                         device const uint& input_size [[buffer(6)]],
                         device const float* recurrent_weights [[buffer(7)]],
                         uint id [[thread_position_in_grid]]) {
    // Early exit for out of bounds threads
    if (id >= max_neurons) return;
    
    // Load neuron data into thread-local storage
    float current_state = neurons[id].state;
    float current_output = neurons[id].output;
    uint num_conn = neurons[id].num_connections;
    uint layer = neurons[id].layer_id;
    
    // Calculate weighted sum of inputs from connected neurons
    float weighted_sum = 0.0f;
    
    // Process connections
    for (uint i = 0; i < num_conn; i++) {
        uint conn_idx = id * max_connections + i;
        uint target = connections[conn_idx];
        
        // Add weight scaling based on layer depth
        float depth_scale = 1.0f / (1.0f + layer);
        float connection_strength = weights[conn_idx] * depth_scale;
        
        // Combine state and output influences
        weighted_sum += neurons[target].state * connection_strength * 0.6f +
                       neurons[target].output * connection_strength * 0.4f;
    }
    
    // Calculate input influence with temporal dynamics
    float input_influence = input_tensor[id % input_size];
    float temporal_factor = 1.0f / (1.0f + id % 4); // Creates wave-like temporal patterns
    
    // Update state with multiple influences
    float new_state = current_state * DECAY_RATE +
                     weighted_sum * CONNECTION_WEIGHT +
                     input_influence * INPUT_WEIGHT * temporal_factor;
    
    // Add recurrent connection influence
    float recurrent_influence = current_output * recurrent_weights[id];
    new_state += recurrent_influence * 0.15f;
    
    // Apply activation function with dynamic scaling
    float dynamic_scale = ACTIVATION_SCALE * (1.0f + 0.1f * metal::sin(input_influence * M_PI_F));
    float new_output = activation_function(new_state, dynamic_scale, ACTIVATION_BIAS);
    
    // Add slight randomization for variability
    float random_val = metal::fract(metal::sin(dot(float2(float(id), new_state), 
                                                  float2(12.9898f, 78.233f))) * 43758.5453f);
    new_output += random_val * 0.01f;
    
    // Ensure outputs stay within valid range
    new_output = metal::clamp(new_output, MIN_ACTIVATION, MAX_ACTIVATION);
    
    // Write back results
    neurons[id].state = new_state;
    neurons[id].output = new_output;
}

kernel void update_weights(device float* weights [[buffer(0)]],
                         device const Neuron* neurons [[buffer(1)]],
                         device const uint* connections [[buffer(2)]],
                         device const float& learning_rate [[buffer(3)]],
                         device const uint& max_neurons [[buffer(4)]],
                         device const uint& max_connections [[buffer(5)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= max_neurons * max_connections) return;
    
    uint neuron_idx = id / max_connections;
    uint conn_idx = id % max_connections;
    
    if (conn_idx >= neurons[neuron_idx].num_connections) return;
    
    uint target_idx = connections[id];
    
    // Hebbian learning with normalization
    float pre_activation = neurons[neuron_idx].state;
    float post_activation = neurons[target_idx].output;
    float current_weight = weights[id];
    
    // Calculate weight update
    float hebbian_term = pre_activation * post_activation;
    float normalization_term = current_weight * WEIGHT_DECAY;
    float delta_w = learning_rate * (hebbian_term - normalization_term);
    
    // Update weight with momentum
    float momentum = 0.9f;
    float new_weight = current_weight + delta_w;
    new_weight = momentum * current_weight + (1.0f - momentum) * new_weight;
    
    // Clip weights
    weights[id] = metal::clamp(new_weight, MIN_WEIGHT, MAX_WEIGHT);
}

kernel void process_neurons(device Neuron* neurons [[buffer(0)]],
                          device const float* weights [[buffer(1)]],
                          device const uint* connections [[buffer(2)]],
                          device const uint& max_neurons [[buffer(3)]],
                          device const uint& max_connections [[buffer(4)]],
                          device const float* input_tensor [[buffer(5)]],
                          device const uint& input_size [[buffer(6)]],
                          device const float* recurrent_weights [[buffer(7)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= max_neurons) return;
    
    float current_state = neurons[id].state;
    float current_output = neurons[id].output;
    uint num_conn = neurons[id].num_connections;
    uint layer = neurons[id].layer_id;
    
    // Calculate weighted sum
    float weighted_sum = 0.0f;
    for (uint i = 0; i < num_conn; i++) {
        uint conn_idx = id * max_connections + i;
        uint target = connections[conn_idx];
        
        float depth_scale = 1.0f / (1.0f + layer);
        float connection_strength = weights[conn_idx] * depth_scale;
        
        weighted_sum += neurons[target].state * connection_strength * 0.6f +
                       neurons[target].output * connection_strength * 0.4f;
    }
    
    // Input processing with temporal dynamics
    float input_influence = input_tensor[id % input_size];
    float temporal_factor = 1.0f / (1.0f + id % 4);
    
    // State update with multiple influences
    float new_state = current_state * DECAY_RATE +
                     weighted_sum * CONNECTION_WEIGHT +
                     input_influence * INPUT_WEIGHT * temporal_factor;
    
    // Add recurrent influence
    float recurrent_influence = current_output * recurrent_weights[id];
    new_state += recurrent_influence * 0.15f;
    
    // Dynamic activation
    float dynamic_scale = ACTIVATION_SCALE * (1.0f + 0.1f * metal::sin(input_influence * M_PI_F));
    float new_output = activation_function(new_state, dynamic_scale, ACTIVATION_BIAS);
    
    // Add controlled randomization
    float random_val = metal::fract(metal::sin(dot(float2(float(id), new_state),
                                                  float2(12.9898f, 78.233f))) * 43758.5453f);
    new_output += random_val * 0.01f;
    
    // Clamp output
    new_output = metal::clamp(new_output, MIN_ACTIVATION, MAX_ACTIVATION);
    
    // Store results
    neurons[id].state = new_state;
    neurons[id].output = new_output;
}

kernel void backwardKernel(
    const device Neuron *neurons [[buffer(0)]],         
    const device float *weights [[buffer(1)]],        
    const device uint2 *connections [[buffer(2)]],     
    const device uint *maxNeurons [[buffer(3)]],       
    const device uint *maxConnections [[buffer(4)]],    
    const device float *targetOutputs [[buffer(5)]],    
    device float *outputErrors [[buffer(6)]],           
    const device float *learningRate [[buffer(7)]],     
    uint gid [[thread_position_in_grid]]) {              // Thread position
    if (gid >= *maxNeurons) return;

    float predictedOutput = neurons[gid].output;

    float targetOutput = targetOutputs[gid]; 

    // Compute the error between predicted and target output (e.g., for MSE)
    float error = predictedOutput - targetOutput;

    // Store the error in the outputErrors buffer
    outputErrors[gid] = error;

    float activationGradient = predictedOutput * (1.0 - predictedOutput);

    for (uint i = 0; i < *maxConnections; i++) {
        uint2 connection = connections[gid * (*maxConnections) + i];
        uint connectedNeuron = connection.x; // Source neuron index
        uint weightIndex = connection.y;     // Weight index

        // Ensure the connected neuron index is valid
        if (connectedNeuron >= *maxNeurons) continue;

        // Calculate gradient for this weight (backpropagation)
        float inputGradient = neurons[connectedNeuron].output * error * activationGradient;

        // Update weight using learning rate (gradient descent)
        float updatedWeight = weights[weightIndex] - (*learningRate) * inputGradient;

        // Store updated weight back to the weights buffer
        ((device float *)weights)[weightIndex] = updatedWeight;
    }
}

kernel void reverse_process(
    device Neuron *neurons [[buffer(0)]],
    device float *reverse_weights [[buffer(1)]],
    device uint *reverse_connections [[buffer(2)]],
    constant uint &max_neurons [[buffer(3)]],
    constant uint &max_connections [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= max_neurons) return;

    float sum = 0.0f;
    for (uint c = 0; c < max_connections; ++c) {
        uint conn_idx = id * max_connections + c;
        uint source_neuron = reverse_connections[conn_idx];

        // Accumulate contributions from reverse connections
        sum += neurons[source_neuron].output * reverse_weights[conn_idx];
    }

    // Update neuron state based on reverse pathway
    neurons[id].state += 0.1f * sum;
}

kernel void memory_replay(
    device Neuron *neurons [[buffer(0)]],
    device float *weights [[buffer(1)]],
    device uint *connections [[buffer(2)]],
    device MemoryEntry *memories [[buffer(3)]],
    constant uint &memory_capacity [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= memory_capacity) return;

    // Retrieve the memory entry
    MemoryEntry memory = memories[id];

    // Reinforce weights based on memory importance
    for (uint i = 0; i < MAX_NEURONS; ++i) {
        for (uint j = 0; j < MAX_CONNECTIONS; ++j) {
            uint conn_idx = i * MAX_CONNECTIONS + j;
            uint target_neuron = connections[conn_idx];

            // Update weights based on memory importance and neuron states
            float weight_delta = 0.01f * memory.importance * 
                                 neurons[i].output * 
                                 memory.vector[target_neuron];
            weights[conn_idx] += weight_delta;
        }
    }
}
