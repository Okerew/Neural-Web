#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <immintrin.h>
#include <json-c/json.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_NEURONS 8
#define MAX_CONNECTIONS 2
#define STEPS 100
#define INPUT_SIZE 6            // Size of the input tensor
#define MEMORY_BUFFER_SIZE 1000 // Size of circular memory buffer
#define MEMORY_VECTOR_SIZE (2 * MAX_NEURONS + INPUT_SIZE)
#define DECAY_FACTOR 0.95f           // Decay factor for memory over time
#define CONSOLIDATION_THRESHOLD 0.7f // Threshold to consolidate memories
#define STRENGTHEN_FACTOR 1.2f       // Factor to increase memory importance
#define REMOVE_THRESHOLD 0.05f // Threshold below which memory is forgotten
#define OPTIMIZATION_WINDOW 5  // Number of steps to consider for optimization
#define PERFORMANCE_THRESHOLD 0.8 // Target performance improvement threshold
#define MAX_BATCH_SIZE 16         // Maximum batch size for processing
#define EMBEDDING_SIZE 16         // Size of word embeddings
#define WEIGHT_DECAY 0.95f        // Weight decay factor
#define MAX_SIMULATIONS 10        // Number of simulation runs
#define DECAY_RATE 0.8f
#define INPUT_WEIGHT 0.1f
#define CONNECTION_WEIGHT 0.2f
#define ACTIVATION_SCALE 1.5f
#define ACTIVATION_BIAS 0.1f
#define MIN_ACTIVATION -1.0f
#define MAX_ACTIVATION 1.0f
#define LEARNING_RATE 0.01f
#define MIN_WEIGHT -1.0f
#define MAX_WEIGHT 1.0f

typedef struct {
  float state;
  float output;
  unsigned int num_connections;
  unsigned int layer_id;
} Neuron;

typedef struct {
  float vector[MEMORY_VECTOR_SIZE];
  float importance;
  unsigned int timestamp;
} MemoryEntry;

typedef struct {
  float states[MAX_NEURONS];
  float outputs[MAX_NEURONS];
  float inputs[INPUT_SIZE];
  int step;
  MemoryEntry current_memory;
} NetworkStateSnapshot;

typedef struct MemoryCluster {
  MemoryEntry *entries;
  float importance_threshold;
  unsigned int size;
  unsigned int capacity;
} MemoryCluster;

typedef struct HierarchicalMemory {
  MemoryCluster short_term;  // Recent memories with high detail
  MemoryCluster medium_term; // Consolidated memories with moderate detail
  MemoryCluster long_term;   // Highly consolidated, abstract memories
  float consolidation_threshold;
  float abstraction_threshold;
  unsigned int total_capacity;
} HierarchicalMemory;

typedef struct MemorySystem {
  HierarchicalMemory hierarchy;
  unsigned int head;
  unsigned int size;
  unsigned int capacity;
  MemoryEntry *entries;
} MemorySystem;

typedef struct {
  double execution_time;
  float average_output;
  float error_rate;
  int batch_size;
  float learning_rate;
} PerformanceMetrics;

typedef struct {
  int optimal_batch_size;
  float optimal_learning_rate;
  double best_execution_time;
  float best_performance_score;
} OptimizationState;

typedef struct {
  float input_noise_resistance;  
  float weight_noise_resistance;
  float adaptation_speed;        // Speed of recovery from perturbations
  float baseline_performance;    // Performance without noise
  float noisy_performance;       // Performance with noise
} AdaptationMetrics;

typedef struct {
  float input_noise_scale;      
  float weight_noise_scale;      
  float base_adaptation_rate;    
  float current_adaptation_rate; 
  float learning_momentum;      
  float stability_threshold;   
  float noise_tolerance;      
  float recovery_rate;          
  float plasticity;             
  float homeostatic_factor;      
} DynamicParameters;

typedef struct {
  OptimizationState opt_state;
  DynamicParameters dynamic_params;
  float best_performance_score;
  float best_stability_measure;
  unsigned long timestamp;
} SystemParameters;

typedef struct {
  int index;
  float similarity;
  unsigned int timestamp;
} PatternMatch;

typedef struct {
  float similarity_threshold; // Minimum similarity score to consider a match
  int temporal_window;        // Number of consecutive memories to consider for
                              // temporal patterns
  float temporal_decay;       // Decay factor for temporal pattern matching
  int max_matches;            // Maximum number of matches to return
} PatternMatchingParams;

typedef struct {
  char instruction[256];
  float confidence;
  bool verified;
  char reasoning[512];
} PromptVerification;

typedef struct {
  char task_description[512];
  float expected_outcome;
  char success_criteria[256];
  PromptVerification verifications[5];
} TaskPrompt;

typedef struct {
  float *region_performance_scores;
  float *region_error_rates;
  float *region_output_variance;
  int num_regions;
} NetworkPerformanceMetrics;

typedef struct {
  float meta_learning_rate;
  float exploration_factor;
  float *region_importance_scores;
  float *learning_efficiency_history;
  int num_regions;
} MetaController;

typedef struct {
  float output_stability;   // Variation in neuron's output
  float prediction_error;   
  float connection_quality; 
  float adaptive_response;  // Neuron's ability to adapt to different inputs
  float importance_score;   // Overall significance in network
} NeuronPerformanceMetric;

typedef struct {
  const char *word;
  const char *category;    // e.g., "fruit", "common", "action"
  float semantic_weight;   // How strongly this word relates to its category
  const char *connects_to; // The most likely word it connects with
  const char *description; // Detailed description of the word
} VocabularyEntry;

typedef struct {
  float prediction_weight;
  float prediction_error;
  float adaptation_rate;
} PredictiveCodingParams;

PredictiveCodingParams predictive_params[MAX_NEURONS];

typedef struct ContextNode {
    char *name;
    float importance;
    float *state_vector;
    int vector_size;
    struct ContextNode **children;
    int num_children;
    int max_children;
    struct ContextNode *parent;
    float temporal_relevance;
    int last_updated;
} ContextNode;

typedef struct GlobalContextManager {
    ContextNode *root;
    int total_nodes;
    float *global_context_vector;
    int vector_size;
    float decay_rate;
    float update_threshold;
    int max_depth;
    int max_children_per_node;
} GlobalContextManager;

typedef struct {
    float *context_weights;
    float *feedback_history;
    float adaptation_rate;
    int history_size;
    int current_index;
    float context_threshold;
    float feedback_decay;
} DynamicContextFeedback;

typedef struct {
    float *recent_outcomes;
    float *input_history;
    int history_length;
    float *correlation_matrix;
    float learning_momentum;
    float minimum_context_weight;
} ContextAdaptation;


__device__ float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

__device__ float fract(float x) { return x - floorf(x); }

__device__ float fast_tanh(float x) {
  float x2 = x * x;
  float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
  return fminf(fmaxf(a / b, MIN_ACTIVATION), MAX_ACTIVATION);
}

// Activation function with configurable response curve
__device__ float activation_function(float x, float scale, float bias) {
  // Apply scale and bias
  float scaled = x * scale + bias;

  // Use fast tanh approximation
  float base_activation = fast_tanh(scaled);

  // Add nonlinearity for more dynamic response
  float sign_val = copysignf(1.0f, base_activation);
  float abs_val = fabsf(base_activation);
  return sign_val * powf(abs_val, 1.1f);
}

__global__ void update_neurons(Neuron *neurons, const float *weights,
               const unsigned int *connections, const unsigned int max_neurons,
               const unsigned int max_connections, const float *input_tensor,
               const unsigned int input_size, const float *recurrent_weights) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_neurons)
    return;

  // Load neuron data into thread-local storage
  float current_state = neurons[id].state;
  float current_output = neurons[id].output;
  unsigned int num_conn = neurons[id].num_connections;
  unsigned int layer = neurons[id].layer_id;

  // Calculate weighted sum of inputs from connected neurons
  float weighted_sum = 0.0f;

  // Process connections
  for (unsigned int i = 0; i < num_conn; i++) {
    unsigned int conn_idx = id * max_connections + i;
    unsigned int target = connections[conn_idx];

    // Add weight scaling based on layer depth
    float depth_scale = 1.0f / (1.0f + layer);
    float connection_strength = weights[conn_idx] * depth_scale;

    // Combine state and output influences
    weighted_sum += neurons[target].state * connection_strength * 0.6f +
                    neurons[target].output * connection_strength * 0.4f;
  }

  // Calculate input influence with temporal dynamics
  float input_influence = input_tensor[id % input_size];
  float temporal_factor =
      1.0f / (1.0f + id % 4); // Creates wave-like temporal patterns

  // Update state with multiple influences
  float new_state = current_state * DECAY_RATE +
                    weighted_sum * CONNECTION_WEIGHT +
                    input_influence * INPUT_WEIGHT * temporal_factor;

  // Add recurrent connection influence
  float recurrent_influence = current_output * recurrent_weights[id];
  new_state += recurrent_influence * 0.15f;

  // Apply activation function with dynamic scaling
  float dynamic_scale =
      ACTIVATION_SCALE * (1.0f + 0.1f * sinf(input_influence * M_PI));
  float new_output =
      activation_function(new_state, dynamic_scale, ACTIVATION_BIAS);

  // Add slight randomization for variability
  float2 hash_input = make_float2(id, new_state);
  float random_val = fract(
      sinf(dot(hash_input, make_float2(12.9898f, 78.233f))) * 43758.5453f);
  new_output += random_val * 0.01f;

  // Ensure outputs stay within valid range
  new_output = fminf(fmaxf(new_output, MIN_ACTIVATION), MAX_ACTIVATION);

  // Write back results
  neurons[id].state = new_state;
  neurons[id].output = new_output;
}

__global__ void update_weights(float *weights, const Neuron *neurons,
                               const unsigned int *connections,
                               const float learning_rate,
                               const unsigned int max_neurons,
                               const unsigned int max_connections) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_neurons * max_connections)
    return;

  unsigned int neuron_idx = id / max_connections;
  unsigned int conn_idx = id % max_connections;

  if (conn_idx >= neurons[neuron_idx].num_connections)
    return;

  unsigned int target_idx = connections[id];

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
  weights[id] = fminf(fmaxf(new_weight, MIN_WEIGHT), MAX_WEIGHT);
}

__global__ void
process_neurons(Neuron *neurons, const float *weights,
                const unsigned int *connections, const unsigned int max_neurons,
                const unsigned int max_connections, const float *input_tensor,
                const unsigned int input_size, const float *recurrent_weights) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_neurons)
    return;

  float current_state = neurons[id].state;
  float current_output = neurons[id].output;
  unsigned int num_conn = neurons[id].num_connections;
  unsigned int layer = neurons[id].layer_id;

  // Calculate weighted sum
  float weighted_sum = 0.0f;
  for (unsigned int i = 0; i < num_conn; i++) {
    unsigned int conn_idx = id * max_connections + i;
    unsigned int target = connections[conn_idx];

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
  float dynamic_scale =
      ACTIVATION_SCALE * (1.0f + 0.1f * sinf(input_influence * M_PI));
  float new_output =
      activation_function(new_state, dynamic_scale, ACTIVATION_BIAS);

  // Add controlled randomization
  float2 hash_input = make_float2(id, new_state);
  float random_val = fract(
      sinf(dot(hash_input, make_float2(12.9898f, 78.233f))) * 43758.5453f);
  new_output += random_val * 0.01f;

  // Clamp output
  new_output = fminf(fmaxf(new_output, MIN_ACTIVATION), MAX_ACTIVATION);

  // Store results
  neurons[id].state = new_state;
  neurons[id].output = new_output;
}

__global__ void
backward_kernel(const Neuron *neurons, const float *weights,
                const uint2 *connections, const unsigned int max_neurons,
                const unsigned int max_connections, const float *target_outputs,
                float *output_errors, const float learning_rate) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_neurons)
    return;

  float predicted_output = neurons[id].output;
  float target_output = target_outputs[id];

  // Compute the error between predicted and target output
  float error = predicted_output - target_output;
  output_errors[id] = error;

  float activation_gradient = predicted_output * (1.0f - predicted_output);

  for (unsigned int i = 0; i < max_connections; i++) {
    uint2 connection = connections[id * max_connections + i];
    unsigned int connected_neuron = connection.x;
    unsigned int weight_index = connection.y;

    if (connected_neuron >= max_neurons)
      continue;

    // Calculate gradient for this weight (backpropagation)
    float input_gradient =
        neurons[connected_neuron].output * error * activation_gradient;

    // Update weight using learning rate (gradient descent)
    float updated_weight =
        weights[weight_index] - learning_rate * input_gradient;

    // Store updated weight back to the weights buffer
    ((float *)weights)[weight_index] = updated_weight;
  }
}

__global__ void reverse_process(Neuron *neurons, float *reverse_weights,
                                unsigned int *reverse_connections,
                                const unsigned int max_neurons,
                                const unsigned int max_connections) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_neurons)
    return;

  float sum = 0.0f;
  for (unsigned int c = 0; c < max_connections; ++c) {
    unsigned int conn_idx = id * max_connections + c;
    unsigned int source_neuron = reverse_connections[conn_idx];

    // Accumulate contributions from reverse connections
    sum += neurons[source_neuron].output * reverse_weights[conn_idx];
  }

  // Update neuron state based on reverse pathway
  neurons[id].state += 0.1f * sum;
}

__global__ void memory_replay(Neuron *neurons, float *weights,
                              unsigned int *connections, MemoryEntry *memories,
                              const unsigned int memory_capacity) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= memory_capacity)
    return;

  // Retrieve the memory entry
  MemoryEntry memory = memories[id];

  // Reinforce weights based on memory importance
  for (unsigned int i = 0; i < MAX_NEURONS; ++i) {
    for (unsigned int j = 0; j < MAX_CONNECTIONS; ++j) {
      unsigned int conn_idx = i * MAX_CONNECTIONS + j;
      unsigned int target_neuron = connections[conn_idx];

      // Update weights based on memory importance and neuron states
      float weight_delta = 0.01f * memory.importance * neurons[i].output *
                           memory.vector[target_neuron];
      weights[conn_idx] += weight_delta;
    }
  }
}

DynamicParameters initDynamicParameters() {
  DynamicParameters params = {.input_noise_scale = 0.1f,
                              .weight_noise_scale = 0.05f,
                              .base_adaptation_rate = 0.01f,
                              .current_adaptation_rate = 0.01f,
                              .learning_momentum = 0.9f,
                              .stability_threshold = 0.8f,
                              .noise_tolerance = 0.2f,
                              .recovery_rate = 0.05f,
                              .plasticity = 1.0f,
                              .homeostatic_factor = 0.1f};
  return params;
}

MemorySystem *createMemorySystem(unsigned int capacity) {
  MemorySystem *system = (MemorySystem *)malloc(sizeof(MemorySystem));

  // Initialize hierarchical structure
  system->hierarchy.short_term.capacity = capacity * 0.5; // 50% for short term
  system->hierarchy.medium_term.capacity =
      capacity * 0.3;                                    // 30% for medium term
  system->hierarchy.long_term.capacity = capacity * 0.2; // 20% for long term

  // Allocate memory for each level
  system->hierarchy.short_term.entries = (MemoryEntry *)malloc(
      system->hierarchy.short_term.capacity * sizeof(MemoryEntry));
  system->hierarchy.medium_term.entries = (MemoryEntry *)malloc(
      system->hierarchy.medium_term.capacity * sizeof(MemoryEntry));
  system->hierarchy.long_term.entries = (MemoryEntry *)malloc(
      system->hierarchy.long_term.capacity * sizeof(MemoryEntry));

  // Initialize thresholds
  system->hierarchy.short_term.importance_threshold = 0.3f;
  system->hierarchy.medium_term.importance_threshold = 0.5f;
  system->hierarchy.long_term.importance_threshold = 0.7f;
  system->hierarchy.consolidation_threshold = 0.6f;
  system->hierarchy.abstraction_threshold = 0.8f;

  // Initialize sizes
  system->hierarchy.short_term.size = 0;
  system->hierarchy.medium_term.size = 0;
  system->hierarchy.long_term.size = 0;

  // Initialize original structure for compatibility
  system->entries = system->hierarchy.short_term.entries;
  system->head = 0;
  system->size = 0;
  system->capacity = capacity;

  return system;
}

void consolidateMemory(MemorySystem *system) {
  for (int i = 0; i < system->size; i++) {
    // Strengthen memories above the threshold
    if (system->entries[i].importance > CONSOLIDATION_THRESHOLD) {
      system->entries[i].importance *= STRENGTHEN_FACTOR;
    }
    // Forget memories that have very low importance
    if (system->entries[i].importance < REMOVE_THRESHOLD) {
      // "Remove" memory by setting its importance to zero
      system->entries[i].importance = 0.0f;
    }
  }
}

void freeMemorySystem(MemorySystem *system) {
  free(system->entries);
  free(system);
}

void computeMemoryVector(float *memory_vector, Neuron *neurons,
                         float *input_tensor) {
  // Combine neuron states and input into memory vector
  int vector_idx = 0;

  // Add neuron states
  for (int i = 0; i < MAX_NEURONS && vector_idx < MEMORY_VECTOR_SIZE; i++) {
    memory_vector[vector_idx++] = neurons[i].state;
  }

  // Add neuron outputs
  for (int i = 0; i < MAX_NEURONS && vector_idx < MEMORY_VECTOR_SIZE; i++) {
    memory_vector[vector_idx++] = neurons[i].output;
  }

  // Add input tensor values
  for (int i = 0; i < INPUT_SIZE && vector_idx < MEMORY_VECTOR_SIZE; i++) {
    memory_vector[vector_idx++] = input_tensor[i];
  }

  // Fill remaining space with zeros if necessary
  while (vector_idx < MEMORY_VECTOR_SIZE) {
    memory_vector[vector_idx++] = 0.0f;
  }
}

float computeImportance(float *memory_vector) {
  float importance = 0.0f;
  for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
    importance += fabsf(memory_vector[i]);
  }
  return importance / MEMORY_VECTOR_SIZE;
}
// Find the least important memory in a given array of memories
int findLeastImportantMemory(MemoryEntry *entries, unsigned int size) {
  int least_important_idx = 0;
  float lowest_importance = entries[0].importance;

  for (unsigned int i = 1; i < size; i++) {
    if (entries[i].importance < lowest_importance) {
      lowest_importance = entries[i].importance;
      least_important_idx = i;
    }
  }

  return least_important_idx;
}

MemoryEntry abstractMemory(MemoryEntry *original) {
  MemoryEntry abstracted;
  abstracted.timestamp = original->timestamp;

  // Initialize abstracted vector with zeros
  for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
    abstracted.vector[i] = 0.0f;
  }

  // Create abstracted representation by averaging neighboring values
  // This reduces detail but maintains important patterns
  for (int i = 0; i < MEMORY_VECTOR_SIZE - 2; i += 2) {
    float avg = (original->vector[i] + original->vector[i + 1]) / 2.0f;
    abstracted.vector[i / 2] = avg;
  }

  // Increase importance for long-term storage
  abstracted.importance = original->importance * 1.2f;

  return abstracted;
}

// Helper function to consolidate memories to higher levels
void consolidateToHigherLevel(MemorySystem *system) {
  // Find memories that meet consolidation threshold
  for (unsigned int i = 0; i < system->hierarchy.medium_term.size; i++) {
    if (system->hierarchy.medium_term.entries[i].importance >=
        system->hierarchy.consolidation_threshold) {
      // Create abstracted memory for long-term storage
      MemoryEntry consolidated =
          abstractMemory(&system->hierarchy.medium_term.entries[i]);

      // Add to long-term if space available
      if (system->hierarchy.long_term.size <
          system->hierarchy.long_term.capacity) {
        system->hierarchy.long_term
            .entries[system->hierarchy.long_term.size++] = consolidated;
      }
    }
  }
}

// Helper function to consolidate to medium term
void consolidateToMediumTerm(MemorySystem *system) {
  // Find related memories in short-term
  for (unsigned int i = 0; i < system->hierarchy.short_term.size; i++) {
    if (system->hierarchy.short_term.entries[i].importance >=
        system->hierarchy.medium_term.importance_threshold) {
      // Move to medium-term memory
      if (system->hierarchy.medium_term.size <
          system->hierarchy.medium_term.capacity) {
        system->hierarchy.medium_term
            .entries[system->hierarchy.medium_term.size++] =
            system->hierarchy.short_term.entries[i];
      }
    }
  }
}

void addMemory(MemorySystem *system, Neuron *neurons, float *input_tensor,
               unsigned int timestamp) {
  MemoryEntry entry;
  computeMemoryVector(entry.vector, neurons, input_tensor);
  entry.importance = computeImportance(entry.vector);
  entry.timestamp = timestamp;

  // Determine which level to store the memory based on importance
  if (entry.importance >= system->hierarchy.long_term.importance_threshold) {
    // Add to long-term memory if space available
    if (system->hierarchy.long_term.size <
        system->hierarchy.long_term.capacity) {
      system->hierarchy.long_term.entries[system->hierarchy.long_term.size++] =
          entry;
    } else {
      // Replace least important long-term memory
      int least_important_idx =
          findLeastImportantMemory(system->hierarchy.long_term.entries,
                                   system->hierarchy.long_term.size);
      system->hierarchy.long_term.entries[least_important_idx] = entry;
    }
  } else if (entry.importance >=
             system->hierarchy.medium_term.importance_threshold) {
    // Add to medium-term memory
    if (system->hierarchy.medium_term.size <
        system->hierarchy.medium_term.capacity) {
      system->hierarchy.medium_term
          .entries[system->hierarchy.medium_term.size++] = entry;
    } else {
      // Consolidate to long-term if important enough
      consolidateToHigherLevel(system);
    }
  } else {
    // Add to short-term memory
    if (system->hierarchy.short_term.size <
        system->hierarchy.short_term.capacity) {
      system->hierarchy.short_term
          .entries[system->hierarchy.short_term.size++] = entry;
    } else {
      // Move to medium-term if important enough
      consolidateToMediumTerm(system);
    }
  }

  // Update original structure for compatibility
  system->entries[system->head] = entry;
  system->head = (system->head + 1) % system->capacity;
  if (system->size < system->capacity) {
    system->size++;
  }
}

void saveMemorySystem(MemorySystem *system, const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Error opening memory file for writing\n");
    return;
  }

  fwrite(&system->capacity, sizeof(unsigned int), 1, fp);
  fwrite(&system->size, sizeof(unsigned int), 1, fp);
  fwrite(&system->head, sizeof(unsigned int), 1, fp);
  fwrite(system->entries, sizeof(MemoryEntry), system->capacity, fp);

  fclose(fp);
}

MemorySystem *loadMemorySystem(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Error opening memory file for reading\n");
    return NULL;
  }

  unsigned int capacity;
  fread(&capacity, sizeof(unsigned int), 1, fp);

  MemorySystem *system = createMemorySystem(capacity);
  fread(&system->size, sizeof(unsigned int), 1, fp);
  fread(&system->head, sizeof(unsigned int), 1, fp);
  fread(system->entries, sizeof(MemoryEntry), capacity, fp);

  fclose(fp);
  return system;
}

void loadHierarchicalMemory(MemorySystem *system, const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Error opening file for reading hierarchical memory\n");
    return;
  }

  // Read structure sizes
  fread(&system->hierarchy.short_term.size, sizeof(unsigned int), 1, fp);
  fread(&system->hierarchy.medium_term.size, sizeof(unsigned int), 1, fp);
  fread(&system->hierarchy.long_term.size, sizeof(unsigned int), 1, fp);

  // Read memory entries for each level
  fread(system->hierarchy.short_term.entries, sizeof(MemoryEntry),
        system->hierarchy.short_term.size, fp);
  fread(system->hierarchy.medium_term.entries, sizeof(MemoryEntry),
        system->hierarchy.medium_term.size, fp);
  fread(system->hierarchy.long_term.entries, sizeof(MemoryEntry),
        system->hierarchy.long_term.size, fp);

  fclose(fp);
}

void saveNetworkStates(NetworkStateSnapshot *history, int total_steps) {
  // Create the root JSON object
  struct json_object *root = json_object_new_object();

  // Add basic information to the root object
  json_object_object_add(root, "total_steps", json_object_new_int(total_steps));
  json_object_object_add(root, "max_neurons", json_object_new_int(MAX_NEURONS));
  json_object_object_add(root, "input_size", json_object_new_int(INPUT_SIZE));

  // Create the "history" array
  struct json_object *history_array = json_object_new_array();

  for (int i = 0; i < total_steps; i++) {
    // Create a new JSON object for each step
    struct json_object *step_data = json_object_new_object();

    // Add states (array of floats)
    struct json_object *states_array = json_object_new_array();
    for (int j = 0; j < MAX_NEURONS; j++) {
      json_object_array_add(states_array,
                            json_object_new_double(history[i].states[j]));
    }
    json_object_object_add(step_data, "states", states_array);

    // Add outputs (array of floats)
    struct json_object *outputs_array = json_object_new_array();
    for (int j = 0; j < MAX_NEURONS; j++) {
      json_object_array_add(outputs_array,
                            json_object_new_double(history[i].outputs[j]));
    }
    json_object_object_add(step_data, "outputs", outputs_array);

    // Add inputs (array of floats)
    struct json_object *inputs_array = json_object_new_array();
    for (int j = 0; j < INPUT_SIZE; j++) {
      json_object_array_add(inputs_array,
                            json_object_new_double(history[i].inputs[j]));
    }
    json_object_object_add(step_data, "inputs", inputs_array);

    struct json_object *memory_object = json_object_new_object();
    json_object_object_add(step_data, "current_memory", memory_object);

    // Add this step's data to the history array
    json_object_array_add(history_array, step_data);
  }

  // Add history array to the root object
  json_object_object_add(root, "history", history_array);

  // Open the JSON file for writing
  FILE *fp = fopen("network_states.json", "w");
  if (fp == NULL) {
    printf("Error opening file for writing\n");
    json_object_put(root); // Clean up
    return;
  }

  // Write JSON to the file
  fprintf(fp, "%s", json_object_to_json_string(root));

  // Clean up
  fclose(fp);
  json_object_put(root); // Free the memory used by the JSON object
}

void initializeNeurons(Neuron *neurons, int *connections, float *weights,
                       float *input_tensor) {
  for (int i = 0; i < MAX_NEURONS; i++) {
    // Initialize state with input
    neurons[i].state = input_tensor[i % INPUT_SIZE];
    // Initialize output with transformed state
    float scale = 1.5f;
    float bias = 0.1f;
    neurons[i].output = tanh(neurons[i].state * scale + bias);
    neurons[i].num_connections = 2;
    neurons[i].layer_id = i % 2;
  }

  // Initialize connections
  for (int i = 0; i < MAX_NEURONS; i++) {
    // Create feedforward connections between layers
    if (neurons[i].layer_id == 0) {
      connections[i * MAX_CONNECTIONS] = (i + 1) % MAX_NEURONS;
      connections[i * MAX_CONNECTIONS + 1] = (i + 2) % MAX_NEURONS;
    } else {
      connections[i * MAX_CONNECTIONS] = (i + 1) % MAX_NEURONS;
      connections[i * MAX_CONNECTIONS + 1] = (i + 3) % MAX_NEURONS;
    }
  }

  // Initialize weights with varied values
  for (int i = 0; i < MAX_NEURONS * MAX_CONNECTIONS; i++) {
    // Create weights between -0.5 and 0.5
    weights[i] = (((float)rand() / RAND_MAX) - 0.5f);
  }
}

typedef struct {
  const char *word;
  const char *category;    // e.g., "fruit", "common", "action"
  float semantic_weight;   // How strongly this word relates to its category
  const char *connects_to; // The most likely word it connects with
  const char *description; // Detailed description of the word
} VocabularyEntry;

const VocabularyEntry vocabulary[] = {
    // Common English Words
    {"the", "common", 0.9f, "be",
     "Definite article identifying a specific noun"},
    {"be", "common", 0.8f, "to", "Verb expressing existence or state"},
    {"to", "common", 0.8f, "the", "Indicates direction or purpose"},
    {"of", "common", 0.9f, "and", "Shows possession or connection"},
    {"and", "common", 0.8f, "a", "Connects words, phrases, or clauses"},
    {"a", "common", 0.9f, "the", "Indefinite article for singular nouns"},
    {"in", "common", 0.7f, "the", "Indicates location or state"},
    {"that", "common", 0.6f, "have", "Introduces a descriptive clause"},
    {"have", "common", 0.5f, "it", "Indicates possession or experience"},
    {"it", "common", 0.6f, "is", "Refers to a previous noun or concept"},
    {"for", "common", 0.8f, "the", "Indicates purpose or recipient"},
    {"not", "common", 0.7f, "be", "Negation or denial"},
    {"on", "common", 0.6f, "the", "Indicates position or contact"},
    {"with", "common", 0.5f, "you", "Indicates accompaniment or method"},
    {"he", "common", 0.7f, "is", "Masculine third-person singular pronoun"},
    {"as", "common", 0.5f, "the", "Comparison or equivalence marker"},
    {"you", "common", 0.7f, "are", "Second-person pronoun"},
    {"do", "common", 0.6f, "not", "Auxiliary verb for questions or emphasis"},
    {"at", "common", 0.6f, "the", "Indicates specific location or time"},
    {"are", "common", 0.7f, "you", "Plural or second-person form of 'to be'"},

    // Fruits
    {"apple", "fruit", 1.0f, "banana",
     "Round, sweet fruit with crisp flesh, red/green/yellow"},
    {"banana", "fruit", 0.9f, "cherry",
     "Long curved tropical fruit with soft yellow flesh"},
    {"cherry", "fruit", 0.8f, "date",
     "Small round fruit, sweet and slightly tart flavor"},
    {"date", "fruit", 0.8f, "fig",
     "Sweet, wrinkled fruit often used in desserts"},
    {"elderberry", "fruit", 0.7f, "grape",
     "Small, dark purple berry with tart flavor"},
    {"fig", "fruit", 0.9f, "grape", "Sweet, soft fruit with unique texture"},
    {"grape", "fruit", 0.8f, "orange",
     "Small round fruit that grows in clusters"},
    {"honeydew", "fruit", 0.7f, "melon",
     "Pale green melon with sweet, mild flavor"},
    {"kiwi", "fruit", 0.9f, "mango",
     "Small oval fruit with fuzzy brown exterior"},
    {"lemon", "fruit", 0.8f, "lime",
     "Bright yellow citrus fruit with sour taste"},
    {"mango", "fruit", 1.0f, "papaya",
     "Sweet tropical fruit with vibrant orange flesh"},
    {"nectarine", "fruit", 0.8f, "peach",
     "Smooth-skinned stone fruit similar to peaches"},
    {"orange", "fruit", 1.0f, "tangerine",
     "Bright orange citrus fruit with sweet flavor"},
    {"papaya", "fruit", 0.9f, "mango",
     "Soft tropical fruit with green to orange color"},
    {"quince", "fruit", 0.6f, "apple", "Hard, yellow fruit used in preserves"},
    {"raspberry", "fruit", 0.8f, "blueberry",
     "Soft red berry with delicate flavor"},
    {"strawberry", "fruit", 0.9f, "cream",
     "Red, heart-shaped berry with sweet taste"},
    {"tangerine", "fruit", 0.8f, "orange", "Small, sweet citrus fruit"},
    {"ugli", "fruit", 0.5f, "fruit", "Jamaican hybrid citrus fruit"},
    {"vanilla", "fruit", 0.7f, "bean", "Flavoring derived from orchid plant"},
    {"watermelon", "fruit", 1.0f, "summer",
     "Large green fruit with sweet red interior"},
    {"xigua", "fruit", 0.6f, "melon", "Chinese word for watermelon"},

    // Vegetables
    {"yam", "vegetable", 0.8f, "potato",
     "Starchy root vegetable with various colors"},
    {"zucchini", "vegetable", 0.8f, "squash",
     "Green summer squash with mild flavor"},

    {"run", "action", 0.7f, "fast", "Moving quickly on foot with rapid steps"},
    {"jump", "action", 0.8f, "high",
     "Propelling oneself upward off the ground"},
    {"sing", "action", 0.6f, "loud", "Producing musical sounds with voice"},
    {"dance", "action", 0.7f, "music",
     "Rhythmic body movement to musical beats"},
    {"write", "action", 0.8f, "book", "Creating text by forming words"},

    {"carrot", "vegetable", 0.9f, "salad",
     "Orange root vegetable, rich in beta-carotene"},
    {"broccoli", "vegetable", 0.8f, "green",
     "Tree-like green vegetable with nutrient-dense florets"},
    {"potato", "vegetable", 0.7f, "starch",
     "Underground tuber, staple in many cuisines"},

    {"love", "emotion", 0.9f, "heart", "Deep affection, care, and connection"},
    {"hope", "emotion", 0.8f, "future", "Optimistic expectation and desire"},
    {"dream", "concept", 0.7f, "imagination",
     "Visionary mental experience or aspiration"},

    // Punctuation
    {".", "punctuation", 0.5f, NULL, "Marks the end of a declarative sentence"},

    {"fast", "adjective", 0.7f, "run",
     "Moving or capable of moving at high speed"},
    {"high", "adjective", 0.7f, "jump",
     "Extending far upward; great vertical extent"},
    {"loud", "adjective", 0.6f, "sing",
     "Producing or capable of producing much noise"},
    {"music", "noun", 0.8f, "dance",
     "Vocal or instrumental sounds combined to produce beauty of form, "
     "harmony, and expression of emotion"},
    {"book", "noun", 0.8f, "write",
     "Written or printed work consisting of pages glued or sewn together"},
    {"salad", "noun", 0.7f, "carrot",
     "Dish consisting of mixed pieces of food, typically vegetables"},
    {"green", "adjective", 0.8f, "broccoli",
     "Color between blue and yellow in the spectrum; colored like grass"},
    {"starch", "noun", 0.6f, "potato",
     "Carbohydrate consisting of a large number of glucose units joined by "
     "glycosidic bonds"},
    {"heart", "noun", 0.9f, "love",
     "Hollow muscular organ that pumps the blood through the circulatory "
     "system by rhythmic contraction and dilation"},
    {"future", "noun", 0.8f, "hope",
     "Time or a period of time following the moment of speaking or writing; "
     "time regarded as still to come"},
    {"imagination", "noun", 0.7f, "dream",
     "Faculty or action of forming new ideas, or images or concepts of "
     "external objects not present to the senses"},
    {"cream", "noun", 0.7f, "strawberry",
     "Thick white or pale yellow fatty liquid which rises to the top when milk "
     "is left to stand and which can be eaten as an accompaniment to desserts "
     "or used as a cooking ingredient"},
    {"bean", "noun", 0.6f, "vanilla",
     "Edible seed, typically kidney-shaped, growing in long pods on certain "
     "leguminous plants"},
    {"summer", "noun", 0.7f, "watermelon",
     "Warmest season of the year, in the northern hemisphere from June to "
     "August and in the southern hemisphere from December to February"},
    {"melon", "noun", 0.7f, "honeydew",
     "Large round fruit with sweet pulpy flesh and thick skin"},
    {"squash", "noun", 0.7f, "zucchini",
     "Edible gourd, typically with green skin and white flesh, eaten as a "
     "vegetable"},
    {"blueberry", "noun", 0.7f, "raspberry", "Small blue-black edible berry"},
    {"peach", "noun", 0.8f, "nectarine",
     "Soft, juicy fruit with sweet yellow or pinkish flesh and downy "
     "pinkish-yellow skin"},
    {"lime", "noun", 0.7f, "lemon",
     "Round citrus fruit with green skin and acidic juice"},
    {"is", "verb", 0.8f, "he", "Third-person singular present of 'be'"},
    {"are", "verb", 0.8f, "you",
     "Second-person singular present and plural present of 'be'"},
    {"have", "verb", 0.7f, "it", "Possess, own, or hold"},
    {"with", "preposition", 0.7f, "you",
     "Accompanied by (another person or thing)"},
    {"he", "pronoun", 0.8f, "is",
     "Used to refer to a man, boy, or male animal previously mentioned or "
     "easily identified"},
};

const int vocab_size = sizeof(vocabulary) / sizeof(vocabulary[0]);

const char *mapToWord(float value) {
  int index = (int)(fabs(value) * vocab_size) % vocab_size;
  return vocabulary[index].word;
}

void tokenizeString(const char *input, char **tokens, int *num_tokens) {
  char buffer[1024];
  strncpy(buffer, input, sizeof(buffer));
  buffer[sizeof(buffer) - 1] = '\0'; // Ensure null-termination

  *num_tokens = 0;
  char *token = strtok(buffer, " ");
  while (token != NULL && *num_tokens < INPUT_SIZE) {
    tokens[*num_tokens] = token;
    (*num_tokens)++;
    token = strtok(NULL, " ");
  }
}

// Word embeddings (randomly initialized)
float embeddings[vocab_size][EMBEDDING_SIZE];

// Initialize word embeddings
void initializeEmbeddings() {
  for (int i = 0; i < vocab_size; i++) {
    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      embeddings[i][j] =
          (float)rand() / RAND_MAX; // Random values between 0 and 1
    }
  }
}

float *getWordEmbedding(const char *word) {
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(word, vocabulary[i].word) == 0) {
      float *base_embedding = embeddings[i];
      float complexity_factor =
          strlen(vocabulary[i].description) /
          50.0f; // More detailed descriptions increase complexity

      // Scale embedding by semantic weight and description complexity
      for (int j = 0; j < EMBEDDING_SIZE; j++) {
        base_embedding[j] *=
            vocabulary[i].semantic_weight * (1.0f + complexity_factor);
      }
      return base_embedding;
    }
  }
  return embeddings[0]; // Default embedding if word is not in vocabulary
}

// Additional utility to find words by category or semantic similarity
void findWordsByCategory(const char *category) {
  printf("Words in category '%s':\n", category);
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(vocabulary[i].category, category) == 0) {
      printf("- %s (semantic weight: %.2f): %s\n", vocabulary[i].word,
             vocabulary[i].semantic_weight, vocabulary[i].description);
    }
  }
}
// Compute cosine similarity between two vectors
float cosineSimilarity(float *vec1, float *vec2, int size) {
  float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
  for (int i = 0; i < size; i++) {
    dot += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }
  return dot / (sqrtf(norm1) * sqrtf(norm2));
}

// Custom attention algorithm
void computeAttentionWeights(float *attention_weights, int step, int num_tokens,
                             float **token_embeddings,
                             MemoryEntry *relevantMemory) {
  // Step-based relevance: Words closer to the current step are more relevant
  for (int i = 0; i < num_tokens; i++) {
    float step_relevance = 1.0f / (1.0f + fabsf((float)i - (float)step));
    attention_weights[i] = step_relevance;
  }

  // Contextual similarity: Words similar to the current context are more
  // relevant
  float context[EMBEDDING_SIZE] = {0};
  if (relevantMemory) {
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      context[i] = relevantMemory->vector[i % MAX_NEURONS];
    }
  }

  for (int i = 0; i < num_tokens; i++) {
    float similarity =
        cosineSimilarity(token_embeddings[i], context, EMBEDDING_SIZE);
    attention_weights[i] += similarity;
  }

  // Normalize attention weights
  float sum = 0.0f;
  for (int i = 0; i < num_tokens; i++) {
    sum += attention_weights[i];
  }
  for (int i = 0; i < num_tokens; i++) {
    attention_weights[i] /= sum;
  }
}

void generateInputTensor(float *input_tensor, int step, const char *text_input,
                         MemoryEntry *relevantMemory,
                         SystemParameters *system_params) {
  float t = step * 0.01f;
  DynamicParameters params = system_params->dynamic_params;

  // Tokenize the text input
  char *tokens[INPUT_SIZE];
  int num_tokens = 0;
  tokenizeString(text_input, tokens, &num_tokens);

  // Enhanced token embedding with category and semantic weight consideration
  float *token_embeddings[INPUT_SIZE];
  float category_weights[INPUT_SIZE] = {0};

  for (int i = 0; i < num_tokens; i++) {
    token_embeddings[i] = getWordEmbedding(tokens[i]);

    // Find word's semantic category weight
    for (int j = 0; j < vocab_size; j++) {
      if (strcmp(tokens[i], vocabulary[j].word) == 0) {
        // Weight based on category and semantic significance
        if (strcmp(vocabulary[j].category, "action") == 0)
          category_weights[i] = 1.2f;
        else if (strcmp(vocabulary[j].category, "emotion") == 0)
          category_weights[i] = 1.1f;
        else
          category_weights[i] = 1.0f;
        break;
      }
    }
  }

  // Compute attention weights with enhanced category sensitivity
  float attention_weights[INPUT_SIZE] = {0};
  computeAttentionWeights(attention_weights, step, num_tokens, token_embeddings,
                          relevantMemory);

  // Adjust attention weights with category influence
  for (int i = 0; i < num_tokens; i++) {
    attention_weights[i] *= category_weights[i];
  }

  for (int i = 0; i < INPUT_SIZE; i++) {
    float phase = (float)i / INPUT_SIZE;
    float signal = 0.4f * sinf(2.0f * M_PI * (t + phase));
    signal += 0.4f * sinf(2.0f * M_PI * (t + phase * 1.5f));
    signal += 0.2f * sinf(5.0f * M_PI * (t + phase * 2.0f));

    // Incorporate word description complexity
    if (i < EMBEDDING_SIZE) {
      for (int j = 0; j < num_tokens; j++) {
        // Find word to get its description length
        int desc_length = 0;
        for (int k = 0; k < vocab_size; k++) {
          if (strcmp(tokens[j], vocabulary[k].word) == 0) {
            desc_length = strlen(vocabulary[k].description);
            break;
          }
        }

        // Use description length to modulate signal
        float desc_factor = 1.0f + (desc_length / 100.0f);
        signal += 0.3f * attention_weights[j] * desc_factor;
      }
    }

    // Add memory-based relevance
    if (relevantMemory) {
      signal += 0.2f * relevantMemory->vector[i % MAX_NEURONS];
    }

    // Noise and drift management
    float noise = ((float)rand() / RAND_MAX - 0.5f) * params.input_noise_scale;
    float drift = params.plasticity * sinf(0.1f * M_PI * t);

    input_tensor[i] = (signal + noise + drift + 1.0f) * 0.5f;
    input_tensor[i] = fmaxf(0.0f, fminf(1.0f, input_tensor[i]));
  }
}

void captureNetworkState(Neuron *neurons, float *input_tensor,
                         NetworkStateSnapshot *snapshot, float *weights,
                         int step) {
  snapshot->step = step;

  // Capture current states and outputs separately
  for (int i = 0; i < MAX_NEURONS; i++) {
    snapshot->states[i] = neurons[i].state;
    snapshot->outputs[i] = neurons[i].output;
  }

  // Capture input tensor
  for (int i = 0; i < INPUT_SIZE; i++) {
    snapshot->inputs[i] = input_tensor[i];
  }

  // Calculate and store updated states based on current outputs and weights
  for (int i = 0; i < MAX_NEURONS; i++) {
    float weighted_output = 0.0f;
    for (int j = 0; j < MAX_CONNECTIONS; j++) {
      weighted_output += neurons[i].output * weights[i * MAX_CONNECTIONS + j];
    }
    // Update state with both input influence and weighted output
    snapshot->states[i] =
        neurons[i].state * 0.7f +            // State persistence
        weighted_output * 0.2f +             // Output influence
        input_tensor[i % INPUT_SIZE] * 0.1f; // Input influence
  }
}

void printNetworkStates(Neuron *neurons, float *input_tensor, int step) {
  printf("Step %d:\n", step);
  printf("Input tensor: ");
  for (int i = 0; i < INPUT_SIZE; i++) {
    printf("%f ", input_tensor[i]);
  }
  printf("\n");
  for (int i = 0; i < MAX_NEURONS; i++) {
    printf("Neuron %d - State: %f, Output: %f\n", i, neurons[i].state,
           neurons[i].output);
  }
  printf("\n");
}

MemoryEntry *retrieveMemory(MemorySystem *system) {
  MemoryEntry *most_relevant = NULL;
  float highest_importance = 0.0f;

  // Search long-term memory first
  for (unsigned int i = 0; i < system->hierarchy.long_term.size; i++) {
    if (system->hierarchy.long_term.entries[i].importance >
        highest_importance) {
      highest_importance = system->hierarchy.long_term.entries[i].importance;
      most_relevant = &system->hierarchy.long_term.entries[i];
    }
  }

  // Search medium-term if nothing found in long-term
  if (!most_relevant) {
    for (unsigned int i = 0; i < system->hierarchy.medium_term.size; i++) {
      if (system->hierarchy.medium_term.entries[i].importance >
          highest_importance) {
        highest_importance =
            system->hierarchy.medium_term.entries[i].importance;
        most_relevant = &system->hierarchy.medium_term.entries[i];
      }
    }
  }

  // Search short-term if still nothing found
  if (!most_relevant) {
    for (unsigned int i = 0; i < system->hierarchy.short_term.size; i++) {
      if (system->hierarchy.short_term.entries[i].importance >
          highest_importance) {
        highest_importance = system->hierarchy.short_term.entries[i].importance;
        most_relevant = &system->hierarchy.short_term.entries[i];
      }
    }
  }

  return most_relevant;
}

// Helper function to remove decayed memories from a specific cluster
void removeDecayedFromCluster(MemoryCluster *cluster, float threshold) {
  unsigned int write_idx = 0;

  for (unsigned int read_idx = 0; read_idx < cluster->size; read_idx++) {
    if (cluster->entries[read_idx].importance >= threshold) {
      if (write_idx != read_idx) {
        cluster->entries[write_idx] = cluster->entries[read_idx];
      }
      write_idx++;
    }
  }

  cluster->size = write_idx;
}

void removeDecayedMemories(MemorySystem *system) {
  // Remove from short-term memory
  removeDecayedFromCluster(&system->hierarchy.short_term,
                           system->hierarchy.short_term.importance_threshold *
                               0.5f);

  // Remove from medium-term memory
  removeDecayedFromCluster(&system->hierarchy.medium_term,
                           system->hierarchy.medium_term.importance_threshold *
                               0.5f);

  // Remove from long-term memory
  removeDecayedFromCluster(&system->hierarchy.long_term,
                           system->hierarchy.long_term.importance_threshold *
                               0.5f);
}

void decayMemorySystem(MemorySystem *system) {
  // Decay short-term memories faster
  for (unsigned int i = 0; i < system->hierarchy.short_term.size; i++) {
    system->hierarchy.short_term.entries[i].importance *= DECAY_FACTOR * 0.9f;
  }

  // Decay medium-term memories normally
  for (unsigned int i = 0; i < system->hierarchy.medium_term.size; i++) {
    system->hierarchy.medium_term.entries[i].importance *= DECAY_FACTOR;
  }

  // Decay long-term memories slower
  for (unsigned int i = 0; i < system->hierarchy.long_term.size; i++) {
    system->hierarchy.long_term.entries[i].importance *= DECAY_FACTOR * 1.1f;
  }

  // Remove decayed memories
  removeDecayedMemories(system);
}

__m128 _mm_tanh_ps(__m128 x) {
  __m128 result = _mm_setzero_ps();
  float *px = (float *)&x;
  float *pr = (float *)&result;

  for (int i = 0; i < 4; i++) {
    float exp_pos = exp(px[i]);
    float exp_neg = exp(-px[i]);
    pr[i] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
  }

  return result;
}

void updateNeuronStates(Neuron *neurons, int num_neurons,
                        float *recurrent_weights, float scaled_factor) {
  // Process neurons in groups of 4
  for (int i = 0; i < num_neurons; i += 4) {
    // Ensure we don't overrun the array
    int remaining = num_neurons - i;
    int group_size = (remaining < 4) ? remaining : 4;

    // Load outputs, states, and weights for the current group
    __m128 current_outputs = _mm_setzero_ps();
    __m128 current_states = _mm_setzero_ps();
    __m128 current_weights = _mm_setzero_ps();

    // Load data conditionally based on group size
    float outputs[4] = {0}, states[4] = {0}, weights[4] = {0};
    for (int j = 0; j < group_size; j++) {
      outputs[j] = neurons[i + j].output;
      states[j] = neurons[i + j].state;
      weights[j] = recurrent_weights[i + j];
    }
    current_outputs = _mm_load_ps(outputs);
    current_states = _mm_load_ps(states);
    current_weights = _mm_load_ps(weights);

    // Update states with decay factor
    __m128 decay_factor = _mm_set1_ps(0.8f);
    __m128 new_states = _mm_mul_ps(current_states, decay_factor);

    // Calculate recurrent inputs
    __m128 recurrent_inputs = _mm_mul_ps(current_outputs, current_weights);

    // Simulate neighbor influence
    __m128 neighbor_influence = current_outputs;

    // Combine influences
    __m128 recurrent_factor = _mm_set1_ps(0.3f);
    __m128 neighbor_factor = _mm_set1_ps(0.2f);
    new_states =
        _mm_add_ps(new_states, _mm_mul_ps(recurrent_inputs, recurrent_factor));
    new_states =
        _mm_add_ps(new_states, _mm_mul_ps(neighbor_influence, neighbor_factor));

    // Apply activation function and scale
    __m128 scale = _mm_set1_ps(scaled_factor);
    __m128 new_outputs = _mm_tanh_ps(_mm_mul_ps(new_states, scale));

    // Store updated values back to neurons
    float result_states[4], result_outputs[4];
    _mm_storeu_ps(result_states, new_states);
    _mm_storeu_ps(result_outputs, new_outputs);

    for (int j = 0; j < group_size; j++) {
      neurons[i + j].state = result_states[j];
      neurons[i + j].output = result_outputs[j];
    }
  }
}

void initializeWeights(float *weights, int max_neurons, int max_connections,
                       float *input_tensor) {
  srand(time(NULL)); // Seed random number generator

  for (int i = 0; i < max_neurons; i++) {
    for (int j = 0; j < max_connections; j++) {
      // Random initialization in range [-0.5, 0.5]
      weights[i * max_connections + j] =
          input_tensor[(i + j) % INPUT_SIZE] * 0.5f - 0.25f;
    }
  }
}

void updateWeights(float *weights, Neuron *neurons, int *connections,
                   float learning_rate) {
  for (int i = 0; i < MAX_NEURONS; i++) {
    for (int j = 0; j < neurons[i].num_connections; j++) {
      int target_idx = connections[i * MAX_CONNECTIONS + j];
      // Modified Hebbian learning rule with normalization
      float pre_activation = neurons[i].state;
      float post_activation = neurons[target_idx].output;
      float current_weight = weights[i * MAX_CONNECTIONS + j];
      // Calculate weight update with normalization term
      float delta_w = learning_rate *
                      (pre_activation * post_activation - // Hebbian term
                       current_weight * 0.1f // Weight decay for normalization
                      );
      // Update weight
      weights[i * MAX_CONNECTIONS + j] += delta_w;
      // Clip weights to prevent unbounded growth
      weights[i * MAX_CONNECTIONS + j] =
          fmaxf(-1.0f, fminf(1.0f, weights[i * MAX_CONNECTIONS + j]));
    }
  }
}

void processNeurons(Neuron *neurons, int num_neurons, float *weights,
                    int *connections, int max_connections,
                    float scaled_factor) {
  // Process neurons in groups of 4
  for (int i = 0; i < num_neurons; i += 4) {
    // Ensure we don't overrun the array
    int remaining = num_neurons - i;
    int group_size = (remaining < 4) ? remaining : 4;

    __m128 weighted_sum = _mm_setzero_ps();

    // Compute weighted sum for connections
    for (int j = 0; j < max_connections; j++) {
      float weight_array[4] = {0};
      float target_state_array[4] = {0};

      for (int k = 0; k < group_size; k++) {
        int neuron_idx = i + k;
        int connection_idx = connections[neuron_idx * max_connections + j];
        weight_array[k] = weights[neuron_idx * max_connections + j];
        target_state_array[k] = neurons[connection_idx].state;
      }

      __m128 weight_vector = _mm_load_ps(weight_array);
      __m128 target_state = _mm_load_ps(target_state_array);

      weighted_sum =
          _mm_add_ps(weighted_sum, _mm_mul_ps(weight_vector, target_state));
    }

    // Load current states
    float current_state_array[4] = {0};
    for (int k = 0; k < group_size; k++) {
      current_state_array[k] = neurons[i + k].state;
    }
    __m128 current_states = _mm_load_ps(current_state_array);

    // Combine with decay factor
    __m128 decay_factor = _mm_set1_ps(0.8f);
    __m128 weighted_factor = _mm_set1_ps(0.2f);
    __m128 new_states = _mm_add_ps(_mm_mul_ps(current_states, decay_factor),
                                   _mm_mul_ps(weighted_sum, weighted_factor));

    // Store and update neuron states
    float result_states[4];
    _mm_storeu_ps(result_states, new_states);

    for (int k = 0; k < group_size; k++) {
      neurons[i + k].state = result_states[k];
      neurons[i + k].output = tanh(result_states[k] * scaled_factor);
    }
  }
}
// Function to measure execution time
double getCurrentTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Function to calculate performance score
float calculatePerformanceScore(PerformanceMetrics metrics) {
  // Combine multiple factors into a single score
  float time_score = 1.0f / (float)metrics.execution_time;
  float output_score = metrics.average_output;
  float error_penalty = 1.0f - metrics.error_rate;

  // Weight different factors
  return (time_score * 0.4f + output_score * 0.4f + error_penalty * 0.2f);
}

// Function to compute average output
float computeAverageOutput(Neuron *neurons) {
  float sum = 0.0f;
  for (int i = 0; i < MAX_NEURONS; i++) {
    sum += fabsf(neurons[i].output);
  }
  return sum / MAX_NEURONS;
}

// Function to compute error rate
float computeErrorRate(Neuron *neurons, float *previous_outputs) {
  float error_sum = 0.0f;
  for (int i = 0; i < MAX_NEURONS; i++) {
    error_sum += fabsf(neurons[i].output - previous_outputs[i]);
  }
  return error_sum / MAX_NEURONS;
}

// Function to optimize parameters based on performance history
void optimizeParameters(OptimizationState *opt_state,
                        PerformanceMetrics *history, int history_size) {
  float current_avg_score = 0.0f;
  for (int i = 0; i < history_size; i++) {
    current_avg_score += calculatePerformanceScore(history[i]);
  }
  current_avg_score /= history_size;

  // If current performance is better than best, update optimal parameters
  if (current_avg_score > opt_state->best_performance_score) {
    opt_state->best_performance_score = current_avg_score;
    opt_state->optimal_batch_size = history[history_size - 1].batch_size;
    opt_state->optimal_learning_rate = history[history_size - 1].learning_rate;
    opt_state->best_execution_time = history[history_size - 1].execution_time;
  }

  // Adjust parameters based on performance trends
  if (history_size >= OPTIMIZATION_WINDOW) {
    float trend = 0.0f;
    for (int i = 1; i < history_size; i++) {
      trend += calculatePerformanceScore(history[i]) -
               calculatePerformanceScore(history[i - 1]);
    }

    // If performance is declining, try different parameters
    if (trend < 0) {
      // Adjust batch size
      int new_batch_size = history[history_size - 1].batch_size;
      if (trend < -0.1) {
        new_batch_size = (new_batch_size % MAX_BATCH_SIZE) + 1;
      }

      // Adjust learning rate
      float new_learning_rate = history[history_size - 1].learning_rate;
      if (trend < -0.05) {
        new_learning_rate *= (rand() / (float)RAND_MAX) * 0.5f + 0.75f;
      }

      // Update history with new parameters
      history[history_size - 1].batch_size = new_batch_size;
      history[history_size - 1].learning_rate = new_learning_rate;
    }
  }
}

// Function to update adaptation parameters based on network state
void updateDynamicParameters(DynamicParameters *params, float performance_delta,
                             float stability_measure, float error_rate) {
  // Adjust noise scales based on network performance
  if (performance_delta < 0) {
    // Reduce noise when performance degrades
    params->input_noise_scale *= 0.95f;
    params->weight_noise_scale *= 0.95f;
  } else {
    // Gradually increase noise tolerance
    params->input_noise_scale *= 1.02f;
    params->weight_noise_scale *= 1.01f;
  }

  // Update adaptation rate using momentum
  float target_rate =
      params->base_adaptation_rate * (1.0f + fabsf(performance_delta));
  params->current_adaptation_rate =
      params->learning_momentum * params->current_adaptation_rate +
      (1.0f - params->learning_momentum) * target_rate;

  // Adjust plasticity based on stability
  if (stability_measure > params->stability_threshold) {
    params->plasticity *= 0.98f; // Reduce plasticity when stable
  } else {
    params->plasticity *= 1.02f; // Increase plasticity when unstable
  }

  // Update noise tolerance based on error rate
  params->noise_tolerance =
      fmaxf(0.1f, params->noise_tolerance * (1.0f - error_rate));

  // Adjust recovery rate based on performance
  params->recovery_rate =
      fmaxf(0.01f, params->recovery_rate * (1.0f + performance_delta));

  // Update homeostatic factor
  params->homeostatic_factor =
      fminf(0.2f, params->homeostatic_factor * (1.0f + stability_measure));

  // Clamp parameters to reasonable ranges
  params->input_noise_scale =
      fmaxf(0.01f, fminf(0.3f, params->input_noise_scale));
  params->weight_noise_scale =
      fmaxf(0.01f, fminf(0.2f, params->weight_noise_scale));
  params->current_adaptation_rate =
      fmaxf(0.001f, fminf(0.1f, params->current_adaptation_rate));
  params->plasticity = fmaxf(0.1f, fminf(2.0f, params->plasticity));
}

// Enhanced network adaptation function with dynamic parameters
void adaptNetworkDynamic(Neuron *neurons, float *weights,
                         DynamicParameters *params, float performance_delta,
                         float *input_tensor) {

  // Calculate adaptation strength based on current parameters
  float adaptation_strength =
      params->current_adaptation_rate * params->plasticity;

  // Track average neuron activity for homeostasis
  float avg_activity = 0.0f;
  for (int i = 0; i < MAX_NEURONS; i++) {
    avg_activity += fabsf(neurons[i].output);
  }
  avg_activity /= MAX_NEURONS;

  // Adapt each neuron individually
  for (int i = 0; i < MAX_NEURONS; i++) {
    // Calculate homeostatic scaling factor
    float homeostatic_scale =
        params->homeostatic_factor * (1.0f - neurons[i].output / avg_activity);

    // Adjust neuron sensitivity with homeostasis
    neurons[i].state *=
        (1.0f + adaptation_strength * (performance_delta + homeostatic_scale));

    // Apply noise tolerance to state
    neurons[i].state += params->noise_tolerance *
                        (input_tensor[i % INPUT_SIZE] - neurons[i].state);

    // Update weights with dynamic adaptation
    for (int j = 0; j < neurons[i].num_connections; j++) {
      int weight_idx = i * MAX_CONNECTIONS + j;

      // Calculate weight update with recovery term
      float weight_update =
          adaptation_strength *
          (performance_delta +
           params->recovery_rate * (weights[weight_idx] - neurons[i].state));

      // Apply weight update with momentum
      weights[weight_idx] = params->learning_momentum * weights[weight_idx] +
                            (1.0f - params->learning_momentum) * weight_update;

      // Ensure weights remain bounded
      weights[weight_idx] = fmaxf(-1.0f, fminf(1.0f, weights[weight_idx]));
    }
  }
}

// Function to measure network stability
float measureNetworkStability(Neuron *neurons, float *previous_states) {
  float stability_measure = 0.0f;
  for (int i = 0; i < MAX_NEURONS; i++) {
    stability_measure += fabsf(neurons[i].state - previous_states[i]);
  }
  return 1.0f - (stability_measure / MAX_NEURONS);
}

void saveSystemParameters(const SystemParameters *params,
                          const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Error opening parameters file for writing\n");
    return;
  }

  fwrite(params, sizeof(SystemParameters), 1, fp);
  fclose(fp);
}

SystemParameters *loadSystemParameters(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("No existing parameters found. Creating new ones...\n");
    SystemParameters *params =
        (SystemParameters *)malloc(sizeof(SystemParameters));

    // Initialize with default values
    params->opt_state.optimal_batch_size = 1;
    params->opt_state.optimal_learning_rate = 0.01f;
    params->opt_state.best_execution_time = INFINITY;
    params->opt_state.best_performance_score = -INFINITY;

    params->dynamic_params = initDynamicParameters();
    params->best_performance_score = -INFINITY;
    params->best_stability_measure = 0.0f;
    params->timestamp = time(NULL);

    return params;
  }

  SystemParameters *params =
      (SystemParameters *)malloc(sizeof(SystemParameters));
  fread(params, sizeof(SystemParameters), 1, fp);
  fclose(fp);

  printf("Loaded existing parameters:\n");
  printf("Optimal batch size: %d\n", params->opt_state.optimal_batch_size);
  printf("Optimal learning rate: %.6f\n",
         params->opt_state.optimal_learning_rate);
  printf("Best performance score: %.6f\n", params->best_performance_score);
  printf("Current adaptation rate: %.6f\n",
         params->dynamic_params.current_adaptation_rate);
  printf("Input noise scale: %.6f\n", params->dynamic_params.input_noise_scale);
  printf("Plasticity: %.6f\n", params->dynamic_params.plasticity);

  return params;
}

// Structure for analysis results
typedef struct {
  float avg_execution_time;
  float avg_average_output;
  float avg_error_rate;

  float var_execution_time;
  float var_average_output;
  float var_error_rate;

  float min_execution_time;
  float max_execution_time;

  float min_average_output;
  float max_average_output;

  float min_error_rate;
  float max_error_rate;
} PerformanceAnalysis;

PerformanceAnalysis analyzeNetworkPerformance(const PerformanceMetrics *metrics,
                                              int steps) {
  PerformanceAnalysis analysis = {0};

  if (steps <= 0) {
    fprintf(stderr, "Error: No steps provided for analysis.\n");
    return analysis;
  }

  float sum_execution_time = 0, sum_average_output = 0, sum_error_rate = 0;
  float sum_execution_time_sq = 0, sum_average_output_sq = 0,
        sum_error_rate_sq = 0;

  analysis.min_execution_time = FLT_MAX;
  analysis.max_execution_time = FLT_MIN;

  analysis.min_average_output = FLT_MAX;
  analysis.max_average_output = FLT_MIN;

  analysis.min_error_rate = FLT_MAX;
  analysis.max_error_rate = FLT_MIN;

  for (int i = 0; i < steps; i++) {
    float execution_time = metrics[i].execution_time;
    float average_output = metrics[i].average_output;
    float error_rate = metrics[i].error_rate;

    // Sum values
    sum_execution_time += execution_time;
    sum_average_output += average_output;
    sum_error_rate += error_rate;

    // Sum squares for variance
    sum_execution_time_sq += execution_time * execution_time;
    sum_average_output_sq += average_output * average_output;
    sum_error_rate_sq += error_rate * error_rate;

    // Min/Max calculations
    if (execution_time < analysis.min_execution_time)
      analysis.min_execution_time = execution_time;
    if (execution_time > analysis.max_execution_time)
      analysis.max_execution_time = execution_time;

    if (average_output < analysis.min_average_output)
      analysis.min_average_output = average_output;
    if (average_output > analysis.max_average_output)
      analysis.max_average_output = average_output;

    if (error_rate < analysis.min_error_rate)
      analysis.min_error_rate = error_rate;
    if (error_rate > analysis.max_error_rate)
      analysis.max_error_rate = error_rate;
  }

  // Calculate averages
  analysis.avg_execution_time = sum_execution_time / steps;
  analysis.avg_average_output = sum_average_output / steps;
  analysis.avg_error_rate = sum_error_rate / steps;

  // Calculate variances
  analysis.var_execution_time =
      (sum_execution_time_sq / steps) -
      (analysis.avg_execution_time * analysis.avg_execution_time);
  analysis.var_average_output =
      (sum_average_output_sq / steps) -
      (analysis.avg_average_output * analysis.avg_average_output);
  analysis.var_error_rate = (sum_error_rate_sq / steps) -
                            (analysis.avg_error_rate * analysis.avg_error_rate);

  return analysis;
}

void generatePerformanceGraph(const PerformanceMetrics *metrics, int steps) {
  printf("\nGenerating performance graph using gnuplot...\n");

  // Temporary file to store data
  const char *dataFileName = "performance_data.dat";
  FILE *dataFile = fopen(dataFileName, "w");
  if (!dataFile) {
    fprintf(stderr, "Error: Unable to create data file for gnuplot.\n");
    return;
  }

  // Write data to file
  for (int i = 0; i < steps; i++) {
    fprintf(dataFile, "%d %.6f %.6f %.6f\n", i, metrics[i].execution_time,
            metrics[i].average_output, metrics[i].error_rate);
  }
  fclose(dataFile);

  // Open a pipe to gnuplot
  FILE *gnuplotPipe = popen("gnuplot", "w");
  if (!gnuplotPipe) {
    fprintf(stderr, "Error: Unable to open gnuplot.\n");
    return;
  }

  // Send gnuplot commands
  fprintf(gnuplotPipe, "set terminal png size 800,600\n");
  fprintf(gnuplotPipe, "set output 'performance_graph.png'\n");
  fprintf(gnuplotPipe, "set title 'Performance Metrics Over Steps'\n");
  fprintf(gnuplotPipe, "set xlabel 'Steps'\n");
  fprintf(gnuplotPipe, "set ylabel 'Metrics'\n");
  fprintf(gnuplotPipe, "set key outside\n");
  fprintf(gnuplotPipe,
          "plot '%s' using 1:2 with lines title 'Execution Time', \\\n",
          dataFileName);
  fprintf(gnuplotPipe,
          "     '%s' using 1:3 with lines title 'Average Output', \\\n",
          dataFileName);
  fprintf(gnuplotPipe, "     '%s' using 1:4 with lines title 'Error Rate'\n",
          dataFileName);

  pclose(gnuplotPipe);

  printf("Graph generated and saved as 'performance_graph.png'.\n");
}

float computeMemoryVectorSimilarity(float *vector1, float *vector2) {
  float dot_product = 0.0f;
  float magnitude1 = 0.0f;
  float magnitude2 = 0.0f;

  for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
    dot_product += vector1[i] * vector2[i];
    magnitude1 += vector1[i] * vector1[i];
    magnitude2 += vector2[i] * vector2[i];
  }

  magnitude1 = sqrt(magnitude1);
  magnitude2 = sqrt(magnitude2);

  if (magnitude1 == 0.0f || magnitude2 == 0.0f) {
    return 0.0f;
  }

  return dot_product / (magnitude1 * magnitude2);
}

// Initialize pattern matching parameters
PatternMatchingParams initPatternMatchingParams() {
  PatternMatchingParams params = {.similarity_threshold = 0.8f,
                                  .temporal_window = 5,
                                  .temporal_decay = 0.9f,
                                  .max_matches = 10};
  return params;
}

// Find similar memories within a cluster
PatternMatch *findSimilarMemoriesInCluster(MemoryCluster *cluster,
                                           float *target_vector,
                                           float similarity_threshold,
                                           int *num_matches) {
  PatternMatch *matches = malloc(cluster->size * sizeof(PatternMatch));
  *num_matches = 0;

  for (unsigned int i = 0; i < cluster->size; i++) {
    float similarity = computeMemoryVectorSimilarity(
        target_vector, cluster->entries[i].vector);

    if (similarity >= similarity_threshold) {
      matches[*num_matches].index = i;
      matches[*num_matches].similarity = similarity;
      matches[*num_matches].timestamp = cluster->entries[i].timestamp;
      (*num_matches)++;
    }
  }

  return matches;
}

// Function to print pattern matching results
void printPatternMatches(MemorySystem *system, PatternMatch *matches,
                         int num_matches) {
  printf("\nPattern Matching Results:\n");
  printf("Found %d matches:\n", num_matches);

  for (int i = 0; i < num_matches; i++) {
    printf("Match %d:\n", i + 1);
    printf("  Index: %d\n", matches[i].index);
    printf("  Similarity: %.4f\n", matches[i].similarity);
    printf("  Timestamp: %u\n", matches[i].timestamp);
    printf("  Importance: %.4f\n",
           system->entries[matches[i].index].importance);
    printf("\n");
  }
}

void setNeuronState(Neuron *neurons, int neuron_index, float new_state) {
  if (neuron_index < 0 || neuron_index >= MAX_NEURONS) {
    printf("Error: Invalid neuron index %d. Must be between 0 and %d\n",
           neuron_index, MAX_NEURONS - 1);
    return;
  }
  neurons[neuron_index].state = new_state;
  // Update output based on new state
  neurons[neuron_index].output = tanh(new_state * 1.5f + 0.1f);
}

void setNeuronOutput(Neuron *neurons, int neuron_index, float new_output) {
  if (neuron_index < 0 || neuron_index >= MAX_NEURONS) {
    printf("Error: Invalid neuron index %d. Must be between 0 and %d\n",
           neuron_index, MAX_NEURONS - 1);
    return;
  }
  neurons[neuron_index].output = new_output;
}

void setWeight(float *weights, int source_neuron, int connection_index,
               float new_weight) {
  if (source_neuron < 0 || source_neuron >= MAX_NEURONS) {
    printf("Error: Invalid source neuron index %d\n", source_neuron);
    return;
  }
  if (connection_index < 0 || connection_index >= MAX_CONNECTIONS) {
    printf("Error: Invalid connection index %d\n", connection_index);
    return;
  }

  weights[source_neuron * MAX_CONNECTIONS + connection_index] = new_weight;
}

void setNeuronWeights(float *weights, int neuron_index, float *new_weights) {
  if (neuron_index < 0 || neuron_index >= MAX_NEURONS) {
    printf("Error: Invalid neuron index %d\n", neuron_index);
    return;
  }

  for (int i = 0; i < MAX_CONNECTIONS; i++) {
    weights[neuron_index * MAX_CONNECTIONS + i] = new_weights[i];
  }
}

void scaleNeuronWeights(float *weights, int neuron_index, float scale_factor) {
  if (neuron_index < 0 || neuron_index >= MAX_NEURONS) {
    printf("Error: Invalid neuron index %d\n", neuron_index);
    return;
  }

  for (int i = 0; i < MAX_CONNECTIONS; i++) {
    weights[neuron_index * MAX_CONNECTIONS + i] *= scale_factor;
  }
}

void resetNeuron(Neuron *neurons, int neuron_index) {
  if (neuron_index < 0 || neuron_index >= MAX_NEURONS) {
    printf("Error: Invalid neuron index %d\n", neuron_index);
    return;
  }

  neurons[neuron_index].state = 0.0f;
  neurons[neuron_index].output = 0.0f;
  neurons[neuron_index].num_connections = MAX_CONNECTIONS;
  neurons[neuron_index].layer_id = neuron_index % 2;
}

float getNeuronState(Neuron *neurons, int neuron_index) {
  if (neuron_index < 0 || neuron_index >= MAX_NEURONS) {
    printf("Error: Invalid neuron index %d\n", neuron_index);
    return 0.0f;
  }
  return neurons[neuron_index].state;
}

float getNeuronOutput(Neuron *neurons, int neuron_index) {
  if (neuron_index < 0 || neuron_index >= MAX_NEURONS) {
    printf("Error: Invalid neuron index %d\n", neuron_index);
    return 0.0f;
  }
  return neurons[neuron_index].output;
}

float getWeight(float *weights, int source_neuron, int connection_index) {
  if (source_neuron < 0 || source_neuron >= MAX_NEURONS) {
    printf("Error: Invalid source neuron index %d\n", source_neuron);
    return 0.0f;
  }
  if (connection_index < 0 || connection_index >= MAX_CONNECTIONS) {
    printf("Error: Invalid connection index %d\n", connection_index);
    return 0.0f;
  }

  return weights[source_neuron * MAX_CONNECTIONS + connection_index];
}

void printNeuronDetails(Neuron *neurons, float *weights, int *connections,
                        int neuron_index) {
  if (neuron_index < 0 || neuron_index >= MAX_NEURONS) {
    printf("Error: Invalid neuron index %d\n", neuron_index);
    return;
  }

  printf("\nNeuron %d Details:\n", neuron_index);
  printf("State: %f\n", neurons[neuron_index].state);
  printf("Output: %f\n", neurons[neuron_index].output);
  printf("Layer ID: %u\n", neurons[neuron_index].layer_id);
  printf("Number of connections: %u\n", neurons[neuron_index].num_connections);

  printf("Connections and weights:\n");
  for (int i = 0; i < MAX_CONNECTIONS; i++) {
    printf("  Connection %d -> Neuron %u (Weight: %f)\n", i,
           connections[neuron_index * MAX_CONNECTIONS + i],
           weights[neuron_index * MAX_CONNECTIONS + i]);
  }
}

void mergeMemoryVectors(MemoryEntry *target, MemoryEntry *source) {
  // Weight vectors by their importance
  float total_importance = target->importance + source->importance;
  float target_weight = target->importance / total_importance;
  float source_weight = source->importance / total_importance;

  // Merge vectors
  for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
    target->vector[i] =
        target->vector[i] * target_weight + source->vector[i] * source_weight;
  }

  // Update importance and timestamp
  target->importance = fmaxf(target->importance, source->importance) * 1.1f;
  target->timestamp = (target->timestamp > source->timestamp)
                          ? target->timestamp
                          : source->timestamp;
}

void mergeSimilarInCluster(MemoryCluster *cluster, float similarity_threshold) {
  for (unsigned int i = 0; i < cluster->size; i++) {
    for (unsigned int j = i + 1; j < cluster->size; j++) {
      float similarity = computeMemoryVectorSimilarity(
          cluster->entries[i].vector, cluster->entries[j].vector);

      if (similarity >= similarity_threshold) {
        // Merge memories and update importance
        mergeMemoryVectors(&cluster->entries[i], &cluster->entries[j]);

        // Remove the second memory by shifting remaining memories
        for (unsigned int k = j; k < cluster->size - 1; k++) {
          cluster->entries[k] = cluster->entries[k + 1];
        }
        cluster->size--;
        j--;
      }
    }
  }
}

void mergeSimilarMemories(MemorySystem *system) {
  mergeSimilarInCluster(&system->hierarchy.medium_term, 0.9f);
  mergeSimilarInCluster(&system->hierarchy.long_term, 0.95f);
}

void saveHierarchicalMemory(MemorySystem *system, const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Error opening file for writing hierarchical memory\n");
    return;
  }

  // Write structure sizes
  fwrite(&system->hierarchy.short_term.size, sizeof(unsigned int), 1, fp);
  fwrite(&system->hierarchy.medium_term.size, sizeof(unsigned int), 1, fp);
  fwrite(&system->hierarchy.long_term.size, sizeof(unsigned int), 1, fp);

  // Write memory entries for each level
  fwrite(system->hierarchy.short_term.entries, sizeof(MemoryEntry),
         system->hierarchy.short_term.size, fp);
  fwrite(system->hierarchy.medium_term.entries, sizeof(MemoryEntry),
         system->hierarchy.medium_term.size, fp);
  fwrite(system->hierarchy.long_term.entries, sizeof(MemoryEntry),
         system->hierarchy.long_term.size, fp);

  fclose(fp);
}

float computeMSELoss(Neuron *neurons, float *target_outputs, int num_neurons) {
  float loss = 0.0f;

  // Loop through all neurons
  for (int i = 0; i < num_neurons; i++) {
    float output = neurons[i].output;        // Neuron's output
    float target_output = target_outputs[i]; // Target output

    // Calculate the squared error and add it to the loss
    float error = output - target_output;
    loss += error * error;
  }

  // Average the loss over all neurons
  loss /= num_neurons;
  return loss;
}

void verifyNetworkState(const Neuron *neurons, TaskPrompt *prompt) {
  prompt->verifications[0] =
      (PromptVerification){.instruction = "Verify neuron activation patterns",
                           .confidence = 0.0f,
                           .verified = false,
                           .reasoning = ""};

  float activation_sum = 0.0f;
  for (int i = 0; i < MAX_NEURONS; i++) {
    activation_sum += neurons[i].output;
  }

  float avg_activation = activation_sum / MAX_NEURONS;
  sprintf(prompt->verifications[0].reasoning,
          "Average activation: %.4f - Pattern stability: %.2f%%",
          avg_activation, (avg_activation > 0.2f) ? 100.0f : 50.0f);
  prompt->verifications[0].confidence = avg_activation;
  prompt->verifications[0].verified = avg_activation > 0.2f;
}

void generateTaskPrompt(TaskPrompt *prompt, int step) {
  sprintf(prompt->task_description,
          "Step %d: Process neural network state with memory integration",
          step);
  prompt->expected_outcome = 0.8f;
  strcpy(prompt->success_criteria,
         "Stable activation patterns with memory recall");
}

float assessMemoryCoherence(const MemoryEntry *memory,
                            const Neuron *currentNeurons) {
  float coherence_score = 0.0f;
  int vector_size =
      MAX_NEURONS * 2; // Size accounts for both state and output values

  // Compare memory vector with current neuron states
  for (int i = 0; i < MAX_NEURONS; i++) {
    // Compare state
    float state_diff = fabs(memory->vector[i] - currentNeurons[i].state);
    coherence_score += (1.0f - state_diff);

    // Compare output
    float output_diff =
        fabs(memory->vector[i + MAX_NEURONS] - currentNeurons[i].output);
    coherence_score += (1.0f - output_diff);
  }

  // Normalize coherence score to [0,1] range
  coherence_score /= (float)(vector_size);

  return coherence_score;
}

float mapWordToValue(const char *word) {
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(word, vocabulary[i].word) == 0) {
      // Map word index to a value in [0, 1), adjusted by semantic weight
      float base_value = (float)i / vocab_size;
      return base_value * vocabulary[i].semantic_weight;
    }
  }
  return 0.0f; // Default value if word is not in vocabulary
}

// Function to transform neuron outputs into text
void transformOutputsToText(float *outputs, int size, char *outputText,
                            int textSize) {
  char buffer[256];
  outputText[0] = '\0'; // Initialize the output text

  for (int i = 0; i < size; i++) {
    const char *word = mapToWord(outputs[i]);
    snprintf(buffer, sizeof(buffer), "%s ", word);
    strncat(outputText, buffer, textSize - strlen(outputText) - 1);
  }
}

NetworkPerformanceMetrics *initializePerformanceMetrics(int num_regions) {
  NetworkPerformanceMetrics *metrics =
      malloc(sizeof(NetworkPerformanceMetrics));
  metrics->num_regions = num_regions;
  metrics->region_performance_scores = calloc(num_regions, sizeof(float));
  metrics->region_error_rates = calloc(num_regions, sizeof(float));
  metrics->region_output_variance = calloc(num_regions, sizeof(float));
  return metrics;
}

void computeRegionPerformanceMetrics(NetworkPerformanceMetrics *metrics,
                                     Neuron *neurons, float *target_outputs,
                                     int max_neurons) {
  int region_size = max_neurons / metrics->num_regions;

  for (int region = 0; region < metrics->num_regions; region++) {
    int start = region * region_size;
    int end = start + region_size;
    float total_error = 0.0f;
    float mean_output = 0.0f;
    float variance_sum = 0.0f;

    // Compute mean output for the region
    for (int i = start; i < end; i++) {
      mean_output += neurons[i].output;
    }
    mean_output /= region_size;

    // Compute error and variance
    for (int i = start; i < end; i++) {
      float error = fabs(neurons[i].output - target_outputs[i]);
      total_error += error;
      variance_sum += pow(neurons[i].output - mean_output, 2);
    }

    metrics->region_error_rates[region] = total_error / region_size;
    metrics->region_output_variance[region] = variance_sum / region_size;
    metrics->region_performance_scores[region] = 1.0f / (1.0f + total_error);
  }
}

MetaController *initializeMetaController(int num_regions) {
  MetaController *controller = malloc(sizeof(MetaController));
  controller->meta_learning_rate = 0.01;
  controller->exploration_factor = 0.1;
  controller->num_regions = num_regions;
  controller->region_importance_scores = calloc(num_regions, sizeof(float));
  controller->learning_efficiency_history = calloc(num_regions, sizeof(float));

  // Initialize with equal importance
  for (int i = 0; i < num_regions; i++) {
    controller->region_importance_scores[i] = 1.0f / num_regions;
  }

  return controller;
}

void updateMetaControllerPriorities(MetaController *controller,
                                    NetworkPerformanceMetrics *performance) {
  for (int i = 0; i < controller->num_regions; i++) {
    // Compute learning delta
    float learning_delta = performance->region_performance_scores[i] -
                           controller->learning_efficiency_history[i];

    // Adaptive importance calculation
    controller->region_importance_scores[i] +=
        controller->meta_learning_rate * learning_delta *
        (1 + controller->exploration_factor);

    // Normalize importance
    controller->region_importance_scores[i] =
        fmin(fmax(controller->region_importance_scores[i], 0), 1);

    // Update efficiency history
    controller->learning_efficiency_history[i] =
        performance->region_performance_scores[i];
  }
}

void applyMetaControllerAdaptations(Neuron *neurons, float *weights,
                                    MetaController *controller,
                                    int max_neurons) {
  int region_size = max_neurons / controller->num_regions;

  for (int region = 0; region < controller->num_regions; region++) {
    int start = region * region_size;
    int end = start + region_size;
    float region_importance = controller->region_importance_scores[region];

    // Adjust connection weights based on region importance
    for (int i = start; i < end; i++) {
      for (int j = 0; j < neurons[i].num_connections; j++) {
        int connection_idx = i * MAX_CONNECTIONS + j;
        // Modulate weights non-linearly with region importance
        weights[connection_idx] *= (1 + region_importance);
      }
    }
  }
}

void printReplayStatistics(MemorySystem *memorySystem) {
  if (memorySystem == NULL || memorySystem->size == 0) {
    printf("No memories available for replay.\n");
    return;
  }

  float total_importance = 0.0f;
  int num_replayed = 0;

  // Iterate through memory entries to calculate statistics
  for (int i = 0; i < memorySystem->size; i++) {
    MemoryEntry *entry = &memorySystem->entries[i];
    if (entry->importance > 0.5f) { // Only consider important memories
      total_importance += entry->importance;
      num_replayed++;
    }
  }

  // Calculate averages
  float avg_importance =
      (num_replayed > 0) ? total_importance / num_replayed : 0.0f;

  // Print statistics
  printf("\nMemory Replay Statistics:\n");
  printf("Total Memories: %u\n", memorySystem->size);
  printf("Memories Replayed: %d\n", num_replayed);
  printf("Average Importance: %.2f\n", avg_importance);
}

void normalizeWeights(float *weights, int num_weights) {
  float max_weight = 0.0f;
  for (int i = 0; i < num_weights; i++) {
    if (fabs(weights[i]) > max_weight) {
      max_weight = fabs(weights[i]);
    }
  }

  if (max_weight > 1.0f) {
    for (int i = 0; i < num_weights; i++) {
      weights[i] /= max_weight;
    }
  }
}

void updateBidirectionalWeights(float *forward_weights, float *reverse_weights,
                                Neuron *neurons, int *forward_connections,
                                int *reverse_connections, float learning_rate) {
  // Update forward weights
  for (int i = 0; i < MAX_NEURONS; i++) {
    for (int j = 0; j < MAX_CONNECTIONS; j++) {
      int conn_idx = i * MAX_CONNECTIONS + j;
      int target_neuron = forward_connections[conn_idx];

      // Calculate weight update for forward pathway
      float forward_delta =
          learning_rate * neurons[i].output * neurons[target_neuron].state;
      forward_weights[conn_idx] += forward_delta;

      // Apply weight decay or regularization if needed
      forward_weights[conn_idx] *= (1.0f - WEIGHT_DECAY);
    }
  }

  // Update reverse weights
  for (int i = 0; i < MAX_NEURONS; i++) {
    for (int j = 0; j < MAX_CONNECTIONS; j++) {
      int conn_idx = i * MAX_CONNECTIONS + j;
      int source_neuron = reverse_connections[conn_idx];

      // Calculate weight update for reverse pathway
      float reverse_delta =
          learning_rate * neurons[i].state * neurons[source_neuron].output;
      reverse_weights[conn_idx] += reverse_delta;

      // Apply weight decay or regularization if needed
      reverse_weights[conn_idx] *= (1.0f - WEIGHT_DECAY);
    }
  }

  // Optional: Normalize weights to prevent explosion
  normalizeWeights(forward_weights, MAX_NEURONS * MAX_CONNECTIONS);
  normalizeWeights(reverse_weights, MAX_NEURONS * MAX_CONNECTIONS);
}

void generatePredictiveInputs(float *input_tensor,
                              NetworkStateSnapshot *previous_states,
                              int max_neurons) {
  for (int i = 0; i < max_neurons; i++) {
    // Use previous states with some variation
    if (previous_states) {
      // Linear prediction with some noise
      input_tensor[i] = previous_states->states[i] *
                        (1.0f + (rand() / (float)RAND_MAX - 0.5f) * 0.2f);
    }
  }
}

void computePredictionErrors(Neuron *neurons, float *actual_inputs,
                             int max_neurons) {
  for (int i = 0; i < max_neurons; i++) {
    // Compute prediction based on current neuron output
    float prediction =
        neurons[i].output * predictive_params[i].prediction_weight;

    // Calculate prediction error
    predictive_params[i].prediction_error = actual_inputs[i] - prediction;

    // Adaptive weight update
    predictive_params[i].prediction_weight +=
        0.01f * predictive_params[i].prediction_error * neurons[i].output;
  }
}

void updateNeuronsWithPredictiveCoding(Neuron *neurons, float *actual_inputs,
                                       int max_neurons, float learning_rate) {
  for (int i = 0; i < max_neurons; i++) {
    // Adjust neuron state based on prediction error
    neurons[i].state += learning_rate * predictive_params[i].prediction_error *
                        predictive_params[i].adaptation_rate;

    // Soft bounds to prevent state explosion
    neurons[i].state = fminf(fmaxf(neurons[i].state, -1.0f), 1.0f);
  }
}

void initPredictiveCodingParams(int max_neurons) {
  for (int i = 0; i < max_neurons; i++) {
    predictive_params[i].prediction_weight = 1.0f;
    predictive_params[i].prediction_error = 0.0f;
    predictive_params[i].adaptation_rate = 0.5f;
  }
}

void removeUnderperformingNeuron(Neuron *neurons, int *connections,
                                 float *weights, int *num_neurons,
                                 int max_neurons, int neuron_index) {
  if (*num_neurons <= 1 || neuron_index >= *num_neurons)
    return;

  // Shift neurons after the removed neuron
  for (int i = neuron_index; i < *num_neurons - 1; i++) {
    neurons[i] = neurons[i + 1];
  }

  // Update connections and weights
  for (int i = 0; i < *num_neurons; i++) {
    for (int j = 0; j < MAX_CONNECTIONS; j++) {
      if (connections[i * MAX_CONNECTIONS + j] == neuron_index) {
        // Remove this connection
        connections[i * MAX_CONNECTIONS + j] = 0;
        weights[i * MAX_CONNECTIONS + j] = 0.0f;
      } else if (connections[i * MAX_CONNECTIONS + j] > neuron_index) {
        // Adjust connections to neurons after the removed neuron
        connections[i * MAX_CONNECTIONS + j]--;
      }
    }
  }

  // Decrease neuron count
  (*num_neurons)--;
}

void addNewNeuron(Neuron *neurons, int *connections, float *weights,
                  int *num_neurons, int max_neurons, float *input_tensor) {
  if (*num_neurons >= max_neurons)
    return;

  // Create a new neuron with default initialization
  Neuron new_neuron = {
      .state = 0.0f,
      .output = 0.0f,
      .num_connections = 2,
      .layer_id = (*num_neurons) % 2 // Alternate layers
  };

  // Add the new neuron
  neurons[*num_neurons] = new_neuron;

  // Initialize connections for the new neuron
  int base_index = *num_neurons * MAX_CONNECTIONS;
  connections[base_index] =
      (*num_neurons + 1) % *num_neurons; // Forward connection
  connections[base_index + 1] =
      (*num_neurons - 1 + *num_neurons) % *num_neurons; // Backward connection

  // Initialize weights for the new neuron
  weights[base_index] = 0.6f;      // Forward weight
  weights[base_index + 1] = -0.4f; // Backward weight

  // Slightly perturb the input tensor to help integration
  input_tensor[*num_neurons] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);

  // Increase neuron count
  (*num_neurons)++;
}

void computeNeuronPerformanceMetrics(Neuron *neurons, float *target_outputs,
                                     int *connections, float *weights,
                                     NeuronPerformanceMetric *metrics,
                                     int num_neurons,
                                     NetworkStateSnapshot *stateHistory,
                                     int current_step) {
  for (int i = 0; i < num_neurons; i++) {
    // Output Stability: Variance of neuron's output over recent steps
    float output_variance = 0.0f;
    if (current_step > 5) {
      float mean_output = 0.0f;
      for (int j = 1; j <= 5; j++) {
        mean_output += stateHistory[current_step - j].states[i];
      }
      mean_output /= 5.0f;

      for (int j = 1; j <= 5; j++) {
        float diff = stateHistory[current_step - j].states[i] - mean_output;
        output_variance += diff * diff;
      }
      output_variance /= 5.0f;
    }
    metrics[i].output_stability = 1.0f / (1.0f + output_variance);

    // Prediction Error
    metrics[i].prediction_error = fabs(neurons[i].output - target_outputs[i]);

    // Connection Quality: Assess weight distribution and connection
    // effectiveness
    float total_connection_strength = 0.0f;
    float weight_variation = 0.0f;
    float mean_weight = 0.0f;

    for (int j = 0; j < neurons[i].num_connections; j++) {
      total_connection_strength += fabs(weights[i * MAX_CONNECTIONS + j]);
      mean_weight += weights[i * MAX_CONNECTIONS + j];
    }
    mean_weight /= neurons[i].num_connections;

    for (int j = 0; j < neurons[i].num_connections; j++) {
      float weight_diff = weights[i * MAX_CONNECTIONS + j] - mean_weight;
      weight_variation += weight_diff * weight_diff;
    }
    weight_variation /= neurons[i].num_connections;

    metrics[i].connection_quality =
        (total_connection_strength / neurons[i].num_connections) *
        (1.0f / (1.0f + weight_variation));

    // Adaptive Response: How well neuron responds to different input patterns
    metrics[i].adaptive_response =
        1.0f - fabs(neurons[i].output - neurons[i].state) /
                   (fabs(neurons[i].output) + 1e-5);

    // Importance Score: Combines all metrics with weighted approach
    metrics[i].importance_score =
        0.3f * (1.0f - metrics[i].output_stability) +
        0.3f * metrics[i].prediction_error +
        0.2f * (1.0f - metrics[i].connection_quality) +
        0.2f * (1.0f - metrics[i].adaptive_response);
  }
}

void advancedNeuronManagement(Neuron *neurons, int *connections, float *weights,
                              int *num_neurons, int max_neurons,
                              float *input_tensor, float *target_outputs,
                              NetworkStateSnapshot *stateHistory,
                              int current_step) {
  NeuronPerformanceMetric *metrics =
      malloc(*num_neurons * sizeof(NeuronPerformanceMetric));

  computeNeuronPerformanceMetrics(neurons, target_outputs, connections, weights,
                                  metrics, *num_neurons, stateHistory,
                                  current_step);

  // Find neurons for potential removal
  float removal_threshold =
      0.7f; // High importance score means poor performance
  for (int i = 0; i < *num_neurons; i++) {
    if (metrics[i].importance_score > removal_threshold) {
      removeUnderperformingNeuron(neurons, connections, weights, num_neurons,
                                  max_neurons, i);

      addNewNeuron(neurons, connections, weights, num_neurons, max_neurons,
                   input_tensor);
      break; // Remove and replace only one neuron per cycle
    }
  }

  free(metrics);
}

float evaluateFutureOutcome(Neuron *simulated_neurons, float *target_outputs,
                            int max_neurons) {
  float total_alignment = 0.0f;
  float prediction_confidence = 0.0f;

  // Compute alignment between simulated neuron states and target outputs
  for (int i = 0; i < max_neurons; i++) {
    float neuron_output = simulated_neurons[i].output;
    float target_output = target_outputs[i];

    // Compute alignment as cosine similarity
    float alignment = 1.0f - fabs(neuron_output - target_output);
    total_alignment += alignment;

    // Compute confidence based on output proximity
    prediction_confidence +=
        1.0f / (1.0f + fabs(neuron_output - target_output));
  }

  // Normalize alignment and confidence
  total_alignment /= max_neurons;
  prediction_confidence /= max_neurons;

  // Combine alignment and confidence with a weighted score
  return (0.6f * total_alignment) + (0.4f * prediction_confidence);
}

float *generatePotentialTargets(int max_neurons, float *previous_outputs,
                                NetworkStateSnapshot *stateHistory, int step,
                                MemoryEntry *relevantMemory,
                                DynamicParameters params) {
  float *target_outputs = (float *)malloc(sizeof(float) * max_neurons);
  if (target_outputs == NULL) {
    fprintf(stderr, "Failed to allocate memory for target outputs.\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < max_neurons; i++) {
    float memory_adjustment = 0.0f;

    // Check if there's relevant memory to influence the target output
    if (relevantMemory != NULL) {
      memory_adjustment = relevantMemory->vector[i] * 0.2f;
    }

    target_outputs[i] = (stateHistory[step].states[i] * 0.5f) +
                        (previous_outputs[i] * 0.3f) + memory_adjustment +
                        0.2f; // Adding a small bias term

    float adaptation_factor = params.current_adaptation_rate;
    target_outputs[i] *= adaptation_factor;
  }

  return target_outputs;
}

void simulateFutureStates(Neuron *neurons, float *weights, int *connections,
                          float *input_tensor, int max_neurons, int steps) {
  for (int step = 0; step < steps; step++) {
    for (int i = 0; i < max_neurons; i++) {
      neurons[i].state += input_tensor[i % INPUT_SIZE] * 0.1f;
    }
    processNeurons(neurons, max_neurons, weights, connections, MAX_CONNECTIONS,
                   1.5f);
  }
}

void selectOptimalDecisionPath(Neuron *neurons, float *weights,
                               int *connections, float *input_tensor,
                               int max_neurons, float *previous_outputs,
                               NetworkStateSnapshot *stateHistory, int step,
                               MemoryEntry *relevantMemory,
                               DynamicParameters params) {
  int simulation_depths[] = {3, 5, 7};
  float depth_outcomes[3];

  for (int i = 0; i < 3; i++) {
    Neuron simulation_copy[MAX_NEURONS];
    memcpy(simulation_copy, neurons, sizeof(Neuron) * max_neurons);

    simulateFutureStates(simulation_copy, weights, connections, input_tensor,
                         max_neurons, simulation_depths[i]);

    // Evaluate overall simulation outcome
    float *potential_targets =
        generatePotentialTargets(max_neurons, previous_outputs, stateHistory,
                                 step, relevantMemory, params);
    depth_outcomes[i] =
        evaluateFutureOutcome(simulation_copy, potential_targets, max_neurons);
  }

  // Select the simulation depth with the best outcome
  int best_depth_index = 0;
  for (int i = 1; i < 3; i++) {
    if (depth_outcomes[i] > depth_outcomes[best_depth_index]) {
      best_depth_index = i;
    }
  }

  // Apply modifications based on the best simulation
  simulateFutureStates(neurons, weights, connections, input_tensor, max_neurons,
                       simulation_depths[best_depth_index]);
}

ContextNode* createContextNode(const char *name, int vector_size, ContextNode *parent) {
    ContextNode *node = (ContextNode*)malloc(sizeof(ContextNode));
    node->name = strdup(name);
    node->importance = 1.0f;
    node->state_vector = (float*)calloc(vector_size, sizeof(float));
    node->vector_size = vector_size;
    node->children = (ContextNode**)malloc(sizeof(ContextNode*) * 10);
    node->num_children = 0;
    node->max_children = 10;
    node->parent = parent;
    node->temporal_relevance = 1.0f;
    node->last_updated = time(NULL);
    return node;
}

GlobalContextManager* initializeGlobalContextManager(int vector_size) {
    GlobalContextManager *manager = (GlobalContextManager*)malloc(sizeof(GlobalContextManager));
    manager->vector_size = vector_size;
    manager->total_nodes = 1;
    manager->decay_rate = 0.95f;
    manager->update_threshold = 0.1f;
    manager->max_depth = 5;
    manager->max_children_per_node = 10;
    
    // Initialize global context vector
    manager->global_context_vector = (float*)calloc(vector_size, sizeof(float));
    
    // Create root node
    manager->root = createContextNode("Global", vector_size, NULL);
    
    // Initialize default context hierarchy
    ContextNode *goals = createContextNode("Goals", vector_size, manager->root);
    ContextNode *constraints = createContextNode("Constraints", vector_size, manager->root);
    ContextNode *environment = createContextNode("Environment", vector_size, manager->root);
    
    manager->root->children[0] = goals;
    manager->root->children[1] = constraints;
    manager->root->children[2] = environment;
    manager->root->num_children = 3;
    
    return manager;
}

void updateContextNode(ContextNode *node, float *new_state, float importance) {
    for (int i = 0; i < node->vector_size; i++) {
        node->state_vector[i] = node->state_vector[i] * (1 - importance) + 
                               new_state[i] * importance;
    }
    node->last_updated = time(NULL);
    node->importance = fmax(node->importance * 0.95f + importance * 0.05f, 0.1f);
}

void propagateContextUpdates(ContextNode *node) {
    if (node->parent != NULL) {
        float *aggregated = (float*)calloc(node->vector_size, sizeof(float));
        float total_importance = 0.0f;
        
        // Aggregate child states weighted by importance
        for (int i = 0; i < node->parent->num_children; i++) {
            ContextNode *sibling = node->parent->children[i];
            float temp_relevance = expf(-(time(NULL) - sibling->last_updated) / 3600.0f);
            float weight = sibling->importance * temp_relevance;
            
            for (int j = 0; j < node->vector_size; j++) {
                aggregated[j] += sibling->state_vector[j] * weight;
            }
            total_importance += weight;
        }
        
        // Normalize and update parent
        if (total_importance > 0) {
            for (int i = 0; i < node->vector_size; i++) {
                aggregated[i] /= total_importance;
            }
            updateContextNode(node->parent, aggregated, 0.3f);
        }
        
        free(aggregated);
        propagateContextUpdates(node->parent);
    }
}

ContextNode* findContextNode(ContextNode *root, const char *name) {
    if (strcmp(root->name, name) == 0) {
        return root;
    }
    
    for (int i = 0; i < root->num_children; i++) {
        ContextNode *result = findContextNode(root->children[i], name);
        if (result) {
            return result;
        }
    }
    
    return NULL;
}


ContextNode* addContextNode(GlobalContextManager *manager, const char *name, 
                          const char *parent_name, float *initial_state) {
    ContextNode *parent = findContextNode(manager->root, parent_name);
    if (!parent || parent->num_children >= manager->max_children_per_node) {
        return NULL;
    }
    
    ContextNode *new_node = createContextNode(name, manager->vector_size, parent);
    if (initial_state) {
        memcpy(new_node->state_vector, initial_state, 
               manager->vector_size * sizeof(float));
    }
    
    parent->children[parent->num_children++] = new_node;
    manager->total_nodes++;
    
    return new_node;
}

float evaluateConstraintSatisfaction(ContextNode *constraint, Neuron *neurons, 
                                   int num_neurons) {
    float satisfaction = 1.0f;
    
    // Example constraint evaluation
    if (strcmp(constraint->name, "ActivityLevel") == 0) {
        float total_activity = 0;
        for (int i = 0; i < num_neurons; i++) {
            total_activity += neurons[i].output;
        }
        satisfaction = 1.0f - fabs(0.5f - (total_activity / num_neurons));
    }
    
    return satisfaction;
}


void updateGlobalContext(GlobalContextManager *manager, Neuron *neurons, 
                        int num_neurons, float *input_tensor) {
    // Extract relevant features from current network state
    float *current_context = (float*)calloc(manager->vector_size, sizeof(float));
    
    // Analyze network activity patterns
    for (int i = 0; i < manager->vector_size && i < num_neurons; i++) {
        current_context[i] = neurons[i].output;
    }
    
    // Update environmental context
    ContextNode *env_node = findContextNode(manager->root, "Environment");
    if (env_node) {
        updateContextNode(env_node, current_context, 0.2f);
        propagateContextUpdates(env_node);
    }
    
    // Update constraint satisfaction levels
    ContextNode *constraints = findContextNode(manager->root, "Constraints");
    if (constraints) {
        float *constraint_state = (float*)calloc(manager->vector_size, sizeof(float));
        // Evaluate current constraint satisfaction
        for (int i = 0; i < constraints->num_children; i++) {
            float satisfaction = evaluateConstraintSatisfaction(
                constraints->children[i], neurons, num_neurons);
            for (int j = 0; j < manager->vector_size; j++) {
                constraint_state[j] += satisfaction;
            }
        }
        updateContextNode(constraints, constraint_state, 0.3f);
        free(constraint_state);
    }
    
    // Update global context vector
    for (int i = 0; i < manager->vector_size; i++) {
        manager->global_context_vector[i] = 0;
        float total_weight = 0;
        
        // Weighted combination of all top-level contexts
        for (int j = 0; j < manager->root->num_children; j++) {
            ContextNode *child = manager->root->children[j];
            float weight = child->importance * child->temporal_relevance;
            manager->global_context_vector[i] += child->state_vector[i] * weight;
            total_weight += weight;
        }
        
        if (total_weight > 0) {
            manager->global_context_vector[i] /= total_weight;
        }
    }
    
    free(current_context);
}

void integrateGlobalContext(GlobalContextManager *manager, Neuron *neurons, 
                           int num_neurons, float *weights, 
                           int max_connections) {
    // Modulate neuron behavior based on global context
    for (int i = 0; i < num_neurons; i++) {
        float context_influence = 0;
        
        // Compute context influence on this neuron
        for (int j = 0; j < manager->vector_size && j < num_neurons; j++) {
            context_influence += manager->global_context_vector[j] * 
                               weights[i * max_connections + j];
        }
        
        // Apply context modulation
        neurons[i].state = neurons[i].state * (1.0f + 0.1f * context_influence);
        neurons[i].output = tanh(neurons[i].state);
    }
}

float computeOutcomeMetric(Neuron *neurons, float *targets, int size) {
    float total_error = 0.0f;
    for (int i = 0; i < size; i++) {
        float error = fabs(neurons[i].output - targets[i]);
        total_error += error;
    }
    return 1.0f / (1.0f + total_error / size); // Normalize to [0,1]
}

void updateCorrelationMatrix(float *correlation_matrix,
                           float *input_history,
                           float *outcomes,
                           int history_length,
                           int input_size) {
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < input_size; j++) {
            float correlation = 0.0f;
            for (int h = 0; h < history_length; h++) {
                correlation += input_history[h * input_size + i] * 
                             input_history[h * input_size + j] * 
                             outcomes[h];
            }
            correlation_matrix[i * input_size + j] = 
                correlation / history_length;
        }
    }
}

float computeFeedbackSignal(float current_outcome,
                          float *feedback_history,
                          int history_size) {
    float recent_average = 0.0f;
    int count = 0;
    for (int i = 0; i < history_size; i++) {
        if (feedback_history[i] > 0.0f) {
            recent_average += feedback_history[i];
            count++;
        }
    }
    if (count > 0) {
        recent_average /= count;
    }
    return current_outcome - recent_average;
}

void applyDynamicContext(Neuron *neurons,
                        float *context_weights,
                        GlobalContextManager *context,
                        int size) {
    for (int i = 0; i < size; i++) {
        neurons[i].state = neurons[i].state * (1.0f - context_weights[i]) +
                          context->global_context_vector[i] * context_weights[i];
    }
}

float computeAverageFeedback(float *feedback_history, int history_size) {
    float sum = 0.0f;
    int valid_count = 0;
    
    for (int i = 0; i < history_size; i++) {
        if (feedback_history[i] != 0.0f) {  // Only count non-zero feedback
            sum += feedback_history[i];
            valid_count++;
        }
    }
    
    return valid_count > 0 ? sum / valid_count : 0.0f;
}

float computeMinWeight(float *weights, int size) {
    if (size <= 0 || weights == NULL) return 0.0f;
    
    float min_weight = weights[0];
    for (int i = 1; i < size; i++) {
        if (weights[i] < min_weight) {
            min_weight = weights[i];
        }
    }
    
    return min_weight;
}

float computeMaxWeight(float *weights, int size) {
    if (size <= 0 || weights == NULL) return 0.0f;
    
    float max_weight = weights[0];
    for (int i = 1; i < size; i++) {
        if (weights[i] > max_weight) {
            max_weight = weights[i];
        }
    }
    
    return max_weight;
}

float computeAverageCorrelation(float *correlation_matrix, int size) {
    if (size <= 0 || correlation_matrix == NULL) return 0.0f;
    
    float sum = 0.0f;
    int count = 0;
    
    // Compute average of absolute correlation values
    // Skip diagonal elements (self-correlation)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i != j) {  // Skip diagonal
                sum += fabs(correlation_matrix[i * size + j]);
                count++;
            }
        }
    }
    
    return count > 0 ? sum / count : 0.0f;
}

int main() {

  // Try to load existing memory system
  MemorySystem *memorySystem = NULL;
  FILE *mem_file = fopen("memory_system.dat", "rb");
  if (mem_file != NULL) {
    fclose(mem_file);
    memorySystem = loadMemorySystem("memory_system.dat");
    if (memorySystem != NULL) {
      printf("Loaded existing memory system\n");

      loadHierarchicalMemory(memorySystem, "hierarchical_memory.dat");

      printf("\nMemory System Statistics:\n");
      printf("Total Capacity: %u\n", memorySystem->capacity);
      printf("Short-term memories: %u/%u\n",
             memorySystem->hierarchy.short_term.size,
             memorySystem->hierarchy.short_term.capacity);
      printf("Medium-term memories: %u/%u\n",
             memorySystem->hierarchy.medium_term.size,
             memorySystem->hierarchy.medium_term.capacity);
      printf("Long-term memories: %u/%u\n",
             memorySystem->hierarchy.long_term.size,
             memorySystem->hierarchy.long_term.capacity);

      printf("\nMemory Samples:\n");
      if (memorySystem->hierarchy.long_term.size > 0) {
        printf("Long-term memory sample (importance: %.2f)\n",
               memorySystem->hierarchy.long_term.entries[0].importance);
      }
      if (memorySystem->hierarchy.medium_term.size > 0) {
        printf("Medium-term memory sample (importance: %.2f)\n",
               memorySystem->hierarchy.medium_term.entries[0].importance);
      }
      if (memorySystem->hierarchy.short_term.size > 0) {
        printf("Short-term memory sample (importance: %.2f)\n",
               memorySystem->hierarchy.short_term.entries[0].importance);
      }
    }
  }

  if (memorySystem == NULL) {
    printf("Creating new hierarchical memory system...\n");
    memorySystem = createMemorySystem(MEMORY_BUFFER_SIZE);
  }

  NetworkStateSnapshot *stateHistory =
      (NetworkStateSnapshot *)malloc(STEPS * sizeof(NetworkStateSnapshot));
  if (stateHistory == NULL) {
    fprintf(stderr, "Failed to allocate memory for state history\n");
    freeMemorySystem(memorySystem);
    return -1;
  }

  PerformanceMetrics *performance_history =
      (PerformanceMetrics *)malloc(STEPS * sizeof(PerformanceMetrics));
  OptimizationState opt_state = {.optimal_batch_size = 1,
                                 .optimal_learning_rate = 0.01f,
                                 .best_execution_time = INFINITY,
                                 .best_performance_score = -INFINITY};

  float *previous_outputs = (float *)malloc(MAX_NEURONS * sizeof(float));

  int reverse_connections[MAX_NEURONS * MAX_CONNECTIONS] = {0};
  float reverse_weights[MAX_NEURONS * MAX_CONNECTIONS] = {0};

  // Initialize neural network structures
  Neuron neurons[MAX_NEURONS];
  int connections[MAX_NEURONS * MAX_CONNECTIONS] = {0};
  float weights[MAX_NEURONS * MAX_CONNECTIONS] = {0};
  // Create constant buffers
  int max_neurons = MAX_NEURONS;
  int max_connections = MAX_CONNECTIONS;
  int input_size = INPUT_SIZE;
  float *input_tensor = (float *)malloc(max_neurons * sizeof(float));

  // Initialize neurons from memory or with default values
  if (memorySystem->size > 0) {
    int lastMemoryIdx = (memorySystem->head - 1 + memorySystem->capacity) %
                        memorySystem->capacity;
    MemoryEntry *lastMemory = &memorySystem->entries[lastMemoryIdx];

    printf("\nInitializing neurons from last memory state...\n");

    for (int i = 0; i < MAX_NEURONS; i++) {
      neurons[i].state = lastMemory->vector[i];
      neurons[i].output = lastMemory->vector[i + MAX_NEURONS];
      neurons[i].num_connections = 2;
      neurons[i].layer_id = i % 2;
    }

    // Initialize connections and weights
    for (int i = 0; i < MAX_NEURONS; i++) {
      connections[i * MAX_CONNECTIONS] = (i + 1) % MAX_NEURONS;
      connections[i * MAX_CONNECTIONS + 1] =
          (i - 1 + MAX_NEURONS) % MAX_NEURONS;
      weights[i * MAX_CONNECTIONS] = 0.6f;
      weights[i * MAX_CONNECTIONS + 1] = -0.4f;
    }
  } else {
    initializeNeurons(neurons, connections, weights, input_tensor);
  }

  float learning_rate = 0.01f;
  for (int i = 0; i < MAX_NEURONS; i++) {
    // Mirror forward connections with reverse direction
    reverse_connections[i * MAX_CONNECTIONS] =
        (i - 1 + MAX_NEURONS) % MAX_NEURONS;
    reverse_weights[i * MAX_CONNECTIONS] = weights[i * MAX_CONNECTIONS + 1];

    reverse_connections[i * MAX_CONNECTIONS + 1] = (i + 2) % MAX_NEURONS;
    reverse_weights[i * MAX_CONNECTIONS + 1] = -0.3f;
  }
  // Initialize weights
  initializeWeights(weights, MAX_NEURONS, MAX_CONNECTIONS, input_tensor);
  DynamicParameters params = initDynamicParameters();
  SystemParameters *system_params =
      loadSystemParameters("system_parameters.dat");
  if (system_params) {
    opt_state = system_params->opt_state;
    params = system_params->dynamic_params;
  }
  float target_outputs[MAX_NEURONS];
  const char *text_input =
      "Apple, banana, cherry, date, and elderberry are fruits.";
  initializeEmbeddings();

  int network_regions = 2; // Assuming 2 layers
  MetaController *metaController = initializeMetaController(network_regions);
  NetworkPerformanceMetrics *performanceMetrics =
      initializePerformanceMetrics(network_regions);


  // Main loop    
  printf("\nStarting training with loaded memory state...\n");
  for (int step = 0; step < STEPS; step++) {
    double step_start_time = getCurrentTime();

    TaskPrompt current_prompt;
    generateTaskPrompt(&current_prompt, step);

    // Store previous outputs for error calculation
    float *previous_outputs = (float *)malloc(max_neurons * sizeof(float));
    for (int i = 0; i < max_neurons; i++) {
      previous_outputs[i] = neurons[i].output;
    }

    // Get last timestamp for continuity
    unsigned int lastTimestamp =
        (memorySystem->size > 0)
            ? memorySystem
                  ->entries[(memorySystem->head - 1 + memorySystem->capacity) %
                            memorySystem->capacity]
                  .timestamp
            : 0;

    // Retrieve the most relevant memory
    MemoryEntry *relevantMemory = retrieveMemory(memorySystem);

    initPredictiveCodingParams(int max_neurons)
    float *predictive_inputs = (float *)malloc(max_neurons * sizeof(float));
    generatePredictiveInputs(predictive_inputs,
                             (step > 0) ? &stateHistory[step - 1] : NULL,
                             max_neurons);

    // Create input tensor
    float *input_tensor = (float *)malloc(max_neurons * sizeof(float));
    memcpy(input_tensor, predictive_inputs, max_neurons * sizeof(float));
    generateInputTensor(input_tensor, step, text_input, relevantMemory,
                        system_params);

    // Memory maintenance
    if (step % 10 == 0) {
      decayMemorySystem(memorySystem);
      mergeSimilarMemories(memorySystem);
      printf("\nMemory System Status (Step %d):\n", step);
      printf("Short-term memories: %u\n",
             memorySystem->hierarchy.short_term.size);
      printf("Medium-term memories: %u\n",
             memorySystem->hierarchy.medium_term.size);
      printf("Long-term memories: %u\n",
             memorySystem->hierarchy.long_term.size);
    }

    // Allocate device memory
    Neuron *d_neurons;
    float *d_weights, *d_input_tensor, *d_recurrent_weights;
    unsigned int *d_connections;

    cudaMalloc(&d_neurons, max_neurons * sizeof(Neuron));
    cudaMalloc(&d_weights, max_neurons * max_connections * sizeof(float));
    cudaMalloc(&d_input_tensor, max_neurons * sizeof(float));
    cudaMalloc(&d_connections,
               max_neurons * max_connections * sizeof(unsigned int));
    cudaMalloc(&d_recurrent_weights, max_neurons * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_neurons, neurons, max_neurons * sizeof(Neuron),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights,
               max_neurons * max_connections * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_tensor, input_tensor, max_neurons * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_connections, connections,
               max_neurons * max_connections * sizeof(unsigned int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_recurrent_weights, recurrent_weights,
               max_neurons * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    dim3 blockDim(256);
    dim3 gridDim((max_neurons + blockDim.x - 1) / blockDim.x);

    update_neurons<<<gridDim, blockDim>>>(
        d_neurons, d_weights, d_connections, max_neurons, max_connections,
        d_input_tensor, input_size, d_recurrent_weights)

        cudaMemcpy(neurons, d_neurons, max_neurons * sizeof(Neuron),
                   cudaMemcpyDeviceToHost);

    computePredictionErrors(neurons, input_tensor, max_neurons);

    float *target_outputs = (float *)malloc(max_neurons * sizeof(float));
    target_outputs =
        generatePotentialTargets(max_neurons, previous_outputs, stateHistory,
                                 step, relevantMemory, params);

    // Decision path selection
    selectOptimalDecisionPath(neurons, weights, connections, input_tensor,
                              MAX_NEURONS, previous_outputs, stateHistory, step,
                              relevantMemory, params);

    // Update performance metrics
    computeRegionPerformanceMetrics(performanceMetrics, neurons, target_outputs,
                                    MAX_NEURONS);

    // Update meta-controller
    updateMetaControllerPriorities(metaController, performanceMetrics);
    applyMetaControllerAdaptations(neurons, weights, metaController,
                                   MAX_NEURONS);

    // Meta-controller logging
    if (step % 20 == 0) {
      printf("\nMeta-Controller Insights (Step %d):\n", step);
      for (int i = 0; i < network_regions; i++) {
        printf("Region %d:\n", i);
        printf("  Importance Score: %.4f\n",
               metaController->region_importance_scores[i]);
        printf("  Performance Score: %.4f\n",
               performanceMetrics->region_performance_scores[i]);
        printf("  Error Rate: %.4f\n",
               performanceMetrics->region_error_rates[i]);
      }
    }

    float *d_target_outputs, *d_output_errors;
    cudaMalloc(&d_target_outputs, max_neurons * sizeof(float));
    cudaMalloc(&d_output_errors, max_neurons * sizeof(float));
    cudaMemcpy(d_target_outputs, target_outputs, max_neurons * sizeof(float),
               cudaMemcpyHostToDevice);

    // Backward pass using backward_kernel
    backward_kernel<<<gridDim, blockDim>>>(
        d_neurons, d_weights, (uint2 *)d_connections, max_neurons,
        max_connections, d_target_outputs, d_output_errors, learning_rate);

    // Update weights
    update_weights<<<gridDim, blockDim>>>(d_weights, d_neurons, d_connections,
                                          learning_rate, max_neurons,
                                          max_connections);

    process_neurons<<<gridDim, blockDim>>>(
        d_neurons, d_weights, d_connections, max_neurons, max_connections,
        d_input_tensor, input_size, d_recurrent_weights);

    reverse_process<<<gridDim, blockDim>>>(
        d_neurons, d_weights, d_connections, max_neurons, max_connections,
        d_input_tensor, input_size, d_recurrent_weights);

    // Memory replay mechanism
    if (step % 5 == 0 && memorySystem->size > 10) {
      MemoryEntry *d_memories;
      cudaMalloc(&d_memories, memorySystem->capacity * sizeof(MemoryEntry));
      cudaMemcpy(d_memories, memorySystem->entries,
                 memorySystem->capacity * sizeof(MemoryEntry),
                 cudaMemcpyHostToDevice);

      memory_replay<<<gridDim, blockDim>>>(d_neurons, d_weights, d_connections,
                                           d_memories, memorySystem->capacity);

      cudaFree(d_memories);
      printf("\nMemory Replay at step %d:", step);
      printReplayStatistics(memorySystem);
    }

    // Copy results back to host
    cudaMemcpy(neurons, d_neurons, max_neurons * sizeof(Neuron),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(weights, d_weights,
               max_neurons * max_connections * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Compute loss
    float loss = computeMSELoss(neurons, target_outputs, max_neurons);
    printf("Loss: %f\n", loss);

    // Verify network state
    verifyNetworkState(neurons, &current_prompt);

    // Update memory system
    addMemory(memorySystem, neurons, input_tensor, lastTimestamp + step + 1);

    // Update state history
    captureNetworkState(neurons, input_tensor, &stateHistory[step], weights,
                        step);
    stateHistory[step].current_memory =
        memorySystem
            ->entries[(memorySystem->head - 1 + memorySystem->capacity) %
                      memorySystem->capacity];

    // Print progress
    printf("\nStep %d (Timestamp: %u):\n", step, lastTimestamp + step + 1);
    printNetworkStates(neurons, input_tensor, step);

    // Task verification logging
    if (step % 10 == 0) {
      printf("\nTask Verification (Step %d):\n", step);
      printf("Description: %s\n", current_prompt.task_description);
      for (int v = 0; v < 5; v++) {
        if (current_prompt.verifications[v].instruction[0] != '\0') {
          printf("- %s: %s (Confidence: %.2f)\n",
                 current_prompt.verifications[v].instruction,
                 current_prompt.verifications[v].verified ? "PASSED" : "FAILED",
                 current_prompt.verifications[v].confidence);
          printf("  Reasoning: %s\n",
                 current_prompt.verifications[v].reasoning);
        }
      }
    }

    // Memory verification
    if (relevantMemory != NULL) {
      PromptVerification memoryVerification = {.instruction =
                                                   "Verify memory integration",
                                               .confidence = 0.0f,
                                               .verified = false};

      float memory_coherence = assessMemoryCoherence(relevantMemory, neurons);
      sprintf(memoryVerification.reasoning,
              "Memory coherence: %.2f%% - Integration quality: %s",
              memory_coherence * 100.0f,
              memory_coherence > 0.7f ? "Good" : "Needs improvement");

      memoryVerification.confidence = memory_coherence;
      memoryVerification.verified = memory_coherence > 0.7f;

      current_prompt.verifications[1] = memoryVerification;
    }

    // Memory consolidation
    if (step % 10 == 0) {
      consolidateMemory(memorySystem);
    }

    // Update weights dynamically
    updateWeights(weights, neurons, connections, learning_rate);

    performance_history[step].execution_time =
        getCurrentTime() - step_start_time;
    performance_history[step].average_output = computeAverageOutput(neurons);
    performance_history[step].error_rate =
        computeErrorRate(neurons, previous_outputs);
    performance_history[step].batch_size = opt_state.optimal_batch_size;
    performance_history[step].learning_rate = opt_state.optimal_learning_rate;

    // Optimize parameters periodically
    if (step % OPTIMIZATION_WINDOW == 0 && step > 0) {
      PromptVerification optVerification = {.instruction =
                                                "Verify parameter optimization",
                                            .confidence = 0.0f,
                                            .verified = false};

      float improvement =
          (opt_state.best_performance_score -
           performance_history[step - OPTIMIZATION_WINDOW].error_rate) /
          performance_history[step - OPTIMIZATION_WINDOW].error_rate;

      sprintf(optVerification.reasoning,
              "Performance improvement: %.2f%% - Parameters updated: %s",
              improvement * 100.0f,
              improvement > 0 ? "Successfully" : "No improvement");

      optVerification.confidence = fmax(0.0f, improvement);
      optVerification.verified = improvement > 0;

      current_prompt.verifications[2] = optVerification;
      optimizeParameters(&opt_state, performance_history, step + 1);

      printf("\nOptimization Update (Step %d):\n", step);
      printf("Current execution time: %.6f seconds\n",
             performance_history[step].execution_time);
      printf("Best execution time: %.6f seconds\n",
             opt_state.best_execution_time);
      printf("Optimal batch size: %d\n", opt_state.optimal_batch_size);
      printf("Optimal learning rate: %.6f\n", opt_state.optimal_learning_rate);
      printf("Performance score: %.4f\n", opt_state.best_performance_score);
    }

    float *previous_states = (float *)malloc(max_neurons * sizeof(float));
    for (int i = 0; i < max_neurons; i++) {
      previous_states[i] = neurons[i].state;
    }

    if (system_params != NULL) {
      system_params->opt_state = opt_state;
      system_params->dynamic_params = params;
      if (opt_state.best_performance_score >
          system_params->best_performance_score) {
        system_params->best_performance_score =
            opt_state.best_performance_score;
      }
      system_params->timestamp = time(NULL);
    }

    float stability = measureNetworkStability(neurons, previous_states);
    float performance_delta =
        performance_history[step].average_output -
        (step > 0 ? performance_history[step - 1].average_output : 0);

    // Update dynamic parameters
    updateDynamicParameters(&params, performance_delta, stability,
                            performance_history[step].error_rate);

    // Adapt network with dynamic parameters
    adaptNetworkDynamic(neurons, weights, &params, performance_delta,
                        input_tensor);

    // Optional: Print adaptation parameters periodically
    if (step % 10 == 0) {
      printf("\nDynamic Parameters at step %d:\n", step);
      printf("Current Adaptation Rate: %.4f\n", params.current_adaptation_rate);
      printf("Input Noise Scale: %.4f\n", params.input_noise_scale);
      printf("Weight Noise Scale: %.4f\n", params.weight_noise_scale);
      printf("Plasticity: %.4f\n", params.plasticity);
      printf("Noise Tolerance: %.4f\n", params.noise_tolerance);
    }

    if (step % 50 == 0 && step > 0) { // Every 50 steps
      printf("\nPerformance Analysis and Graph Generation at step %d:\n", step);
      analyzeNetworkPerformance(performance_history, step + 1);
      generatePerformanceGraph(performance_history, step + 1);
    }
    char outputText[4096];
    transformOutputsToText(previous_outputs, MAX_NEURONS, outputText,
                           sizeof(outputText));
    printf("\nStep %d Outputs (Text):\n%s\n", step, outputText);

    if (step % 20 == 0) {
      PatternMatchingParams params = {.similarity_threshold = 0.8f,
                                      .temporal_window = 5,
                                      .temporal_decay = 0.9f,
                                      .max_matches = 3};

      // Find similar patterns in each memory level
      int num_matches;
      PatternMatch *matches = findSimilarMemoriesInCluster(
          &memorySystem->hierarchy.long_term,
          stateHistory[step].current_memory.vector, params.similarity_threshold,
          &num_matches);

      if (num_matches > 0) {
        printf("\nFound %d similar patterns in long-term memory\n",
               num_matches);
        free(matches);
      }
    }

    // Use optimized parameters
    learning_rate = opt_state.optimal_learning_rate;
    updateNeuronsWithPredictiveCoding(neurons, input_tensor, max_neurons,
                                      learning_rate);

    // Process in batches
    for (int b = 0; b < MAX_NEURONS; b += opt_state.optimal_batch_size) {
      int batch_end = b + opt_state.optimal_batch_size;
      if (batch_end > MAX_NEURONS)
        batch_end = MAX_NEURONS;

      // Process batch
      for (int i = b; i < batch_end; i++) {
        updateNeuronStates(neurons, max_neurons, weights, 1.5f);
      }
    }
    float total_error = 0.0f;
    for (int i = 0; i < max_neurons; i++) {
      float error = fabs(neurons[i].output - target_outputs[i]);
      total_error += error;
    }
    updateBidirectionalWeights(weights, reverse_weights, neurons, connections,
                               reverse_connections, learning_rate);

    float average_error = total_error / max_neurons;
    if (step % 15 == 0) {
      advancedNeuronManagement(neurons, connections, weights, &max_neurons,
                               MAX_NEURONS, input_tensor, target_outputs,
                               stateHistory, step);
    }
    printf("Average Error: %f", average_error);
    double throughput = STEPS / performance_history[step].execution_time;
    printf("Throughput: %f steps/s", throughput);
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("Memory Usage Benchmark:\n");
    printf("Max Resident Set Size: %ld KB\n", usage.ru_maxrss);
  }

  // Save final state
  saveNetworkStates(stateHistory, STEPS);
  saveMemorySystem(memorySystem, "memory_system.dat");
  saveHierarchicalMemory(memorySystem, "hierarchical_memory.dat");
  saveSystemParameters(system_params, "system_parameters.dat");

  printf("\nNeural network states, memory system and system parameters have "
         "been saved\n");

  generatePerformanceGraph(performance_history, STEPS);

  // Cleanup
  freeMemorySystem(memorySystem);
  free(stateHistory);
  free(system_params);
  return 0;
}
