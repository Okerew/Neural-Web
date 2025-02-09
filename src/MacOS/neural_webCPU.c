#include <float.h>
#include <json-c/json.h>
#include <simd/simd.h>
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
#define MIN_WEIGHT -1.0f
#define MAX_WEIGHT 1.0f
#define DECAY_RATE 0.8f
#define CONNECTION_WEIGHT 0.2f
#define INPUT_WEIGHT 0.1f
#define ACTIVATION_SCALE 1.5f
#define ACTIVATION_BIAS 0.1f
#define MIN_ACTIVATION -1.0f
#define MAX_ACTIVATION 1.0f
#define FEATURE_VECTOR_SIZE 128
#define CONTEXT_VECTOR_SIZE 256
#define CLAMP_MIN -1e6f // Min value for feature or coherence
#define CLAMP_MAX 1e6f  // Max value for feature or coherence

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
  float adaptation_speed;     // Speed of recovery from perturbations
  float baseline_performance; // Performance without noise
  float noisy_performance;    // Performance with noise
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
  float output_stability; // Variation in neuron's output
  float prediction_error;
  float connection_quality;
  float adaptive_response; // Neuron's ability to adapt to different inputs
  float importance_score;  // Overall significance in network
} NeuronPerformanceMetric;

typedef struct {
  const char *word;
  const char *category;    // e.g., "fruit", "common", "action"
  float semantic_weight;   // How strongly this word relates to its category
  const char *connects_to; // The most likely word it connects with
  const char *description; // Detailed description of the word
  float letter_weight;     // New field for letter-based weight
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

// Add these new structures after the existing includes
typedef struct {
  float novelty_score;
  float competence_score;
  float autonomy_score;
  float mastery_level;
  float curiosity_drive;
  float achievement_drive;
  float exploration_rate;
} IntrinsicMotivation;

typedef struct {
  char description[256];
  float priority;
  float progress;
  float reward_value;
  bool achieved;
  int timestamp;
} Goal;

typedef struct {
  Goal *goals;
  int num_goals;
  int capacity;
  float planning_horizon;
  float discount_factor;
} GoalSystem;

// Enhanced memory structures with semantic clustering
typedef struct {
  float *vector;     // Semantic vector representing the cluster center
  unsigned int size; // Number of memories in cluster
  float coherence;   // Measure of cluster coherence
  float *activation; // Dynamic activation level
} SemanticCluster;

typedef struct {
  SemanticCluster *clusters;
  unsigned int num_clusters;
  float *similarity_matrix;
} DynamicClusterSystem;

typedef struct {
  float *features;         // Extracted semantic features
  float abstraction_level; // Level of detail/abstraction
  float *context_vector;   // Contextual information
  unsigned int depth;      // Hierarchical depth
} WorkingMemoryEntry;

typedef struct {
  struct {
    WorkingMemoryEntry *entries;
    unsigned int size;
    unsigned int capacity;
    float attention_threshold;
  } focus; // Focused attention component

  struct {
    WorkingMemoryEntry *entries;
    unsigned int size;
    unsigned int capacity;
    float activation_decay;
  } active; // Active working memory

  DynamicClusterSystem clusters;
  float *global_context;
} WorkingMemorySystem;

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

// Normalize a vector to unit length
void normalizeVector(float *vector, unsigned int size) {
  float norm = 0.0f;

  // Calculate L2 norm (Euclidean length)
  for (unsigned int i = 0; i < size; i++) {
    norm += vector[i] * vector[i];
  }
  norm = sqrt(norm);

  // Prevent division by zero
  if (norm > 1e-6f) {
    // Divide each component by norm
    for (unsigned int i = 0; i < size; i++) {
      vector[i] /= norm;
    }
  }
}

// Add a weighted vector to a target vector
void addWeightedVector(float *target, const float *source, float weight,
                       unsigned int size) {
  for (unsigned int i = 0; i < size; i++) {
    target[i] += source[i] * weight;
  }
}

WorkingMemorySystem *createWorkingMemorySystem(unsigned int capacity) {
  WorkingMemorySystem *system = malloc(sizeof(WorkingMemorySystem));

  // Initialize focused attention component
  system->focus.capacity = capacity * 0.2; // 20% for focused attention
  system->focus.entries =
      malloc(system->focus.capacity * sizeof(WorkingMemoryEntry));
  system->focus.size = 0;
  system->focus.attention_threshold = 0.8f;

  // Initialize active working memory
  system->active.capacity = capacity * 0.8; // 80% for active memory
  system->active.entries =
      malloc(system->active.capacity * sizeof(WorkingMemoryEntry));
  system->active.size = 0;
  system->active.activation_decay = 0.95f;

  // Initialize dynamic clustering
  system->clusters.num_clusters = 5; // Start with 5 clusters
  system->clusters.clusters =
      malloc(system->clusters.num_clusters * sizeof(SemanticCluster));

  // Initialize global context
  system->global_context = calloc(CONTEXT_VECTOR_SIZE, sizeof(float));

  return system;
}

float clampValue(float value) {
  if (!isfinite(value)) {
    return 0.0f; // Reset NaN or Inf to 0
  }
  if (value < CLAMP_MIN) {
    return CLAMP_MIN; // Avoid underflow
  }
  if (value > CLAMP_MAX) {
    return CLAMP_MAX; // Avoid overflow
  }
  return value;
}

float computeActivation(const WorkingMemoryEntry *entry) {
  float activation = 0.0f;

  // Base activation on feature similarity to current context
  for (unsigned int i = 0; i < FEATURE_VECTOR_SIZE; i++) {
    if (!isfinite(entry->features[i])) {
      entry->features[i] = 0.0f;
    }
    if (!isfinite(entry->context_vector[i])) {
      entry->context_vector[i] = 0.0f;
    }

    // Accumulate the activation
    activation += entry->features[i] * entry->context_vector[i];
  }

  // Clamp the activation value to prevent overflow issues with the sigmoid
  // function
  activation = clampValue(activation);

  // Normalize activation to [0,1] range using sigmoid function
  activation = 1.0f / (1.0f + expf(-activation));

  return activation;
}

// Compute cosine similarity between two vectors
float computeCosineSimilarity(const float *vec1, const float *vec2,
                              unsigned int size) {
  float dot_product = 0.0f;
  float norm1 = 0.0f;
  float norm2 = 0.0f;

  for (unsigned int i = 0; i < size; i++) {
    dot_product += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }

  norm1 = sqrt(norm1);
  norm2 = sqrt(norm2);

  if (norm1 > 1e-6f && norm2 > 1e-6f) {
    return dot_product / (norm1 * norm2);
  }

  return 0.0f;
}

// Update coherence measure for a semantic cluster
void updateClusterCoherence(SemanticCluster *cluster) {
  float total_similarity = 0.0f;
  unsigned int num_pairs = 0;

  // Calculate average pairwise similarity between all vectors in cluster
  for (unsigned int i = 0; i < cluster->size; i++) {
    for (unsigned int j = i + 1; j < cluster->size; j++) {
      float similarity = computeCosineSimilarity(
          cluster->vector, // Using cluster center as reference
          cluster->vector, // This would normally be individual memories
          FEATURE_VECTOR_SIZE);
      total_similarity += similarity;
      num_pairs++;
    }
  }

  // Update coherence measure
  if (num_pairs > 0) {
    cluster->coherence = total_similarity / num_pairs;
  } else {
    cluster->coherence = 1.0f; // Perfect coherence for single-element clusters
  }
}

void updateSemanticClusters(WorkingMemorySystem *system,
                            WorkingMemoryEntry *entry) {
  if (system->clusters.num_clusters == 0) {
    return; // No clusters to update
  }

  float *similarities = malloc(system->clusters.num_clusters * sizeof(float));

  // Calculate similarities to existing clusters
  for (unsigned int i = 0; i < system->clusters.num_clusters; i++) {
    if (!system->clusters.clusters[i].vector) {
      free(similarities);
      return; // Avoid accessing uninitialized memory
    }
    similarities[i] = computeCosineSimilarity(
        entry->features, system->clusters.clusters[i].vector,
        FEATURE_VECTOR_SIZE);
  }

  // Find best matching cluster
  unsigned int best_cluster = 0;
  float best_similarity = similarities[0];
  for (unsigned int i = 1; i < system->clusters.num_clusters; i++) {
    if (similarities[i] > best_similarity) {
      best_similarity = similarities[i];
      best_cluster = i;
    }
  }

  // Ensure best cluster index is valid
  if (best_cluster >= system->clusters.num_clusters) {
    free(similarities);
    return;
  }

  // Update cluster center
  SemanticCluster *cluster = &system->clusters.clusters[best_cluster];
  if (!cluster->vector) {
    free(similarities);
    return; // Avoid accessing uninitialized memory
  }

  for (unsigned int i = 0; i < FEATURE_VECTOR_SIZE; i++) {
    cluster->vector[i] =
        (cluster->vector[i] * cluster->size + entry->features[i]) /
        (cluster->size + 1);
  }
  cluster->size++;

  // Update cluster coherence
  updateClusterCoherence(cluster);

  free(similarities);
}

// Find the least active memory in a given array
int findLeastActiveMemory(const WorkingMemoryEntry *entries,
                          unsigned int size) {
  int least_active_idx = 0;
  float lowest_activation = computeActivation(&entries[0]);

  for (unsigned int i = 1; i < size; i++) {
    float activation = computeActivation(&entries[i]);
    if (activation < lowest_activation) {
      lowest_activation = activation;
      least_active_idx = i;
    }
  }

  return least_active_idx;
}

// Compute attention weight for a memory entry
float computeAttentionWeight(const WorkingMemoryEntry *entry) {
  // Base weight on combination of recency and importance
  float recency_weight = expf(-entry->depth * 0.1f); // Decay with depth
  float importance_weight = entry->abstraction_level;

  // Combine weights (could be adjusted based on needs)
  return recency_weight * 0.6f + importance_weight * 0.4f;
}

void extractSemanticFeatures(
    float *memory_vector, float *features,
    float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE]) {
  // Initialize feature projection matrix
  // Apply dimensionality reduction
  for (unsigned int i = 0; i < FEATURE_VECTOR_SIZE; i++) {
    features[i] = 0;
    for (unsigned int j = 0; j < MEMORY_VECTOR_SIZE; j++) {
      features[i] += memory_vector[j] * feature_projection_matrix[i][j];
    }
  }

  // Apply non-linear activation
  for (unsigned int i = 0; i < FEATURE_VECTOR_SIZE; i++) {
    features[i] = tanh(features[i]);
  }
}

void updateContext(WorkingMemorySystem *system) {
  // Reset global context
  memset(system->global_context, 0, CONTEXT_VECTOR_SIZE * sizeof(float));

  // Incorporate focused attention
  for (unsigned int i = 0; i < system->focus.size; i++) {
    float attention_weight = computeAttentionWeight(&system->focus.entries[i]);
    addWeightedVector(system->global_context,
                      system->focus.entries[i].context_vector, attention_weight,
                      CONTEXT_VECTOR_SIZE);
  }

  // Incorporate active memory with decay
  for (unsigned int i = 0; i < system->active.size; i++) {
    float activation = computeActivation(&system->active.entries[i]);
    addWeightedVector(
        system->global_context, system->active.entries[i].context_vector,
        activation * system->active.activation_decay, CONTEXT_VECTOR_SIZE);
  }

  // Normalize global context
  normalizeVector(system->global_context, CONTEXT_VECTOR_SIZE);
}

void addMemory(
    MemorySystem *system, WorkingMemorySystem *working_memory, Neuron *neurons,
    float *input_tensor, unsigned int timestamp,
    float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE]) {
  // Create memory entry
  MemoryEntry entry;
  computeMemoryVector(entry.vector, neurons, input_tensor);
  entry.importance = computeImportance(entry.vector);
  entry.timestamp = timestamp;
  WorkingMemoryEntry enhanced;

  // Handle Working Memory System first
  if (entry.importance > working_memory->focus.attention_threshold) {
    // Add to focused attention
    if (working_memory->focus.size < working_memory->focus.capacity) {
      enhanced.features = malloc(FEATURE_VECTOR_SIZE * sizeof(float));
      extractSemanticFeatures(entry.vector, enhanced.features,
                              feature_projection_matrix);
      enhanced.context_vector = malloc(CONTEXT_VECTOR_SIZE * sizeof(float));
      memcpy(enhanced.context_vector, working_memory->global_context,
             CONTEXT_VECTOR_SIZE * sizeof(float));
      working_memory->focus.entries[working_memory->focus.size++] = enhanced;
      updateSemanticClusters(working_memory, &enhanced);
    }
  } else {
    // Add to active memory
    if (working_memory->active.size < working_memory->active.capacity) {
      enhanced.features = malloc(FEATURE_VECTOR_SIZE * sizeof(float));
      extractSemanticFeatures(entry.vector, enhanced.features,
                              feature_projection_matrix);
      enhanced.context_vector = malloc(CONTEXT_VECTOR_SIZE * sizeof(float));
      memcpy(enhanced.context_vector, working_memory->global_context,
             CONTEXT_VECTOR_SIZE * sizeof(float));
      working_memory->active.entries[working_memory->active.size++] = enhanced;
      updateSemanticClusters(working_memory, &enhanced);
    }
  }

  // Update global context
  updateContext(working_memory);

  // Then handle original hierarchical storage
  if (entry.importance >= system->hierarchy.long_term.importance_threshold) {
    if (system->hierarchy.long_term.size <
        system->hierarchy.long_term.capacity) {
      system->hierarchy.long_term.entries[system->hierarchy.long_term.size++] =
          entry;
    } else {
      int least_important_idx =
          findLeastImportantMemory(system->hierarchy.long_term.entries,
                                   system->hierarchy.long_term.size);
      system->hierarchy.long_term.entries[least_important_idx] = entry;
    }
  } else if (entry.importance >=
             system->hierarchy.medium_term.importance_threshold) {
    if (system->hierarchy.medium_term.size <
        system->hierarchy.medium_term.capacity) {
      system->hierarchy.medium_term
          .entries[system->hierarchy.medium_term.size++] = entry;
    } else {
      consolidateToHigherLevel(system);
    }
  } else {
    if (system->hierarchy.short_term.size <
        system->hierarchy.short_term.capacity) {
      system->hierarchy.short_term
          .entries[system->hierarchy.short_term.size++] = entry;
    } else {
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

void consolidateToLongTermMemory(WorkingMemorySystem *working_memory,
                                 MemorySystem *memorySystem,
                                 unsigned int step) {
  // Validate input parameters
  if (!working_memory || !memorySystem) {
    return;
  }

  // Only process at specified intervals
  if (step % 10 != 0) {
    return;
  }

  // Process items in working memory focus
  for (unsigned int i = 0; i < working_memory->focus.size; i++) {
    WorkingMemoryEntry *enhanced_entry = &working_memory->focus.entries[i];

    // Skip invalid entries
    if (!enhanced_entry || !enhanced_entry->features ||
        !enhanced_entry->context_vector) {
      continue;
    }

    // Create new memory entry
    MemoryEntry new_entry;
    memset(&new_entry, 0, sizeof(MemoryEntry));
    new_entry.timestamp = step;

    // Calculate memory vector based on features and context
    float combined_importance = 0.0f;
    for (unsigned int j = 0; j < MEMORY_VECTOR_SIZE; j++) {
      // Initialize vector element
      new_entry.vector[j] = 0.0f;

      // Combine features and context if within bounds
      if (j < FEATURE_VECTOR_SIZE) {
        float feature_val = enhanced_entry->features[j];
        float context_val = (j < CONTEXT_VECTOR_SIZE)
                                ? enhanced_entry->context_vector[j]
                                : 0.0f;

        // Only use valid values
        if (!isnan(feature_val) && !isinf(feature_val) && !isnan(context_val) &&
            !isinf(context_val)) {
          // Weight features more heavily than context
          new_entry.vector[j] = (feature_val * 0.7f + context_val * 0.3f);
          combined_importance += fabs(new_entry.vector[j]);
        }
      }
    }

    // Normalize the vector if it has non-zero values
    if (combined_importance > 0.0f) {
      for (unsigned int j = 0; j < MEMORY_VECTOR_SIZE; j++) {
        new_entry.vector[j] /= combined_importance;
      }
    }

    // Calculate importance based on abstraction level and depth
    float depth_factor =
        expf(-enhanced_entry->depth * 0.1f); // Decay with depth
    float abstraction = enhanced_entry->abstraction_level;
    if (isnan(abstraction) || isinf(abstraction)) {
      abstraction = 0.5f;
    }

    // Combine factors for final importance
    new_entry.importance = (abstraction * 0.6f + depth_factor * 0.4f);
    new_entry.importance = fmaxf(0.0f, fminf(1.0f, new_entry.importance));

    // Verify the entry is valid
    bool valid_entry = true;
    for (unsigned int j = 0; j < MEMORY_VECTOR_SIZE; j++) {
      if (isnan(new_entry.vector[j]) || isinf(new_entry.vector[j])) {
        valid_entry = false;
        break;
      }
    }

    // Store if valid and meets importance threshold
    if (valid_entry &&
        new_entry.importance >
            memorySystem->hierarchy.long_term.importance_threshold &&
        memorySystem->hierarchy.long_term.size <
            memorySystem->hierarchy.long_term.capacity) {

      memorySystem->hierarchy.long_term
          .entries[memorySystem->hierarchy.long_term.size] = new_entry;
      memorySystem->hierarchy.long_term.size++;
    }
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

const float letter_weights[26] = {1.0f,  0.9f,  0.8f, 0.85f, 0.95f, 0.75f, 0.7f,
                                  0.8f,  0.9f,  0.6f, 0.7f,  0.85f, 0.75f, 0.9f,
                                  1.0f,  0.65f, 0.6f, 0.85f, 0.95f, 0.8f,  0.7f,
                                  0.65f, 0.75f, 0.6f, 0.7f,  0.6f};

const int vocab_size = sizeof(vocabulary) / sizeof(vocabulary[0]);

void swap(char *a, char *b) {
  char temp = *a;
  *a = *b;
  *b = temp;
}

const char *mapToWord(float value) {
  int index = (int)(fabs(value) * vocab_size) % vocab_size;

  if (value > 1.0f || value < 0.0f) {
    static char customWord[64];
    int wordLength = (int)(fabs(value) * 10) % 8 + 3; // Length between 3-10

    unsigned int seed = (unsigned int)(fabs(value) * 1000000);
    srand(seed); // Pseudo-random, but value-based

    for (int i = 0; i < wordLength; i++) {
      float weightSum = 0;
      for (int j = 0; j < 26; j++)
        weightSum += letter_weights[j];

      float randomValue = ((float)rand() / RAND_MAX) * weightSum;
      float currentSum = 0;
      int selectedLetter = -1;

      for (int j = 0; j < 26; j++) {
        currentSum += letter_weights[j];
        if (randomValue <= currentSum) {
          selectedLetter = j;
          break;
        }
      }

      // If letter selection failed, default to a random one
      if (selectedLetter == -1) {
        selectedLetter = rand() % 26;
      }

      customWord[i] = 'a' + selectedLetter;
    }

    // Shuffle the first letter to avoid predictable 'a' starts
    if (wordLength > 1) {
      int swapIdx = rand() % wordLength;
      swap(&customWord[0], &customWord[swapIdx]);
    }

    customWord[wordLength] = '\0';
    return customWord;
  }

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

// Function to calculate letter-based weight for a word
float computeLetterWeight(const char *word) {
  float weight_sum = 0.0f;
  int length = strlen(word);
  for (int i = 0; i < length; i++) {
    if (word[i] >= 'a' && word[i] <= 'z') {
      weight_sum += letter_weights[word[i] - 'a'];
    } else if (word[i] >= 'A' && word[i] <= 'Z') {
      weight_sum += letter_weights[word[i] - 'A'];
    }
  }
  return (length > 0) ? (weight_sum / length) : 0.0f; // Normalize by length
}

// Initialize vocabulary letter weights
void initializeVocabularyWeights() {
  for (int i = 0; i < vocab_size; i++) {
    ((VocabularyEntry *)&vocabulary[i])->letter_weight =
        computeLetterWeight(vocabulary[i].word);
  }
}

float embeddings[vocab_size][EMBEDDING_SIZE];

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
            (vocabulary[i].semantic_weight + vocabulary[i].letter_weight) *
            (1.0f + complexity_factor);
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

// Function to apply element-wise tanh using SIMD
simd_float4 simd_tanh(simd_float4 input) {
  simd_float4 result;
  for (int i = 0; i < 4; i++) {
    result[i] = tanhf(input[i]);
  }
  return result;
}

simd_float4 simd_add(simd_float4 a, simd_float4 b) {
  return a + b; // SIMD supports operator overloading for addition
}

// Helper function for element-wise multiplication of simd_float4
simd_float4 simd_mul(simd_float4 a, simd_float4 b) {
  simd_float4 result;
  for (int i = 0; i < 4; i++) {
    result[i] = a[i] * b[i];
  }
  return result;
}

// Function to update neuron states using SIMD
void updateNeuronStates(Neuron *neurons, int num_neurons,
                        float *recurrent_weights, simd_float4 scaled_factor) {
  // Compute attention weights between neurons
  float **attention_weights = malloc(num_neurons * sizeof(float *));
  for (int i = 0; i < num_neurons; i++) {
    attention_weights[i] = malloc(num_neurons * sizeof(float));

    // Simplified dot product attention
    for (int j = 0; j < num_neurons; j++) {
      attention_weights[i][j] = neurons[i].output * neurons[j].output;
    }

    // Softmax normalization
    float row_sum = 0;
    for (int j = 0; j < num_neurons; j++) {
      attention_weights[i][j] = exp(attention_weights[i][j]);
      row_sum += attention_weights[i][j];
    }

    for (int j = 0; j < num_neurons; j++) {
      attention_weights[i][j] /= row_sum;
    }
  }

  // Process neurons in groups of 4
  for (int i = 0; i < num_neurons; i += 4) {
    int remaining = num_neurons - i;
    int group_size = (remaining < 4) ? remaining : 4;

    simd_float4 current_outputs = {0};
    simd_float4 current_states = {0};
    simd_float4 current_weights = {0};

    for (int j = 0; j < group_size; j++) {
      current_outputs[j] = neurons[i + j].output;
      current_states[j] = neurons[i + j].state;
      current_weights[j] = recurrent_weights[i + j];
    }

    // Update states with decay factor
    simd_float4 new_states = simd_mul(current_states, 0.8f);

    // Compute attended inputs
    simd_float4 attended_inputs = {0};
    for (int j = 0; j < group_size; j++) {
      float attended_value = 0;
      for (int k = 0; k < num_neurons; k++) {
        attended_value += attention_weights[i + j][k] * neurons[k].output;
      }
      attended_inputs[j] = attended_value;
    }

    // Recurrent and attention-based inputs
    simd_float4 recurrent_inputs = simd_mul(current_outputs, current_weights);
    new_states = simd_add(new_states, simd_mul(recurrent_inputs, 0.3f));
    new_states = simd_add(new_states, simd_mul(attended_inputs, 0.2f));

    // Apply activation and scaling
    simd_float4 new_outputs = simd_tanh(new_states * scaled_factor);

    // Store updated values
    for (int j = 0; j < group_size; j++) {
      neurons[i + j].state = new_states[j];
      neurons[i + j].output = new_outputs[j];
    }
  }

  // Free attention weights
  for (int i = 0; i < num_neurons; i++) {
    free(attention_weights[i]);
  }
  free(attention_weights);
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
  // Compute attention weights between neurons
  float **attention_weights = malloc(num_neurons * sizeof(float *));
  for (int i = 0; i < num_neurons; i++) {
    attention_weights[i] = malloc(num_neurons * sizeof(float));

    // Simplified dot product attention
    for (int j = 0; j < num_neurons; j++) {
      attention_weights[i][j] = neurons[i].state * neurons[j].state;
    }

    // Softmax normalization
    float row_sum = 0;
    for (int j = 0; j < num_neurons; j++) {
      attention_weights[i][j] = exp(attention_weights[i][j]);
      row_sum += attention_weights[i][j];
    }

    for (int j = 0; j < num_neurons; j++) {
      attention_weights[i][j] /= row_sum;
    }
  }

  // Process neurons in groups of 4
  for (int i = 0; i < num_neurons; i += 4) {
    // Ensure we don't overrun the array
    int remaining = num_neurons - i;
    int group_size = (remaining < 4) ? remaining : 4;

    simd_float4 weighted_sum = simd_make_float4(0, 0, 0, 0);
    simd_float4 attended_sum = simd_make_float4(0, 0, 0, 0);

    // Compute weighted sum for connections
    for (int j = 0; j < max_connections; j++) {
      simd_float4 weight_vector = {0};
      simd_float4 target_state = {0};

      for (int k = 0; k < group_size; k++) {
        int neuron_idx = i + k;
        int connection_idx = connections[neuron_idx * max_connections + j];
        weight_vector[k] = weights[neuron_idx * max_connections + j];
        target_state[k] = neurons[connection_idx].state;
      }

      // Fill remaining SIMD lanes with zeros
      for (int k = group_size; k < 4; k++) {
        weight_vector[k] = 0.0f;
        target_state[k] = 0.0f;
      }

      weighted_sum =
          simd_add(weighted_sum, simd_mul(weight_vector, target_state));
    }

    // Compute self-attention based sum
    for (int j = 0; j < group_size; j++) {
      float attended_value = 0;
      for (int k = 0; k < num_neurons; k++) {
        attended_value += attention_weights[i + j][k] * neurons[k].state;
      }
      attended_sum[j] = attended_value;
    }

    // Combine current states with weighted and attended sums
    simd_float4 current_states = {0};
    for (int k = 0; k < group_size; k++) {
      current_states[k] = neurons[i + k].state;
    }

    // Fill remaining SIMD lanes with zeros
    for (int k = group_size; k < 4; k++) {
      current_states[k] = 0.0f;
    }

    simd_float4 new_states =
        simd_add(simd_mul(current_states, 0.8f),        // Decay current states
                 simd_add(simd_mul(weighted_sum, 0.2f), // Add weighted sum
                          simd_mul(attended_sum, 0.1f)  // Add attended sum
                          ));

    // Update neuron states and outputs
    for (int k = 0; k < group_size; k++) {
      neurons[i + k].state = new_states[k];
      neurons[i + k].output = tanh(new_states[k] * scaled_factor);
    }
  }

  // Free attention weights
  for (int i = 0; i < num_neurons; i++) {
    free(attention_weights[i]);
  }
  free(attention_weights);
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

float computeAbstractionLevel(float output, float error) {
  // Higher abstraction for:
  // - Moderate activation (not too low or high)
  // - Lower error (more reliable/stable representations)

  // Normalize output to 0-1 range if not already
  float normalized_output = fabs(output);
  if (normalized_output > 1.0f) {
    normalized_output = 1.0f;
  }

  // Calculate how close the output is to optimal moderate activation (0.5)
  // Result is 1.0 at 0.5, decreasing towards 0.0 at extremes
  float activation_factor = 1.0f - fabs(normalized_output - 0.5f) * 2.0f;

  // Error factor decreases as error increases
  float error_factor = 1.0f - error;
  if (error_factor < 0.0f) {
    error_factor = 0.0f;
  }

  // Combine factors - higher values indicate higher abstraction
  // Weight activation_factor more heavily (0.7) than error_factor (0.3)
  float abstraction_level = (activation_factor * 0.7f) + (error_factor * 0.3f);

  return abstraction_level;
}

void updateWorkingMemory(WorkingMemorySystem *working_memory, Neuron *neurons,
                         float *input_tensor, float *target_outputs,
                         unsigned int step) {
  // Update focused attention based on prediction errors
  float total_error = 0.0f;
  for (unsigned int i = 0; i < working_memory->focus.capacity; i++) {
    if (neurons[i].output != 0) { // Only consider active neurons
      float error = fabs(neurons[i].output - target_outputs[i]);
      total_error += error;

      // High error items get promoted to focused attention
      if (error > working_memory->focus.attention_threshold) {
        if (working_memory->focus.size < working_memory->focus.capacity) {
          WorkingMemoryEntry *entry =
              &working_memory->focus.entries[working_memory->focus.size++];
          entry->features = malloc(FEATURE_VECTOR_SIZE * sizeof(float));
          entry->context_vector = malloc(CONTEXT_VECTOR_SIZE * sizeof(float));
          entry->depth = 0;
          entry->abstraction_level =
              computeAbstractionLevel(neurons[i].output, error);
        }
      }
    }
  }

  // Decay activation levels in active memory
  for (unsigned int i = 0; i < working_memory->active.size; i++) {
    float activation = computeActivation(&working_memory->active.entries[i]);
    activation *= working_memory->active.activation_decay;

    // Remove items that fall below threshold
    if (activation < 0.1f) {
      // Shift remaining entries
      memmove(&working_memory->active.entries[i],
              &working_memory->active.entries[i + 1],
              (working_memory->active.size - i - 1) *
                  sizeof(WorkingMemoryEntry));
      working_memory->active.size--;
      i--; // Adjust index after removal
    }
  }

  // Update semantic clusters
  updateSemanticClusters(working_memory, &working_memory->focus.entries[0]);

  // Update global context
  updateContext(working_memory);
}

void integrateWorkingMemory(WorkingMemorySystem *working_memory,
                            Neuron *neurons, float *input_tensor,
                            float *target_outputs, float *weights,
                            unsigned int step) {
  // Update working memory state
  updateWorkingMemory(working_memory, neurons, input_tensor, target_outputs,
                      step);

  // Apply working memory influence to network processing
  for (unsigned int i = 0; i < working_memory->focus.size; i++) {
    WorkingMemoryEntry *focused_item = &working_memory->focus.entries[i];
    float attention_weight = computeAttentionWeight(focused_item);

    if (!isfinite(attention_weight)) {
      printf("Invalid attention weight in focus[%u], skipping\n", i);
      continue;
    }

    // Modulate neuron activity based on focused items
    for (unsigned int j = 0; j < FEATURE_VECTOR_SIZE; j++) {
      unsigned int neuron_idx = j % MAX_NEURONS;
      if (!isfinite(focused_item->features[j])) {
        printf("NaN in focused_item->features[%u], setting to 0\n", j);
        focused_item->features[j] = 0.0f;
      }
      // Clamp the feature value to avoid overflow
      focused_item->features[j] = clampValue(focused_item->features[j]);
      neurons[neuron_idx].state += focused_item->features[j] * attention_weight;
    }
  }

  // Apply active memory influence
  for (unsigned int i = 0; i < working_memory->active.size; i++) {
    WorkingMemoryEntry *active_item = &working_memory->active.entries[i];
    float activation = computeActivation(active_item);

    if (!isfinite(activation)) {
      printf("Invalid activation in active[%u], skipping\n", i);
      continue;
    }

    // Modulate network weights based on active memory
    for (unsigned int j = 0; j < FEATURE_VECTOR_SIZE; j++) {
      unsigned int weight_idx = j % (MAX_NEURONS * MAX_CONNECTIONS);
      if (!isfinite(active_item->features[j])) {
        printf("NaN in active_item->features[%u], setting to 0\n", j);
        active_item->features[j] = 0.0f;
      }
      // Clamp the feature value to avoid overflow
      active_item->features[j] = clampValue(active_item->features[j]);
      weights[weight_idx] *=
          (1.0f + active_item->features[j] * activation * 0.1f);
    }
  }

  // Apply sanitized semantic cluster influence
  for (unsigned int i = 0; i < working_memory->clusters.num_clusters; i++) {
    SemanticCluster *cluster = &working_memory->clusters.clusters[i];

    if (!isfinite(cluster->coherence)) {
      printf("Skipping cluster[%u] due to invalid coherence\n", i);
      continue;
    }

    // Clamp coherence to a safe range
    cluster->coherence = clampValue(cluster->coherence);

    if (cluster->coherence > 0.7f) { // Only use highly coherent clusters
      for (unsigned int j = 0; j < FEATURE_VECTOR_SIZE; j++) {
        unsigned int neuron_idx = j % MAX_NEURONS;

        // Check for NaN and clamp the feature values
        if (!isfinite(cluster->vector[j])) {
          printf("NaN in cluster->vector[%u] of cluster[%u], setting to 0\n", j,
                 i);
          cluster->vector[j] = 0.0f;
        }

        // Clamp the vector value to avoid overflow
        cluster->vector[j] = clampValue(cluster->vector[j]);
        neurons[neuron_idx].state +=
            cluster->vector[j] * cluster->coherence * 0.05f;
      }
    }
  }
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

void updateNeuronsOnCPU(Neuron *neurons, const float *weights,
                        const int *connections, int max_neurons,
                        int max_connections, const float *input_tensor,
                        int input_size, const float *recurrent_weights) {
  for (int id = 0; id < max_neurons; id++) {
    float current_state = neurons[id].state;
    float current_output = neurons[id].output;
    int num_conn = neurons[id].num_connections;
    int layer = neurons[id].layer_id;

    // Calculate weighted sum
    float weighted_sum = 0.0f;
    for (int i = 0; i < num_conn; i++) {
      int conn_idx = id * max_connections + i;
      int target = connections[conn_idx];

      float depth_scale = 1.0f / (1.0f + layer);
      float connection_strength = weights[conn_idx] * depth_scale;

      weighted_sum += neurons[target].state * connection_strength * 0.6f +
                      neurons[target].output * connection_strength * 0.4f;
    }

    float input_influence = input_tensor[id % input_size];
    float temporal_factor = 1.0f / (1.0f + id % 4);

    float new_state = current_state * DECAY_RATE +
                      weighted_sum * CONNECTION_WEIGHT +
                      input_influence * INPUT_WEIGHT * temporal_factor;

    float recurrent_influence = current_output * recurrent_weights[id];
    new_state += recurrent_influence * 0.15f;

    // Activation function
    float dynamic_scale =
        ACTIVATION_SCALE * (1.0f + 0.1f * sin(input_influence * M_PI));
    float new_output = tanh(new_state * dynamic_scale + ACTIVATION_BIAS);

    // Add small random noise
    float random_val = ((float)rand() / RAND_MAX) * 0.01f;
    new_output += random_val;

    // Clamp output
    new_output = fmax(-1.0f, fmin(1.0f, new_output));

    neurons[id].state = new_state;
    neurons[id].output = new_output;
  }
}

void updateWeightsOnCPU(float *weights, const Neuron *neurons,
                        const int *connections, const float learning_rate,
                        const int max_neurons, const int max_connections) {
  const float momentum = 0.9f;

  for (int id = 0; id < max_neurons * max_connections; id++) {
    int neuron_idx = id / max_connections;
    int conn_idx = id % max_connections;

    if (conn_idx >= neurons[neuron_idx].num_connections)
      continue;

    int target_idx = connections[id];

    // Modified Hebbian learning with normalization
    float pre_activation = neurons[neuron_idx].state;
    float post_activation = neurons[target_idx].output;
    float current_weight = weights[id];

    // Calculate weight update
    float hebbian_term = pre_activation * post_activation;
    float normalization_term = current_weight * WEIGHT_DECAY;
    float delta_w = learning_rate * (hebbian_term - normalization_term);

    // Update weight with momentum
    float new_weight = current_weight + delta_w;
    new_weight = momentum * current_weight + (1.0f - momentum) * new_weight;

    // Clip weights
    weights[id] = fmin(fmax(new_weight, MIN_WEIGHT), MAX_WEIGHT);
  }
}

void backpropagationOnCPU(const Neuron *neurons, float *weights,
                          const int *connections, const int max_neurons,
                          const int max_connections,
                          const float *target_outputs, float *output_errors,
                          const float learning_rate) {
  for (int id = 0; id < max_neurons; id++) {
    // Compute output error for the current neuron
    float predicted_output = neurons[id].output;
    float target_output = target_outputs[id];

    // Compute the error between predicted and target output
    float error = predicted_output - target_output;
    output_errors[id] = error;

    float activation_gradient = predicted_output * (1.0f - predicted_output);

    // Calculate weight updates for all connections to this neuron
    for (int i = 0; i < max_connections; i++) {
      int conn_idx = id * max_connections + i;
      int connected_neuron = connections[conn_idx];

      if (connected_neuron >= max_neurons)
        continue;

      // Calculate gradient for this weight (backpropagation)
      float input_gradient =
          neurons[connected_neuron].output * error * activation_gradient;

      // Update weight using learning rate (gradient descent)
      weights[conn_idx] -= learning_rate * input_gradient;
    }
  }
}

void reverseProcessOnCPU(Neuron *neurons, float *reverse_weights,
                         int *reverse_connections, int max_neurons,
                         int max_connections) {
  for (int id = 0; id < max_neurons; id++) {
    float sum = 0.0f;
    for (int c = 0; c < max_connections; c++) {
      int conn_idx = id * max_connections + c;
      int source_neuron = reverse_connections[conn_idx];
      sum += neurons[source_neuron].output * reverse_weights[conn_idx];
    }
    neurons[id].state += 0.1f * sum;
  }
}

void memoryReplayOnCPU(Neuron *neurons, float *weights, int *connections,
                       MemoryEntry *memories, int memory_capacity,
                       int max_neurons, int max_connections) {
  for (int id = 0; id < memory_capacity; id++) {
    MemoryEntry memory = memories[id];

    for (int i = 0; i < max_neurons; i++) {
      for (int j = 0; j < max_connections; j++) {
        int conn_idx = i * max_connections + j;
        int target_neuron = connections[conn_idx];

        float weight_delta = 0.01f * memory.importance * neurons[i].output *
                             memory.vector[target_neuron];
        weights[conn_idx] += weight_delta;
      }
    }
  }
}

float activation_function(float x, float scale, float bias) {
  // Sigmoid activation with scaling and bias
  return 1.0f / (1.0f + expf(-(x * scale + bias)));
}

float generate_random(int seed, float value) {
  float x = sin(seed * 12.9898f + value * 78.233f) * 43758.5453f;
  return x - floor(x);
}

void processNeuronsOnCPU(Neuron *neurons, const float *weights,
                         const int *connections, const int max_neurons,
                         const int max_connections, const float *input_tensor,
                         const int input_size) {
  for (int id = 0; id < max_neurons; id++) {
    float current_state = neurons[id].state;
    float current_output = neurons[id].output;
    int num_conn = neurons[id].num_connections;
    int layer = neurons[id].layer_id;

    // Calculate weighted sum
    float weighted_sum = 0.0f;
    for (int i = 0; i < num_conn; i++) {
      int conn_idx = id * max_connections + i;
      int target = connections[conn_idx];

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

    // Dynamic activation
    float dynamic_scale =
        ACTIVATION_SCALE * (1.0f + 0.1f * sinf(input_influence * M_PI));
    float new_output =
        activation_function(new_state, dynamic_scale, ACTIVATION_BIAS);

    // Add controlled randomization
    float random_val = generate_random(id, new_state);
    new_output += random_val * 0.01f;

    // Clamp output
    new_output = fminf(fmaxf(new_output, MIN_ACTIVATION), MAX_ACTIVATION);

    // Store results
    neurons[id].state = new_state;
    neurons[id].output = new_output;
  }
}

ContextNode *createContextNode(const char *name, int vector_size,
                               ContextNode *parent) {
  ContextNode *node = (ContextNode *)malloc(sizeof(ContextNode));
  node->name = strdup(name);
  node->importance = 1.0f;
  node->state_vector = (float *)calloc(vector_size, sizeof(float));
  node->vector_size = vector_size;
  node->children = (ContextNode **)malloc(sizeof(ContextNode *) * 10);
  node->num_children = 0;
  node->max_children = 10;
  node->parent = parent;
  node->temporal_relevance = 1.0f;
  node->last_updated = time(NULL);
  return node;
}

GlobalContextManager *initializeGlobalContextManager(int vector_size) {
  GlobalContextManager *manager =
      (GlobalContextManager *)malloc(sizeof(GlobalContextManager));
  manager->vector_size = vector_size;
  manager->total_nodes = 1;
  manager->decay_rate = 0.95f;
  manager->update_threshold = 0.1f;
  manager->max_depth = 5;
  manager->max_children_per_node = 10;

  // Initialize global context vector
  manager->global_context_vector = (float *)calloc(vector_size, sizeof(float));

  // Create root node
  manager->root = createContextNode("Global", vector_size, NULL);

  // Initialize default context hierarchy
  ContextNode *goals = createContextNode("Goals", vector_size, manager->root);
  ContextNode *constraints =
      createContextNode("Constraints", vector_size, manager->root);
  ContextNode *environment =
      createContextNode("Environment", vector_size, manager->root);

  manager->root->children[0] = goals;
  manager->root->children[1] = constraints;
  manager->root->children[2] = environment;
  manager->root->num_children = 3;

  return manager;
}

void updateContextNode(ContextNode *node, float *new_state, float importance) {
  for (int i = 0; i < node->vector_size; i++) {
    node->state_vector[i] =
        node->state_vector[i] * (1 - importance) + new_state[i] * importance;
  }
  node->last_updated = time(NULL);
  node->importance = fmax(node->importance * 0.95f + importance * 0.05f, 0.1f);
}

void propagateContextUpdates(ContextNode *node) {
  if (node->parent != NULL) {
    float *aggregated = (float *)calloc(node->vector_size, sizeof(float));
    float total_importance = 0.0f;

    // Aggregate child states weighted by importance
    for (int i = 0; i < node->parent->num_children; i++) {
      ContextNode *sibling = node->parent->children[i];
      float temp_relevance =
          expf(-(time(NULL) - sibling->last_updated) / 3600.0f);
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

ContextNode *findContextNode(ContextNode *root, const char *name) {
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

ContextNode *addContextNode(GlobalContextManager *manager, const char *name,
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

  // Enhanced constraint evaluation
  if (strcmp(constraint->name, "ActivityLevel") == 0) {
    float total_activity = 0;
    float variance = 0.0f;

    // Compute total activity and mean
    for (int i = 0; i < num_neurons; i++) {
      total_activity += neurons[i].output;
    }
    float mean_activity = total_activity / num_neurons;

    // Compute variance
    for (int i = 0; i < num_neurons; i++) {
      float diff = neurons[i].output - mean_activity;
      variance += diff * diff;
    }
    variance /= num_neurons;

    // Satisfaction based on activity level and variance
    satisfaction = 1.0f - fabs(0.5f - (total_activity / num_neurons));
    satisfaction *= (1.0f - (variance / 0.25f)); // Penalize high variance
  }

  return fmaxf(0.0f, fminf(1.0f, satisfaction));
}

void updateGlobalContext(GlobalContextManager *manager, Neuron *neurons,
                         int num_neurons, float *input_tensor) {
  // Extract relevant features from current network state
  float *current_context = (float *)calloc(manager->vector_size, sizeof(float));

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
    float *constraint_state =
        (float *)calloc(manager->vector_size, sizeof(float));
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
  // Compute network-wide entropy as a measure of state variability
  float total_entropy = 0.0f;
  for (int i = 0; i < num_neurons; i++) {
    total_entropy += fabs(neurons[i].output);
  }
  float network_entropy = total_entropy / num_neurons;
  float context_sensitivity = 1.0f - network_entropy;

  // Modulate neuron behavior based on global context
  for (int i = 0; i < num_neurons; i++) {
    float context_influence = 0;

    // Compute context influence on this neuron
    for (int j = 0; j < manager->vector_size && j < num_neurons; j++) {
      context_influence +=
          manager->global_context_vector[j] * weights[i * max_connections + j];
    }

    // Adaptive context modulation with sensitivity
    float modulation_factor = 0.1f * context_influence * context_sensitivity;
    neurons[i].state = neurons[i].state * (1.0f + modulation_factor);
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

void updateCorrelationMatrix(float *correlation_matrix, float *input_history,
                             float *outcomes, int history_length,
                             int input_size) {
  for (int i = 0; i < input_size; i++) {
    for (int j = 0; j < input_size; j++) {
      float correlation = 0.0f;
      for (int h = 0; h < history_length; h++) {
        correlation += input_history[h * input_size + i] *
                       input_history[h * input_size + j] * outcomes[h];
      }
      correlation_matrix[i * input_size + j] = correlation / history_length;
    }
  }
}

float computeFeedbackSignal(float current_outcome, float *feedback_history,
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

void applyDynamicContext(Neuron *neurons, float *context_weights,
                         GlobalContextManager *context, int size) {
  for (int i = 0; i < size; i++) {
    neurons[i].state = neurons[i].state * (1.0f - context_weights[i]) +
                       context->global_context_vector[i] * context_weights[i];
  }
}

float computeAverageFeedback(float *feedback_history, int history_size) {
  float sum = 0.0f;
  int valid_count = 0;

  for (int i = 0; i < history_size; i++) {
    if (feedback_history[i] != 0.0f) { // Only count non-zero feedback
      sum += feedback_history[i];
      valid_count++;
    }
  }

  return valid_count > 0 ? sum / valid_count : 0.0f;
}

float computeMinWeight(float *weights, int size) {
  if (size <= 0 || weights == NULL)
    return 0.0f;

  float min_weight = weights[0];
  for (int i = 1; i < size; i++) {
    if (weights[i] < min_weight) {
      min_weight = weights[i];
    }
  }

  return min_weight;
}

float computeMaxWeight(float *weights, int size) {
  if (size <= 0 || weights == NULL)
    return 0.0f;

  float max_weight = weights[0];
  for (int i = 1; i < size; i++) {
    if (weights[i] > max_weight) {
      max_weight = weights[i];
    }
  }

  return max_weight;
}

float computeAverageCorrelation(float *correlation_matrix, int size) {
  if (size <= 0 || correlation_matrix == NULL)
    return 0.0f;

  float sum = 0.0f;
  int count = 0;

  // Compute average of absolute correlation values
  // Skip diagonal elements (self-correlation)
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (i != j) { // Skip diagonal
        sum += fabs(correlation_matrix[i * size + j]);
        count++;
      }
    }
  }

  return count > 0 ? sum / count : 0.0f;
}

// Add after the existing MetaController initialization
IntrinsicMotivation *initializeMotivationSystem() {
  IntrinsicMotivation *motivation = malloc(sizeof(IntrinsicMotivation));
  motivation->novelty_score = 0.5f;
  motivation->competence_score = 0.0f;
  motivation->autonomy_score = 0.5f;
  motivation->mastery_level = 0.0f;
  motivation->curiosity_drive = 1.0f;
  motivation->achievement_drive = 0.8f;
  motivation->exploration_rate = 0.3f;
  return motivation;
}

GoalSystem *initializeGoalSystem(int capacity) {
  GoalSystem *system = malloc(sizeof(GoalSystem));
  system->goals = malloc(sizeof(Goal) * capacity);
  system->num_goals = 0;
  system->capacity = capacity;
  system->planning_horizon = 10.0f;
  system->discount_factor = 0.95f;
  return system;
}

void updateMotivationSystem(IntrinsicMotivation *motivation,
                            float performance_delta, float novelty,
                            float task_difficulty) {
  // Update competence based on performance
  motivation->competence_score =
      fmin(1.0f, motivation->competence_score + 0.1f * performance_delta);

  // Update novelty score
  motivation->novelty_score = novelty;

  // Adjust curiosity based on competence and novelty
  motivation->curiosity_drive =
      fmin(1.0f, motivation->curiosity_drive +
                     0.05f * (novelty - motivation->competence_score));

  // Update mastery level
  float mastery_delta = performance_delta *
                        (task_difficulty / (motivation->mastery_level + 0.1f));
  motivation->mastery_level =
      fmin(1.0f, motivation->mastery_level + 0.1f * mastery_delta);

  // Adjust exploration rate based on curiosity
  motivation->exploration_rate = 0.1f + 0.4f * motivation->curiosity_drive;
}

void addGoal(GoalSystem *system, const char *description, float priority) {
  if (system->num_goals >= system->capacity)
    return;

  Goal *goal = &system->goals[system->num_goals++];
  strncpy(goal->description, description, 255);
  goal->priority = priority;
  goal->progress = 0.0f;
  goal->reward_value = priority * 10.0f;
  goal->achieved = false;
  goal->timestamp = time(NULL);
}

float evaluateGoalProgress(Goal *goal, const Neuron *neurons,
                           const float *target_outputs) {
  if (strstr(goal->description, "Minimize prediction error")) {
    float total_error = 0.0f;
    for (int i = 0; i < MAX_NEURONS; i++) {
      float error = fabs(neurons[i].output - target_outputs[i]);
      total_error += error;
    }
    return 1.0f - fmin(1.0f, total_error / MAX_NEURONS);
  }

  if (strstr(goal->description, "Develop stable representations")) {
    float stability = 0.0f;

    // Check activation stability through connected neurons
    for (int i = 0; i < MAX_NEURONS; i++) {
      float connected_outputs = 0.0f;
      for (int j = 0; j < neurons[i].num_connections; j++) {
        connected_outputs += neurons[j].output;
      }
      float avg_connected = neurons[i].num_connections > 0
                                ? connected_outputs / neurons[i].num_connections
                                : 0;

      // Higher stability when neuron output aligns with connected neurons
      stability += 1.0f - fabs(neurons[i].output - avg_connected);
    }
    return stability / MAX_NEURONS;
  }

  if (strstr(goal->description, "Maximize information gain")) {
    float entropy = 0.0f;

    // Calculate output entropy
    for (int i = 0; i < MAX_NEURONS; i++) {
      if (neurons[i].output > 0.0f) {
        entropy -= neurons[i].output * log2f(neurons[i].output);
      }
    }
    return fmin(1.0f, entropy / MAX_NEURONS);
  }

  return 0.0f;
}

float estimateTaskDifficulty(TaskPrompt current_prompt, float error_rate) {
  // Base difficulty on error rate and expected outcome
  float difficulty_score = 0.0f;

  // Increase difficulty if error rate is high
  if (error_rate > 0.5f) {
    difficulty_score +=
        (error_rate - 0.5f) * 2.0f; // Scale factor for error rate
  }

  // Factor in expected outcome from the current prompt
  difficulty_score += 1.0f - current_prompt.expected_outcome;

  // Normalize score to be between 0 and 1
  if (difficulty_score > 1.0f) {
    difficulty_score = 1.0f;
  } else if (difficulty_score < 0.0f) {
    difficulty_score = 0.0f;
  }

  return difficulty_score;
}

float addRandomNoise(float value, float noise_level) {
  // Generate random noise within the range [-noise_level, noise_level]
  float noise =
      ((float)arc4random() / UINT32_MAX) * 2.0f * noise_level - noise_level;
  return value + noise;
}

float computeNovelty(Neuron *neurons, NetworkStateSnapshot stateHistory,
                     int step) {
  float novelty_score = 0.0f;

  // Compare current neurons to historical state to compute novelty
  for (int i = 0; i < step; i++) {
    float state_difference = fabs(neurons[i].state - stateHistory.states[i]);
    novelty_score += state_difference;
  }

  // Normalize novelty score based on the number of neurons
  if (step > 0) {
    novelty_score /= step;
  }

  // Ensure novelty score is between 0 and 1
  if (novelty_score > 1.0f) {
    novelty_score = 1.0f;
  } else if (novelty_score < 0.0f) {
    novelty_score = 0.0f;
  }

  return novelty_score;
}

int main() {
  // Try to load existing memory system
  MemorySystem *memorySystem = NULL;
  WorkingMemorySystem *working_memory =
      createWorkingMemorySystem(200); // adjust capacity as needed
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
  IntrinsicMotivation *motivation = initializeMotivationSystem();
  GoalSystem *goalSystem = initializeGoalSystem(10);

  GlobalContextManager *contextManager =
      initializeGlobalContextManager(MAX_NEURONS);
  NetworkPerformanceMetrics *performanceMetrics =
      initializePerformanceMetrics(network_regions);

  addGoal(goalSystem, "Minimize prediction error", 1.0f);
  addGoal(goalSystem, "Develop stable representations", 0.8f);
  addGoal(goalSystem, "Maximize information gain", 0.7f);

  // Main training loop
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

    initPredictiveCodingParams(max_neurons);

    float *predictive_inputs = malloc(max_neurons * sizeof(float));
    generatePredictiveInputs(predictive_inputs,
                             (step > 0) ? &stateHistory[step - 1] : NULL,
                             max_neurons);

    // Prepare input tensor
    float *input_tensor = (float *)malloc(max_neurons * sizeof(float));
    memcpy(input_tensor, predictive_inputs, max_neurons * sizeof(float));
    generateInputTensor(input_tensor, step, text_input, relevantMemory,
                        system_params);

    if (step % 10 == 0) { // Periodic memory maintenance
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

    // Forward pass using CPU implementation
    updateNeuronsOnCPU(neurons, weights, connections, max_neurons,
                       max_connections, input_tensor, input_size, weights);

    computePredictionErrors(neurons, input_tensor, max_neurons);

    // Generate target outputs
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
    float *outputErrors = (float *)malloc(max_neurons * sizeof(float));

    float beta1 = 0.9f; // Exponential decay rate for the first moment estimate
    float beta2 =
        0.999f; // Exponential decay rate for the second moment estimate
    float epsilon = 1e-8f;        // Small constant to avoid division by zero
    float learning_rate = 0.001f; // Learning rate for gradient descent

    // Moment buffers for Adam optimizer (initialize with zeros)
    float *m =
        (float *)calloc(max_connections, sizeof(float)); // First moment (m)
    float *v =
        (float *)calloc(max_connections, sizeof(float)); // Second moment (v)

    int t = 1;

    backpropagationOnCPU(
        neurons,         // Pointer to the array of neurons
        weights,         // Pointer to the array of weights
        connections,     // Pointer to the array of connections
        max_neurons,     // Maximum number of neurons
        max_connections, // Maximum number of connections
        target_outputs,  // Pointer to the array of target outputs
        outputErrors,    // Pointer to the array to store output errors
        learning_rate    // Learning rate for weight updates
    );

    // Update weights using CPU implementation
    updateWeightsOnCPU(weights, neurons, connections, learning_rate,
                       max_neurons, max_connections);
    processNeuronsOnCPU(neurons, weights, connections, max_neurons,
                        max_connections, input_tensor, input_size);

    // Reverse process using CPU implementation
    reverseProcessOnCPU(neurons, reverse_weights, reverse_connections,
                        max_neurons, max_connections);

    // Memory replay every 5 steps
    if (step % 5 == 0 && memorySystem->size > 10) {
      memoryReplayOnCPU(neurons, weights, connections, memorySystem->entries,
                        memorySystem->capacity, max_neurons, max_connections);
      printf("\nMemory Replay at step %d:", step);
      printReplayStatistics(memorySystem);
    }

    // Compute loss
    float loss = computeMSELoss(neurons, target_outputs, max_neurons);
    printf("Loss: %f\n", loss);

    verifyNetworkState(neurons, &current_prompt);

    // Update global context based on current network state
    updateGlobalContext(contextManager, neurons, max_neurons, input_tensor);

    // Integrate context into network processing
    integrateGlobalContext(contextManager, neurons, max_neurons, weights,
                           max_connections);

    DynamicContextFeedback feedback = {.adaptation_rate = 0.01f,
                                       .history_size = 100,
                                       .current_index = 0,
                                       .context_threshold = 0.3f,
                                       .feedback_decay = 0.95f};

    feedback.context_weights = (float *)calloc(max_neurons, sizeof(float));
    feedback.feedback_history =
        (float *)calloc(feedback.history_size, sizeof(float));

    ContextAdaptation adaptation = {.history_length = 50,
                                    .learning_momentum = 0.8f,
                                    .minimum_context_weight = 0.1f};

    adaptation.recent_outcomes =
        (float *)calloc(adaptation.history_length, sizeof(float));
    adaptation.input_history =
        (float *)calloc(adaptation.history_length * max_neurons, sizeof(float));
    adaptation.correlation_matrix =
        (float *)calloc(max_neurons * max_neurons, sizeof(float));

    // Update global context based on current network state
    updateGlobalContext(contextManager, neurons, max_neurons, input_tensor);

    // Calculate outcome metrics for feedback
    float current_outcome =
        computeOutcomeMetric(neurons, target_outputs, max_neurons);

    // Store outcome and input in history
    int history_idx = step % adaptation.history_length;
    adaptation.recent_outcomes[history_idx] = current_outcome;
    memcpy(&adaptation.input_history[history_idx * max_neurons], input_tensor,
           max_neurons * sizeof(float));

    // Update correlation matrix
    updateCorrelationMatrix(
        adaptation.correlation_matrix, adaptation.input_history,
        adaptation.recent_outcomes, adaptation.history_length, max_neurons);

    // Compute feedback signal
    float feedback_signal = computeFeedbackSignal(
        current_outcome, feedback.feedback_history, feedback.history_size);

    // Update context weights based on feedback
    for (int i = 0; i < max_neurons; i++) {
      float weight_update = feedback_signal * feedback.adaptation_rate;

      // Apply correlation-based adjustments
      for (int j = 0; j < max_neurons; j++) {
        weight_update += adaptation.correlation_matrix[i * max_neurons + j] *
                         adaptation.learning_momentum;
      }

      // Update weight with momentum and bounds
      feedback.context_weights[i] =
          fmax(adaptation.minimum_context_weight,
               feedback.context_weights[i] + weight_update);
    }

    // Store feedback for history
    feedback.feedback_history[feedback.current_index] = feedback_signal;
    feedback.current_index =
        (feedback.current_index + 1) % feedback.history_size;

    // Apply updated context weights to network processing
    applyDynamicContext(neurons, feedback.context_weights, contextManager,
                        max_neurons);

    // Decay historical feedback influence
    for (int i = 0; i < feedback.history_size; i++) {
      feedback.feedback_history[i] *= feedback.feedback_decay;
    }

    // Integrate context into network processing with dynamic weights
    integrateGlobalContext(contextManager, neurons, max_neurons, weights,
                           max_connections);

    // Print context adaptation metrics periodically
    if (step % 20 == 0) {
      printf("\nContext Adaptation Metrics (Step %d):\n", step);
      printf("Average Feedback Signal: %.4f\n",
             computeAverageFeedback(feedback.feedback_history,
                                    feedback.history_size));
      printf("Context Weight Range: %.4f - %.4f\n",
             computeMinWeight(feedback.context_weights, max_neurons),
             computeMaxWeight(feedback.context_weights, max_neurons));
      printf("Correlation Strength: %.4f\n",
             computeAverageCorrelation(adaptation.correlation_matrix,
                                       max_neurons));
    }

    float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE] = {
        {0.1, 0.2, 0.3, 0.4, 0.5},      // row 1
        {0.6, 0.7, 0.8, 0.9, 0.10},     // row 2
        {0.11, 0.12, 0.13, 0.14, 0.15}, // row 3
    };

    // Update memory system with new outputs
    addMemory(memorySystem, working_memory, neurons, input_tensor,
              lastTimestamp + step + 1, feature_projection_matrix);

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

    float novelty = computeNovelty(neurons, *stateHistory, step);
    float task_difficulty = estimateTaskDifficulty(
        current_prompt, performance_history[step].error_rate);

    updateMotivationSystem(motivation, performance_delta, novelty,
                           task_difficulty);

    // Update goals and generate rewards
    for (int i = 0; i < goalSystem->num_goals; i++) {
      Goal *goal = &goalSystem->goals[i];
      float new_progress = evaluateGoalProgress(goal, neurons, target_outputs);
      float progress_delta = new_progress - goal->progress;
      goal->progress = new_progress;

      // Generate intrinsic reward based on progress
      float intrinsic_reward = progress_delta * goal->reward_value;

      // Apply reward to learning
      learning_rate *= (1.0f + 0.1f * intrinsic_reward);
    }

    // Modify exploration vs exploitation based on motivation
    float explore_prob = motivation->exploration_rate;
    if (rand() / (float)RAND_MAX < explore_prob) {
      // Take exploratory action
      addRandomNoise(*input_tensor, motivation->curiosity_drive * 0.1f);
    }

    // Add to periodic logging
    if (step % 20 == 0) {
      printf("\nMotivation System Status:\n");
      printf("Competence: %.2f\n", motivation->competence_score);
      printf("Curiosity: %.2f\n", motivation->curiosity_drive);
      printf("Mastery: %.2f\n", motivation->mastery_level);
      printf("Exploration Rate: %.2f\n", motivation->exploration_rate);

      printf("\nActive Goals:\n");
      for (int i = 0; i < goalSystem->num_goals; i++) {
        printf("%s: %.1f%% complete (Priority: %.2f)\n",
               goalSystem->goals[i].description,
               goalSystem->goals[i].progress * 100.0f,
               goalSystem->goals[i].priority);
      }
    }

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

    learning_rate = opt_state.optimal_learning_rate;
    updateNeuronsWithPredictiveCoding(neurons, input_tensor, max_neurons,
                                      learning_rate);

    updateWorkingMemory(working_memory, neurons, input_tensor, target_outputs,
                        step);

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
    if (step % 10 == 0) {
      consolidateToLongTermMemory(working_memory, memorySystem, step);
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
  free(working_memory);
  free(performance_history);
  free(contextManager);
  free(performanceMetrics);
  free(metaController);
  return 0;
}
