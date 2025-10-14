#include "../../include/definitions.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <curl/curl.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <immintrin.h>
#include <json-c/json.h>
#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define arc4random() rand()

typedef struct {
  float state;
  float output;
  unsigned int num_connections;
  unsigned int layer_id;
} Neuron;

typedef enum {
  SPEC_NONE = 0,
  SPEC_PATTERN_DETECTOR,
  SPEC_FEATURE_EXTRACTOR,
  SPEC_TEMPORAL_PROCESSOR,
  SPEC_CONTEXT_INTEGRATOR,
  SPEC_DECISION_MAKER,
  SPEC_MEMORY_ENCODER,
  SPEC_EMOTIONAL_PROCESSOR,
  SPEC_PREDICTION_GENERATOR
} NeuronSpecializationType;

typedef struct {
  unsigned int neuron_id;
  NeuronSpecializationType type;
  float specialization_score;
  float activation_history[50]; // Recent activation history
  unsigned int history_index;   // Current index in circular buffer
  float avg_activation;         // Average activation level
  float importance_factor;      // How important this specialized neuron is
} SpecializedNeuron;

typedef struct {
  SpecializedNeuron neurons[MAX_SPECIALIZED_NEURONS];
  unsigned int count;
  float type_distribution[MAX_SPECIALIZATIONS]; // Distribution of
                                                // specialization types
  float specialization_threshold; // Minimum score to be considered specialized
} NeuronSpecializationSystem;

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

struct VocabularyEntry {
  char word[64];
  char category[32];
  float semantic_weight;
  char *connects_to;
  char *description;
  float letter_weight;

  VocabularyEntry()
      : semantic_weight(1.0f), connects_to(nullptr), description(nullptr),
        letter_weight(1.0f) {
    word[0] = '\0';
    category[0] = '\0';
  }

  ~VocabularyEntry() {
    if (connects_to) {
      free(connects_to);
      connects_to = nullptr;
    }
    if (description) {
      free(description);
      description = nullptr;
    }
  }

  VocabularyEntry(const VocabularyEntry &other)
      : semantic_weight(other.semantic_weight),
        letter_weight(other.letter_weight) {
    strncpy(word, other.word, sizeof(word) - 1);
    word[sizeof(word) - 1] = '\0';
    strncpy(category, other.category, sizeof(category) - 1);
    category[sizeof(category) - 1] = '\0';

    connects_to = other.connects_to ? strdup(other.connects_to) : nullptr;
    description = other.description ? strdup(other.description) : nullptr;
  }

  VocabularyEntry &operator=(const VocabularyEntry &other) {
    if (this != &other) {
      if (connects_to)
        free(connects_to);
      if (description)
        free(description);

      strncpy(word, other.word, sizeof(word) - 1);
      word[sizeof(word) - 1] = '\0';
      strncpy(category, other.category, sizeof(category) - 1);
      category[sizeof(category) - 1] = '\0';

      semantic_weight = other.semantic_weight;
      letter_weight = other.letter_weight;

      connects_to = other.connects_to ? strdup(other.connects_to) : nullptr;
      description = other.description ? strdup(other.description) : nullptr;
    }
    return *this;
  }
};

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
  uint32_t vector_size;
  struct ContextNode **children;
  uint32_t num_children;
  uint32_t max_children;
  struct ContextNode *parent;
  float temporal_relevance;
  uint64_t last_updated;
} ContextNode;

typedef struct GlobalContextManager {
  ContextNode *root;
  uint32_t total_nodes;
  float *global_context_vector;
  uint32_t vector_size;
  float decay_rate;
  float update_threshold;
  uint32_t max_depth;
  uint32_t max_children_per_node;
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
  char description[256];   // Goal description
  float priority;          // Priority level (0.1 to 1.0)
  float progress;          // Current progress towards goal (0.0 to 1.0)
  float previous_progress; // Previous progress value for delta calculation
  float reward_value;      // Reward value when goal is achieved
  bool achieved;           // Whether the goal has been achieved
  time_t timestamp;        // When the goal was created/updated
  int stability_counter;   // Counter for tracking stability instead of just
                           // improvements
} Goal;

typedef struct {
  Goal *goals;              // Array of goals
  int num_goals;            // Number of active goals
  int capacity;             // Maximum number of goals
  float planning_horizon;   // Time horizon for planning
  float discount_factor;    // Discount factor for future rewards
  float min_learning_rate;  // Minimum bound for learning rate
  float max_learning_rate;  // Maximum bound for learning rate
  float base_learning_rate; // Base learning rate to return to
} GoalSystem;

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

// Self-reflection system structures
typedef struct {
  float confidence_score;
  float coherence_score;
  float novelty_score;
  float consistency_score;
  char reasoning[1024];
  bool potentially_confabulated;
} ReflectionMetrics;

typedef struct {
  float historical_confidence[100];
  float historical_coherence[100];
  float historical_consistency[100];
  int history_index;
  float confidence_threshold;
  float coherence_threshold;
  float consistency_threshold;
} ReflectionHistory;

typedef struct {
  float current_adaptation_rate;
  float input_noise_scale;
  float weight_noise_scale;
  float plasticity;
  float noise_tolerance;
  float learning_rate;
} ReflectionParameters;

typedef struct {
  float *core_values;         // Stable personality traits/values
  float *belief_system;       // Current belief states
  float *identity_markers;    // Unique identifying characteristics
  float *experience_history;  // Compressed history of experiences
  float *behavioral_patterns; // Consistent behavior patterns

  int num_core_values;
  int num_beliefs;
  int num_markers;
  int history_size;
  int pattern_size;

  float consistency_score; // Measure of identity stability
  float adaptation_rate;   // Rate of identity evolution
  float confidence_level;  // Self-confidence in identity

  // Temporal consistency tracking
  float *temporal_coherence; // Track consistency over time
  int coherence_window;      // Time window for coherence analysis

  // Identity verification system
  struct {
    float threshold;        // Minimum consistency threshold
    float *reference_state; // Reference identity state
    int state_size;         // Size of reference state
  } verification;

} SelfIdentitySystem;

// Knowledge category structure
typedef struct {
  char name[64];
  float *feature_vector;
  float importance;
  float confidence;
  int usage_count;
  time_t last_accessed;
} KnowledgeCategory;

typedef struct {
  char description[256];
  float *feature_vector;
  float difficulty;
  float success_rate;
  KnowledgeCategory *category;
  time_t timestamp;
} ProblemInstance;

typedef struct {
  KnowledgeCategory *categories;
  int num_categories;
  int capacity;
  ProblemInstance *problem_history;
  int num_problems;
  int problem_capacity;
  float *category_similarity_matrix;
} KnowledgeFilter;

typedef struct DecisionPath {
  float *states;    // Predicted neuron states
  float *weights;   // Weight adjustments
  int *connections; // Connection changes
  float score;      // Path evaluation score
  int num_steps;    // Number of prediction steps
} DecisionPath;

typedef struct MetacognitionMetrics {
  float confidence_level;                    // Overall confidence in decisions
  float adaptation_rate;                     // Rate of learning adjustment
  float cognitive_load;                      // Current processing complexity
  float error_awareness;                     // Awareness of prediction errors
  float context_relevance;                   // Relevance of current context
  float performance_history[HISTORY_LENGTH]; // Historical performance tracking
} MetacognitionMetrics;

typedef struct MetaLearningState {
  float learning_efficiency; // Current learning effectiveness
  float exploration_rate;    // Balance between exploration/exploitation
  float stability_index;     // System stability measure
  float *priority_weights;   // Attention allocation weights
  int current_phase;         // Current learning phase
} MetaLearningState;

typedef struct {
  float avg_success_rate;
  float avg_difficulty;
  int total_instances;
  time_t last_encounter;
} CategoryStatistics;

typedef struct {
  bool critical_violation;
  uint64_t suspect_address;
  const char *violation_type;
} SecurityValidationStatus;

typedef struct {
  float *core_values;
  float *belief_system;
  float *identity_markers;
  float *experience_history;
  float *behavioral_patterns;
  float *temporal_coherence;
  float *reference_state;
  float consistency_score;
  float adaptation_rate;
  float confidence_level;
  uint32_t num_core_values;
  uint32_t num_beliefs;
  uint32_t num_markers;
  uint32_t history_size;
  uint32_t pattern_size;
  uint32_t coherence_window;
  uint32_t state_size;
} SelfIdentityBackup;

// Structure to store analysis results
typedef struct {
  uint32_t core_value_conflicts; // Number of unstable core values
  uint32_t belief_conflicts;     // Number of inconsistent beliefs
  uint32_t marker_conflicts;     // Number of deviated identity markers
  float temporal_instability;    // Measure of temporal coherence deviation
  float pattern_deviation;       // Deviation in behavioral patterns
  float overall_consistency;     // Overall system consistency score
  float confidence_impact;       // Impact on confidence level
} IdentityAnalysis;

typedef struct {
  int symbol_id;
  char description[256];
} InternalSymbol;

typedef struct {
  int question_id;
  int symbol_ids[MAX_SYMBOLS];
  int num_symbols;
} InternalQuestion;

typedef struct {
  char *data;
  size_t size;
} HttpResponse;

typedef struct {
  char **titles;
  char **snippets;
  char **urls;
  int count;
} SearchResults;

typedef struct {
  float importance;      // How important this principle is (0.0-1.0)
  float adherence;       // Current adherence level (0.0-1.0)
  char description[256]; // Description of the principle
  int violations;        // Count of violations
  int activations;       // Count of successful applications
} EthicalPrinciple;

typedef struct {
  float benefit_score;    // Positive impact measurement
  float harm_score;       // Negative impact measurement
  float uncertainty;      // Level of uncertainty in assessment
  int affected_parties;   // Number of parties potentially affected
  float reversibility;    // How reversible the decision is (0-1)
  float long_term_impact; // Long-term consequence rating
} DecisionImpact;

typedef struct {
  EthicalPrinciple *principles; // Array of ethical principles
  int num_principles;           // Number of principles
  float overall_alignment;      // Overall ethical alignment (0.0-1.0)
  DecisionImpact last_decision; // Impact of the last decision
  float confidence_threshold;   // Minimum confidence for ethical decisions
  int dilemma_count;            // Number of ethical dilemmas encountered
  int resolution_count;         // Number of dilemmas successfully resolved
} MoralCompass;

typedef struct {
  bool is_readable;
  bool is_writable;
  bool is_executable;
  size_t region_size;
} MemoryProtection;

typedef struct {
  float intensity;          // Strength of the emotion (0.0 to 1.0)
  float decay_rate;         // How quickly the emotion fades
  float influence_factor;   // How much this emotion affects decision making
  float threshold;          // Activation threshold for this emotion
  float previous_intensity; // For tracking changes
  float momentum;           // Carries emotional momentum across steps
  unsigned int last_update; // Timestamp of last update
} EmotionState;

typedef struct {
  EmotionState emotions[MAX_EMOTION_TYPES];
  float cognitive_impact;     // How much emotions affect logical processing
  float emotional_regulation; // System's ability to regulate emotions (0.0-1.0)
  float emotional_memory[MAX_EMOTION_TYPES]
                        [10]; // Recent emotional memory traces
  int memory_index;           // Current index in circular memory buffer
} EmotionalSystem;

typedef struct {
  float probability;
  float confidence;
  float impact_score;
  float plausibility;
  float vector[MEMORY_VECTOR_SIZE];
  char description[256];
} ImaginedOutcome;

typedef struct {
  int num_outcomes;
  ImaginedOutcome outcomes[10];
  float divergence_factor;
  float creativity_level;
} ImaginationScenario;

typedef struct {
  ImaginationScenario scenarios[MAX_SCENARIOS];
  int num_scenarios;
  int current_scenario;
  float creativity_factor;
  float coherence_threshold;
  float novelty_weight;
  float memory_influence;
  float identity_influence;
  bool active;
  int steps_simulated;
  float divergence_history[100];
  char current_scenario_name[MAX_SCENARIO_NAME_LENGTH];
  int total_scenarios_generated;
} ImaginationSystem;

typedef struct {
  unsigned int timestamp;
  int person_id;              // ID of the person involved
  float emotional_state[5];   // Emotional state during interaction
  float cooperation_level;    // How cooperative the interaction was
  float outcome_satisfaction; // How satisfied both parties were
  char interaction_type[32];  // Type of interaction (negotiation, casual, etc.)
  char *context;              // Context of the interaction
} SocialInteraction;

typedef struct {
  int person_id;
  char person_name[64];
  float observed_traits[10];   // Personality traits inferred
  float prediction_confidence; // Confidence in behavioral predictions
  float relationship_quality;  // Quality of relationship with this person
  float trust_level;           // Trust built with this person
  int interaction_count;       // Number of interactions with this person
} PersonModel;

typedef struct {
  // Core social capabilities
  float empathy_level;     // Ability to understand others' emotions
  float negotiation_skill; // Ability to find mutually beneficial solutions
  float behavior_prediction_accuracy; // Accuracy in predicting others' actions
  float social_awareness;             // Awareness of social dynamics and norms

  // Social interaction history
  int interaction_count;
  SocialInteraction *interactions; // Array of past interactions
  int max_interactions;            // Maximum number of interactions to store

  // Social models of others
  int model_count;
  PersonModel
      *person_models; // Models of individuals the system has interacted with
  int max_models;     // Maximum number of models to maintain

  // Social learning parameters
  float learning_rate;     // Rate at which social skills improve
  float forgetting_factor; // Rate at which old interactions lose relevance
} SocialSystem;

typedef struct {
  int *active_dims;                        // Indices of active dimensions
  float *values;                           // Values for active dimensions only
  int num_active;                          // Number of active dimensions
  float norm;                              // Cached L2 norm for efficiency
  int semantic_layer[NUM_SEMANTIC_LAYERS]; // Hierarchical features
} SparseEmbedding;

typedef struct {
  char context_hash[32]; // Hash of recent context
  SparseEmbedding embedding;
  float recency; // How recently this was accessed
} ContextEmbedding;

typedef struct {
  float query_weights[NUM_HEADS][EMBEDDING_SIZE][HEAD_DIM];
  float key_weights[NUM_HEADS][EMBEDDING_SIZE][HEAD_DIM];
  float value_weights[NUM_HEADS][EMBEDDING_SIZE][HEAD_DIM];
  float output_weights[EMBEDDING_SIZE][EMBEDDING_SIZE];
  float positional_encoding[INPUT_SIZE][EMBEDDING_SIZE];
  int initialized;
} AttentionParams;

static AttentionParams g_attention_params = {0};
InternalSymbol symbol_table[MAX_SYMBOLS];
InternalQuestion question_table[MAX_QUESTIONS];
int num_symbols = 0;
int num_questions = 0;

__device__ float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

__device__ float fract(float x) { return x - floorf(x); }

__device__ float fast_tanh(float x) {
  float x2 = x * x;
  float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
  return fminf(fmaxf(a / b, MIN_ACTIVATION), MAX_ACTIVATION);
}

// ReLU activation function
__device__ float relu(float x) { return fmaxf(0.0f, x); }

// Sigmoid activation function
__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Leaky ReLU activation function
__device__ float leaky_relu(float x, float alpha = 0.01f) {
  return x > 0.0f ? x : alpha * x;
}

// Swish activation function (x * sigmoid(x))
__device__ float swish(float x) { return x * sigmoid(x); }

// Activation function with configurable response curve and type
__device__ float activation_function(float x, float scale, float bias,
                                     unsigned int activation_type) {
  // Apply scale and bias
  float scaled = x * scale + bias;

  // Select activation function based on type
  float base_activation;
  switch (activation_type) {
  case ACTIVATION_RELU:
    base_activation = relu(scaled);
    break;
  case ACTIVATION_SIGMOID:
    base_activation = sigmoid(scaled);
    break;
  case ACTIVATION_LEAKY_RELU:
    base_activation = leaky_relu(scaled);
    break;
  case ACTIVATION_SWISH:
    base_activation = swish(scaled);
    break;
  case ACTIVATION_TANH:
  default:
    base_activation = fast_tanh(scaled);
    break;
  }

  // Add nonlinearity for more dynamic response
  if (activation_type == ACTIVATION_TANH ||
      activation_type == ACTIVATION_SIGMOID) {
    float sign_val = copysignf(1.0f, base_activation);
    float abs_val = fabsf(base_activation);
    return sign_val * powf(abs_val, 1.1f);
  } else {
    return base_activation;
  }
}

__global__ void
update_neurons(Neuron *neurons, const float *weights,
               const unsigned int *connections, const unsigned int max_neurons,
               const unsigned int max_connections, const float *input_tensor,
               const unsigned int input_size, const float *recurrent_weights,
               const unsigned int activation_type) {
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
  float new_output = activation_function(new_state, dynamic_scale,
                                         ACTIVATION_BIAS, activation_type);

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
                const unsigned int input_size, const float *recurrent_weights,
                const unsigned int activation_type) {
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
  float new_output = activation_function(new_state, dynamic_scale,
                                         ACTIVATION_BIAS, activation_type);

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
backward_kernel(const Neuron *neurons, float *weights, const int *connections,
                const unsigned int max_neurons,
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
  for (size_t i = 0; i < system->size; i++) {
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

int *findLeastImportantMemory(MemoryEntry *entries, unsigned int size,
                              unsigned int count, unsigned int *result_count) {
  if (count == 0 || size == 0) {
    *result_count = 0;
    return NULL;
  }

  // Don't return more than available
  count = (count > size) ? size : count;
  *result_count = count;

  // Create array of indices with their importance values
  typedef struct {
    int index;
    float importance;
  } IndexPair;

  IndexPair *pairs = new IndexPair[size];
  for (unsigned int i = 0; i < size; i++) {
    pairs[i].index = i;
    pairs[i].importance = entries[i].importance;
  }

  // Sort by importance (ascending - least important first)
  for (unsigned int i = 0; i < size - 1; i++) {
    for (unsigned int j = i + 1; j < size; j++) {
      if (pairs[i].importance > pairs[j].importance) {
        IndexPair temp = pairs[i];
        pairs[i] = pairs[j];
        pairs[j] = temp;
      }
    }
  }

  // Extract the least important indices
  int *result = new int[count];
  for (unsigned int i = 0; i < count; i++) {
    result[i] = pairs[i].index;
  }

  free(pairs);
  return result;
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
  WorkingMemorySystem *system =
      (WorkingMemorySystem *)malloc(sizeof(WorkingMemorySystem));

  // Initialize focused attention component
  system->focus.capacity = capacity * 0.2; // 20% for focused attention
  system->focus.entries = new WorkingMemoryEntry[sizeof(
      WorkingMemoryEntry)]; // Use 'new' instead of malloc
  system->focus.size = 0;
  system->focus.attention_threshold = 0.8f;

  // Initialize active working memory
  system->active.capacity = capacity * 0.8; // 80% for active memory
  system->active.entries = new WorkingMemoryEntry[sizeof(
      WorkingMemoryEntry)]; // Use 'new' instead of malloc
  system->active.size = 0;
  system->active.activation_decay = 0.95f;

  // Initialize dynamic clustering
  system->clusters.num_clusters = 5; // Start with 5 clusters
  system->clusters.clusters =
      static_cast<SemanticCluster *>(malloc(sizeof(SemanticCluster)));

  // Initialize global context
  system->global_context =
      new float[256](); // Use 'new' and initialize with zeroes

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

  float *similarities = static_cast<float *>(
      malloc(system->clusters.num_clusters * sizeof(float)));

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

  if (entry.importance > working_memory->focus.attention_threshold) {
    // Add to focused attention
    if (working_memory->focus.size < working_memory->focus.capacity) {
      enhanced.features = new float[FEATURE_VECTOR_SIZE];
      extractSemanticFeatures(entry.vector, enhanced.features,
                              feature_projection_matrix);

      enhanced.context_vector = new float[CONTEXT_VECTOR_SIZE];
      memcpy(enhanced.context_vector, working_memory->global_context,
             CONTEXT_VECTOR_SIZE * sizeof(float));

      working_memory->focus.entries[working_memory->focus.size++] = enhanced;
      updateSemanticClusters(working_memory, &enhanced);
    }
  } else {
    // Add to active memory
    if (working_memory->active.size < working_memory->active.capacity) {
      enhanced.features = new float[FEATURE_VECTOR_SIZE];
      extractSemanticFeatures(entry.vector, enhanced.features,
                              feature_projection_matrix);

      enhanced.context_vector = new float[CONTEXT_VECTOR_SIZE];
      memcpy(enhanced.context_vector, working_memory->global_context,
             CONTEXT_VECTOR_SIZE * sizeof(float));

      working_memory->active.entries[working_memory->active.size++] = enhanced;
      updateSemanticClusters(working_memory, &enhanced);
    }
  }

  // Update global context
  updateContext(working_memory);

  // Then handle original hierarchical storage - NOW WITH BATCH REPLACEMENT
  if (entry.importance >= system->hierarchy.long_term.importance_threshold) {
    if (system->hierarchy.long_term.size <
        system->hierarchy.long_term.capacity) {
      system->hierarchy.long_term.entries[system->hierarchy.long_term.size++] =
          entry;
    } else {
      // Find multiple least important memories and replace the worst one
      unsigned int replace_count;
      int *least_important = findLeastImportantMemory(
          system->hierarchy.long_term.entries, system->hierarchy.long_term.size,
          5, // Get 5 least important
          &replace_count);

      if (least_important && replace_count > 0) {
        // Replace the absolute worst (first in sorted array)
        system->hierarchy.long_term.entries[least_important[0]] = entry;
        free(least_important);
      }
    }
  } else if (entry.importance >=
             system->hierarchy.medium_term.importance_threshold) {
    if (system->hierarchy.medium_term.size <
        system->hierarchy.medium_term.capacity) {
      system->hierarchy.medium_term
          .entries[system->hierarchy.medium_term.size++] = entry;
    } else {
      // Try to replace least important in medium term first
      unsigned int replace_count;
      int *least_important =
          findLeastImportantMemory(system->hierarchy.medium_term.entries,
                                   system->hierarchy.medium_term.size,
                                   3, // Get 3 least important
                                   &replace_count);

      if (least_important && replace_count > 0) {
        // Check if new entry is more important than the worst existing one
        if (entry.importance >
            system->hierarchy.medium_term.entries[least_important[0]]
                .importance) {
          system->hierarchy.medium_term.entries[least_important[0]] = entry;
        }
        free(least_important);
      } else {
        consolidateToHigherLevel(system);
      }
    }
  } else {
    if (system->hierarchy.short_term.size <
        system->hierarchy.short_term.capacity) {
      system->hierarchy.short_term
          .entries[system->hierarchy.short_term.size++] = entry;
    } else {
      // Replace least important in short term or consolidate
      unsigned int replace_count;
      int *least_important =
          findLeastImportantMemory(system->hierarchy.short_term.entries,
                                   system->hierarchy.short_term.size,
                                   5, // Get 5 least important
                                   &replace_count);

      if (least_important && replace_count > 0) {
        // Always replace the worst one in short term
        system->hierarchy.short_term.entries[least_important[0]] = entry;
        free(least_important);
      } else {
        consolidateToMediumTerm(system);
      }
    }
  }

  // Update original structure for compatibility
  system->entries[system->head] = entry;
  system->head = (system->head + 1) % system->capacity;
  if (system->size < system->capacity) {
    system->size++;
  }
}

namespace py = pybind11;

void addMemoryWrapper(MemorySystem *system, WorkingMemorySystem *working_memory,
                      Neuron *neurons, py::array_t<float> input_tensor,
                      unsigned int timestamp,
                      py::array_t<float> feature_projection_matrix) {

  // Check dimensions
  auto buf_input = input_tensor.request();
  if (buf_input.ndim != 1)
    throw std::runtime_error("Input tensor must be 1D");

  auto buf_matrix = feature_projection_matrix.request();
  if (buf_matrix.ndim != 2 || buf_matrix.shape[0] != FEATURE_VECTOR_SIZE ||
      buf_matrix.shape[1] != MEMORY_VECTOR_SIZE)
    throw std::runtime_error("Feature projection matrix must have shape "
                             "[FEATURE_VECTOR_SIZE, MEMORY_VECTOR_SIZE]");

  // Convert input tensor to float pointer
  float *input_ptr = static_cast<float *>(buf_input.ptr);

  // Convert matrix to C-style array pointer
  float (*matrix_ptr)[MEMORY_VECTOR_SIZE] =
      reinterpret_cast<float (*)[MEMORY_VECTOR_SIZE]>(buf_matrix.ptr);

  // Call the original function
  addMemory(system, working_memory, neurons, input_ptr, timestamp, matrix_ptr);
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
    neurons[i].num_connections = MAX_CONNECTIONS;
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

extern VocabularyEntry vocabulary[VOCAB_SIZE];

int safe_vocab_size = 0;

int loadVocabularyFromFile(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    return -1;
  }

  int entryCount = 0;
  char buffer[500];

  while (fgets(buffer, sizeof(buffer), file) != NULL) {
    if (buffer[0] == '#' || buffer[0] == '\n' || buffer[0] == '\r') {
      continue;
    }
    entryCount++;
  }

  if (entryCount > VOCAB_SIZE) {
    fprintf(stderr,
            "Warning: File has %d entries but VOCAB_SIZE "
            "is %d. Truncating.\n",
            entryCount, VOCAB_SIZE);
    entryCount = VOCAB_SIZE;
  }

  if (entryCount == 0) {
    fprintf(stderr, "Error: No valid entries found in file\n");
    fclose(file);
    return -1;
  }

  rewind(file);

  int index = 0;
  int line_number = 0;

  while (fgets(buffer, sizeof(buffer), file) != NULL && index < entryCount) {
    line_number++;

    if (buffer[0] == '#' || buffer[0] == '\n' || buffer[0] == '\r') {
      continue;
    }

    buffer[strcspn(buffer, "\n")] = 0;
    buffer[strcspn(buffer, "\r")] = 0;

    std::vector<std::string> tokens;
    char *saveptr = nullptr;
    char *token = strtok_r(buffer, ",", &saveptr);

    while (token != nullptr) {
      while (*token == ' ' || *token == '\t')
        token++;

      char *end = token + strlen(token) - 1;
      while (end > token && (*end == ' ' || *end == '\t')) {
        *end = '\0';
        end--;
      }

      tokens.push_back(std::string(token));
      token = strtok_r(nullptr, ",", &saveptr);
    }

    if (tokens.empty() || tokens[0].empty()) {
      fprintf(stderr, "Warning: Skipping malformed line %d\n", line_number);
      continue;
    }

    strncpy(vocabulary[index].word, tokens[0].c_str(),
            sizeof(vocabulary[index].word) - 1);
    vocabulary[index].word[sizeof(vocabulary[index].word) - 1] = '\0';

    if (tokens.size() > 1 && !tokens[1].empty()) {
      strncpy(vocabulary[index].category, tokens[1].c_str(),
              sizeof(vocabulary[index].category) - 1);
      vocabulary[index].category[sizeof(vocabulary[index].category) - 1] = '\0';
    } else {
      strcpy(vocabulary[index].category, "unknown");
    }

    vocabulary[index].semantic_weight =
        (tokens.size() > 2 && !tokens[2].empty())
            ? static_cast<float>(atof(tokens[2].c_str()))
            : 1.0f;

    if (tokens.size() > 3 && !tokens[3].empty() && tokens[3] != "NULL" &&
        tokens[3] != "null") {
      vocabulary[index].connects_to = strdup(tokens[3].c_str());
      if (!vocabulary[index].connects_to) {
        fprintf(stderr,
                "Warning: Memory allocation failed for "
                "connects_to at line %d\n",
                line_number);
        vocabulary[index].connects_to = nullptr;
      }
    } else {
      vocabulary[index].connects_to = nullptr;
    }

    if (tokens.size() > 4 && !tokens[4].empty()) {
      vocabulary[index].description = strdup(tokens[4].c_str());
      if (!vocabulary[index].description) {
        fprintf(stderr,
                "Warning: Memory allocation failed for "
                "description at line %d\n",
                line_number);
        vocabulary[index].description = nullptr;
      }
    } else {
      vocabulary[index].description = nullptr;
    }

    vocabulary[index].letter_weight =
        (tokens.size() > 5 && !tokens[5].empty())
            ? static_cast<float>(atof(tokens[5].c_str()))
            : 1.0f;

    index++;
  }

  fclose(file);
  safe_vocab_size = index;

  printf("Successfully loaded %d vocabulary entries\n", safe_vocab_size);

  return index;
}
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

bool isWordMeaningful(const char *word) {
  // Check if the word is in the vocabulary
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(vocabulary[i].word, word) == 0) {
      return true;
    }
  }

  size_t len = strlen(word);

  // Check word length: meaningful words should be within a reasonable length
  if (len < 2 || len > 30) {
    return false;
  }

  // Check if all characters are letters or hyphens or apostrophes
  bool valid_chars = true;
  for (size_t i = 0; i < len; i++) {
    if (!isalpha(word[i]) && word[i] != '-' && word[i] != '\'') {
      valid_chars = false;
      break;
    }
  }
  if (!valid_chars) {
    return false;
  }

  // Check for at least one vowel (or other meaningful characters)
  bool has_vowel = false;
  for (size_t i = 0; i < len; i++) {
    if (strchr("aeiouAEIOU", word[i]) != NULL) {
      has_vowel = true;
      break;
    }
  }

  // If no vowels, check if it's a valid abbreviation or acronym
  if (!has_vowel) {
    bool is_acronym = true;
    for (size_t i = 0; i < len; i++) {
      if (!isupper(word[i])) {
        is_acronym = false;
        break;
      }
    }
    if (is_acronym) {
      return true;
    }
  }

  // Check for common prefixes/suffixes
  const char *prefixes[] = {"un",   "re",   "pre",   "in",  "dis",
                            "mis",  "over", "under", "sub", "post",
                            "anti", "de",   "en",    "co",  "non"};
  const char *suffixes[] = {"ing",  "tion", "ment", "ness", "able",
                            "ible", "er",   "est",  "ful",  "less",
                            "ly",   "ed",   "s",    "es",   "ies"};

  // Check prefixes
  for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++) {
    if (strncmp(word, prefixes[i], strlen(prefixes[i])) == 0) {
      return true;
    }
  }

  // Check suffixes
  for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
    if (strlen(word) >= strlen(suffixes[i]) &&
        strcmp(word + strlen(word) - strlen(suffixes[i]), suffixes[i]) == 0) {
      return true;
    }
  }

  // Check for proper nouns (capitalized first letter)
  if (isupper(word[0])) {
    return true;
  }

  // If none of the above checks pass, the word is not considered meaningful
  return false;
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
    if (!isWordMeaningful(customWord)) {
      // If the word doesn't make sense, regenerate it
      return mapToWord(fabs(value) *
                       0.9f); // Slightly adjust value and try again
    }
    return customWord;
  }

  return vocabulary[index].word;
}

void tokenizeString(const char *input, char **tokens, int *num_tokens) {
  *num_tokens = 0;
  int input_len = strlen(input);
  int i = 0;

  while (i < input_len && *num_tokens < INPUT_SIZE) {
    int best_match_len = 0;
    const char *best_match = NULL;

    // Greedy longest subword match
    for (int j = 0; j < vocab_size; j++) {
      const char *vocab_word = vocabulary[j].word;
      int vocab_len = strlen(vocab_word);

      if (vocab_len == 0 || i + vocab_len > input_len)
        continue;

      if (strncmp(&input[i], vocab_word, vocab_len) == 0) {
        if (vocab_len > best_match_len) {
          best_match_len = vocab_len;
          best_match = vocab_word;
        }
      }
    }

    if (best_match_len > 0) {
      tokens[*num_tokens] = strdup(best_match);
      (*num_tokens)++;
      i += best_match_len;
    } else {
      // Fallback to single character token
      char *fallback = (char *)malloc(2);
      fallback[0] = input[i];
      fallback[1] = '\0';
      tokens[*num_tokens] = fallback;
      (*num_tokens)++;
      i++;
    }
  }
}

float embeddings[vocab_size][EMBEDDING_SIZE];
SparseEmbedding word_embeddings[vocab_size];
ContextEmbedding context_cache[vocab_size * 4];
float similarity_hash[HASH_BUCKETS][vocab_size];
float semantic_weights[NUM_SEMANTIC_LAYERS][EMBEDDING_SIZE];

unsigned int hash_token(const char *token) {
  unsigned int hash = 5381;
  for (int i = 0; token[i] != '\0'; i++) {
    hash = ((hash << 5) + hash) + token[i];
  }
  return hash % HASH_BUCKETS;
}

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
  return (length > 0) ? (weight_sum / length) : 0.0f;
}

void initializeVocabularyWeights() {
  for (int i = 0; i < vocab_size; i++) {
    ((VocabularyEntry *)&vocabulary[i])->letter_weight =
        computeLetterWeight(vocabulary[i].word);
  }
}

void importPretrainedEmbeddings(const char *embedding_file) {
  if (safe_vocab_size <= 0) {
    fprintf(stderr, "Error: Vocabulary not loaded. Call "
                    "loadVocabularyFromFile first.\n");
    return;
  }

  FILE *file = fopen(embedding_file, "r");
  if (!file) {
    fprintf(stderr, "Error: Could not open embedding file: %s\n",
            embedding_file);
    printf("Falling back to random initialization...\n");

    for (int i = 0; i < safe_vocab_size; i++) {
      for (int j = 0; j < EMBEDDING_SIZE; j++) {
        float u1 = static_cast<float>(rand()) / RAND_MAX;
        float u2 = static_cast<float>(rand()) / RAND_MAX;
        if (u1 < 1e-8f)
          u1 = 1e-8f;
        float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
        embeddings[i][j] = z * 0.02f;
      }
    }
    return;
  }

  printf("Loading pre-trained embeddings from %s...\n", embedding_file);

  std::vector<bool> vocab_found(safe_vocab_size, false);

  std::unique_ptr<char[]> line(new char[MAX_LINE_LENGTH]);

  int file_dim = EMBEDDING_SIZE;
  if (fgets(line.get(), MAX_LINE_LENGTH, file) != nullptr) {
    int file_vocab_size;
    if (sscanf(line.get(), "%d %d", &file_vocab_size, &file_dim) == 2) {
      printf("Word2Vec format detected: %d words, %d dimensions\n",
             file_vocab_size, file_dim);

      if (file_dim != EMBEDDING_SIZE) {
        printf("Warning: File embedding size (%d) doesn't match "
               "system size (%d)\n",
               file_dim, EMBEDDING_SIZE);
        printf("Embeddings will be %s\n",
               file_dim > EMBEDDING_SIZE ? "truncated" : "padded with zeros");
      }
    } else {
      rewind(file);
    }
  }

  int loaded_count = 0;
  int line_num = 0;

  while (fgets(line.get(), MAX_LINE_LENGTH, file) != nullptr) {
    line_num++;

    char word[MAX_WORD_LENGTH];
    char *saveptr = nullptr;
    char *word_token = strtok_r(line.get(), " \t", &saveptr);

    if (!word_token || strlen(word_token) == 0)
      continue;

    strncpy(word, word_token, MAX_WORD_LENGTH - 1);
    word[MAX_WORD_LENGTH - 1] = '\0';

    int vocab_idx = -1;
    for (int i = 0; i < safe_vocab_size; i++) {
      if (strcmp(word, vocabulary[i].word) == 0) {
        vocab_idx = i;
        break;
      }
    }

    if (vocab_idx == -1)
      continue;

    vocab_found[vocab_idx] = true;
    loaded_count++;

    std::vector<float> temp_values;
    char *token = strtok_r(nullptr, " \t\n", &saveptr);

    while (token != nullptr &&
           temp_values.size() < static_cast<size_t>(file_dim)) {
      temp_values.push_back(static_cast<float>(atof(token)));
      token = strtok_r(nullptr, " \t\n", &saveptr);
    }

    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      if (j < static_cast<int>(temp_values.size())) {
        embeddings[vocab_idx][j] = temp_values[j];
      } else {
        embeddings[vocab_idx][j] = 0.0f;
      }
    }
  }

  fclose(file);

  printf("Successfully loaded %d/%d vocabulary words from "
         "pretrained embeddings\n",
         loaded_count, safe_vocab_size);

  for (int i = 0; i < safe_vocab_size; i++) {
    if (!vocab_found[i]) {
      bool found_category_match = false;
      std::vector<float> category_vector(EMBEDDING_SIZE, 0.0f);
      int category_matches = 0;

      for (int j = 0; j < safe_vocab_size; j++) {
        if (i != j && vocab_found[j] &&
            strcmp(vocabulary[i].category, vocabulary[j].category) == 0) {
          for (int k = 0; k < EMBEDDING_SIZE; k++) {
            category_vector[k] += embeddings[j][k];
          }
          category_matches++;
          found_category_match = true;
        }
      }

      if (found_category_match && category_matches > 0) {
        for (int k = 0; k < EMBEDDING_SIZE; k++) {
          category_vector[k] /= category_matches;
          float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
          embeddings[i][k] = category_vector[k] + noise;
        }
      } else {
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
          float u1 = static_cast<float>(rand()) / RAND_MAX;
          float u2 = static_cast<float>(rand()) / RAND_MAX;
          if (u1 < 1e-8f)
            u1 = 1e-8f;
          float z =
              std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
          embeddings[i][j] = z * 0.02f;
        }
      }
    }
  }

  for (int i = 0; i < safe_vocab_size; ++i) {
    float *emb_i = embeddings[i];
    const char *cat = vocabulary[i].category;

    if (cat && cat[0] != '\0') {
      int s, e;
      if (strcmp(cat, "fruit") == 0) {
        clip_range(0, 10, &s, &e);
        for (int j = s; j < e; ++j)
          emb_i[j] += 0.2f;
      } else if (strcmp(cat, "action") == 0) {
        clip_range(10, 20, &s, &e);
        for (int j = s; j < e; ++j)
          emb_i[j] += 0.2f;
      } else if (strcmp(cat, "emotion") == 0) {
        clip_range(20, 30, &s, &e);
        for (int j = s; j < e; ++j)
          emb_i[j] += 0.2f;
      }
    }

    {
      int s, e;
      clip_range(30, 40, &s, &e);
      float lw = vocabulary[i].letter_weight;
      for (int j = s; j < e; ++j)
        emb_i[j] += lw * 0.1f;
    }

    {
      int s, e;
      clip_range(40, 50, &s, &e);
      float sw = vocabulary[i].semantic_weight;
      for (int j = s; j < e; ++j)
        emb_i[j] += sw * 0.1f;
    }

    if (vocabulary[i].connects_to != nullptr) {
      const char *conn = vocabulary[i].connects_to;
      for (int j = 0; j < safe_vocab_size; ++j) {
        if (j == i)
          continue;
        if (strcmp(conn, vocabulary[j].word) == 0) {
          float *emb_j = embeddings[j];
          int s, e;
          clip_range(50, 60, &s, &e);
          for (int k = s; k < e; ++k) {
            float avg = (emb_i[k] + emb_j[k]) * 0.5f;
            emb_i[k] = emb_i[k] * 0.8f + avg * 0.2f;
            emb_j[k] = emb_j[k] * 0.8f + avg * 0.2f;
          }
          break;
        }
      }
    }
  }

  for (int i = 0; i < safe_vocab_size; i++) {
    float norm = 0.0f;
    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      norm += embeddings[i][j] * embeddings[i][j];
    }
    norm = std::sqrt(norm);

    if (norm > 1e-8f) {
      for (int j = 0; j < EMBEDDING_SIZE; j++) {
        embeddings[i][j] /= norm;
      }
    } else {
      for (int j = 0; j < EMBEDDING_SIZE; j++) {
        embeddings[i][j] = 0.0f;
      }
    }
  }

  printf("Embedding initialization completed with custom "
         "modifiers applied\n");
}

void initializeSparseEmbedding(SparseEmbedding *emb, int word_idx) {
  if (word_idx < 0 || word_idx >= safe_vocab_size) {
    fprintf(stderr, "Error: Invalid word_idx %d\n", word_idx);
    emb->num_active = 0;
    emb->active_dims = nullptr;
    emb->values = nullptr;
    emb->norm = 0.0f;
    return;
  }

  int target_active = static_cast<int>(EMBEDDING_SIZE * SPARSE_DENSITY);
  if (target_active <= 0)
    target_active = 1;
  if (target_active > EMBEDDING_SIZE)
    target_active = EMBEDDING_SIZE;

  emb->active_dims = new (std::nothrow) int[target_active];
  emb->values = new (std::nothrow) float[target_active];

  if (!emb->active_dims || !emb->values) {
    fprintf(stderr, "Error: Memory allocation failed for sparse "
                    "embedding\n");
    if (emb->active_dims)
      delete[] emb->active_dims;
    if (emb->values)
      delete[] emb->values;
    emb->num_active = 0;
    emb->active_dims = nullptr;
    emb->values = nullptr;
    emb->norm = 0.0f;
    return;
  }

  emb->num_active = target_active;
  emb->norm = 0.0f;

  std::vector<int> candidates(EMBEDDING_SIZE);
  std::vector<float> scores(EMBEDDING_SIZE);

  const char *word = vocabulary[word_idx].word;
  float word_length_factor = std::log(strlen(word) + 1) / std::log(10.0f);

  for (int i = 0; i < EMBEDDING_SIZE; i++) {
    candidates[i] = i;
    scores[i] = 0.0f;

    for (int j = 0; word[j]; j++) {
      if (word[j] >= 'a' && word[j] <= 'z') {
        scores[i] += letter_weights[word[j] - 'a'] * std::sin(i * 0.1f + j);
      }
    }

    int category_hash = 0;
    for (int j = 0; vocabulary[word_idx].category[j]; j++) {
      category_hash += vocabulary[word_idx].category[j] * (j + 1);
    }
    scores[i] +=
        std::sin(category_hash * 0.001f + i * 0.05f) * word_length_factor;
    scores[i] += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
  }

  for (int i = 0; i < EMBEDDING_SIZE - 1; i++) {
    for (int j = i + 1; j < EMBEDDING_SIZE; j++) {
      if (scores[j] > scores[i]) {
        std::swap(scores[i], scores[j]);
        std::swap(candidates[i], candidates[j]);
      }
    }
  }

  for (int i = 0; i < target_active; i++) {
    emb->active_dims[i] = candidates[i];

    float value = 0.0f;
    for (int layer = 0; layer < NUM_SEMANTIC_LAYERS; layer++) {
      int layer_contrib = (word_idx * 17 + candidates[i] * 23 + layer) % 1000;
      value += semantic_weights[layer][candidates[i]] *
               std::sin(layer_contrib * 0.01f);
      emb->semantic_layer[layer] = layer_contrib % EMBEDDING_SIZE;
    }

    emb->values[i] = std::tanh(value * vocabulary[word_idx].semantic_weight);
    emb->norm += emb->values[i] * emb->values[i];
  }

  emb->norm = std::sqrt(emb->norm);

  if (emb->norm > 1e-8f) {
    for (int i = 0; i < emb->num_active; i++) {
      emb->values[i] /= emb->norm;
    }
    emb->norm = 1.0f;
  }
}

float sparseCosineSimilarity(const SparseEmbedding *a,
                             const SparseEmbedding *b) {
  if (a->norm < 1e-8f || b->norm < 1e-8f)
    return 0.0f;

  float dot_product = 0.0f;
  int i = 0, j = 0;

  // Merge-like algorithm for sparse vectors
  while (i < a->num_active && j < b->num_active) {
    if (a->active_dims[i] == b->active_dims[j]) {
      dot_product += a->values[i] * b->values[j];
      i++;
      j++;
    } else if (a->active_dims[i] < b->active_dims[j]) {
      i++;
    } else {
      j++;
    }
  }

  // Both are normalized, so we can just return dot product
  return dot_product;
}

void computeContextHash(char *hash, const char **context_words, int num_words) {
  unsigned int hash_val = 5381; // djb2 hash

  for (int i = 0; i < num_words; i++) {
    for (int j = 0; context_words[i][j]; j++) {
      hash_val = ((hash_val << 5) + hash_val) + context_words[i][j];
    }
    hash_val = ((hash_val << 5) + hash_val) + i; // Position matters
  }

  snprintf(hash, 32, "%u", hash_val);
}

SparseEmbedding *getContextualEmbedding(const char *word, const char **context,
                                        int context_len) {
  int word_idx = -1;
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(word, vocabulary[i].word) == 0) {
      word_idx = i;
      break;
    }
  }

  if (word_idx == -1)
    return NULL;

  char context_hash[32];
  computeContextHash(context_hash, context, context_len);

  int cache_start = word_idx * 4;
  for (int i = 0; i < 4; i++) {
    if (strcmp(context_cache[cache_start + i].context_hash, context_hash) ==
        0) {
      context_cache[cache_start + i].recency = 1.0f;
      return &context_cache[cache_start + i].embedding;
    }
  }

  int cache_idx = cache_start;
  float min_recency = 1.0f;

  for (int i = 1; i < 4; i++) {
    if (context_cache[cache_start + i].recency < min_recency) {
      min_recency = context_cache[cache_start + i].recency;
      cache_idx = cache_start + i;
    }
  }

  if (context_cache[cache_idx].embedding.active_dims) {
    free(context_cache[cache_idx].embedding.active_dims);
    free(context_cache[cache_idx].embedding.values);
  }

  SparseEmbedding *base = &word_embeddings[word_idx];
  SparseEmbedding *contextual = &context_cache[cache_idx].embedding;

  contextual->num_active = base->num_active;
  contextual->active_dims = (int *)malloc(contextual->num_active * sizeof(int));
  contextual->values = (float *)malloc(contextual->num_active * sizeof(float));

  memcpy(contextual->active_dims, base->active_dims,
         base->num_active * sizeof(int));
  memcpy(contextual->values, base->values, base->num_active * sizeof(float));
  memcpy(contextual->semantic_layer, base->semantic_layer,
         NUM_SEMANTIC_LAYERS * sizeof(int));

  for (int c = 0; c < context_len && c < CONTEXT_WINDOW; c++) {
    int context_word_idx = -1;
    for (int i = 0; i < vocab_size; i++) {
      if (strcmp(context[c], vocabulary[i].word) == 0) {
        context_word_idx = i;
        break;
      }
    }

    if (context_word_idx != -1) {
      SparseEmbedding *context_emb = &word_embeddings[context_word_idx];
      float context_strength = 0.1f / (c + 1);

      for (int i = 0; i < contextual->num_active; i++) {
        for (int j = 0; j < context_emb->num_active; j++) {
          if (contextual->active_dims[i] == context_emb->active_dims[j]) {
            contextual->values[i] += context_emb->values[j] * context_strength;
            break;
          }
        }
      }
    }
  }

  contextual->norm = 0.0f;
  for (int i = 0; i < contextual->num_active; i++) {
    contextual->norm += contextual->values[i] * contextual->values[i];
  }
  contextual->norm = sqrt(contextual->norm);

  if (contextual->norm > 1e-8f) {
    for (int i = 0; i < contextual->num_active; i++) {
      contextual->values[i] /= contextual->norm;
    }
    contextual->norm = 1.0f;
  }

  strcpy(context_cache[cache_idx].context_hash, context_hash);
  context_cache[cache_idx].recency = 1.0f;

  return contextual;
}

void initializeBrainInspiredEmbeddings(const char *pretrained_file) {
  if (safe_vocab_size <= 0) {
    fprintf(stderr, "Error: Vocabulary not loaded\n");
    return;
  }

  printf("Initializing brain-inspired sparse embedding system...\n");

  for (int layer = 0; layer < NUM_SEMANTIC_LAYERS; layer++) {
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      float freq = 0.01f * (layer + 1);
      semantic_weights[layer][i] = std::sin(i * freq) * std::exp(-layer * 0.1f);
    }
  }

  for (int i = 0; i < safe_vocab_size; i++) {
    initializeSparseEmbedding(&word_embeddings[i], i);
  }

  std::memset(context_cache, 0, sizeof(context_cache));

  for (int i = 0; i < safe_vocab_size; i++) {
    for (int bucket = 0; bucket < HASH_BUCKETS; bucket++) {
      similarity_hash[bucket][i] = 0.0f;

      if (word_embeddings[i].num_active > 0 &&
          word_embeddings[i].active_dims != nullptr) {
        for (int j = 0; j < word_embeddings[i].num_active; j++) {
          int dim = word_embeddings[i].active_dims[j];
          if ((dim * 17 + bucket * 23) % 2) {
            similarity_hash[bucket][i] += word_embeddings[i].values[j];
          }
        }
      }
    }
  }

  printf("Brain-inspired embedding system initialized with %d "
         "sparse vectors\n",
         safe_vocab_size);
  printf("Average sparsity: %.1f%% (%.0f active dims per word)\n",
         SPARSE_DENSITY * 100, EMBEDDING_SIZE * SPARSE_DENSITY);
}

void initializeEmbeddings(const char *embedding_file) {
  if (safe_vocab_size <= 0) {
    fprintf(stderr, "Error: Load vocabulary before initializing "
                    "embeddings\n");
    return;
  }

  importPretrainedEmbeddings(embedding_file);
  initializeBrainInspiredEmbeddings(embedding_file);
}

void cleanupVocabulary() {
  for (int i = 0; i < safe_vocab_size; i++) {
    if (vocabulary[i].connects_to) {
      free(vocabulary[i].connects_to);
      vocabulary[i].connects_to = nullptr;
    }
    if (vocabulary[i].description) {
      free(vocabulary[i].description);
      vocabulary[i].description = nullptr;
    }
  }
}

void cleanupEmbeddings() {
  for (int i = 0; i < safe_vocab_size; i++) {
    if (word_embeddings[i].active_dims) {
      delete[] word_embeddings[i].active_dims;
      word_embeddings[i].active_dims = nullptr;
    }
    if (word_embeddings[i].values) {
      delete[] word_embeddings[i].values;
      word_embeddings[i].values = nullptr;
    }
  }

  for (int i = 0; i < VOCAB_SIZE * 4; i++) {
    if (context_cache[i].embedding.active_dims) {
      delete[] context_cache[i].embedding.active_dims;
      context_cache[i].embedding.active_dims = nullptr;
    }
    if (context_cache[i].embedding.values) {
      delete[] context_cache[i].embedding.values;
      context_cache[i].embedding.values = nullptr;
    }
  }
}

float *getWordEmbedding(const char *word, const char **context,
                        int context_len) {
  static float contextual_embedding[EMBEDDING_SIZE];
  int word_index = -1;

  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(word, vocabulary[i].word) == 0) {
      word_index = i;
      break;
    }
  }

  if (word_index == -1) {
    memset(contextual_embedding, 0, EMBEDDING_SIZE * sizeof(float));

    SparseEmbedding *sparse_emb =
        getContextualEmbedding(word, context, context_len);
    if (sparse_emb) {
      for (int i = 0; i < sparse_emb->num_active; i++) {
        contextual_embedding[sparse_emb->active_dims[i]] =
            sparse_emb->values[i];
      }
    } else {
      char *subword_tokens[INPUT_SIZE];
      int num_subwords = 0;
      tokenizeString(word, subword_tokens, &num_subwords);

      if (num_subwords == 0) {
        size_t len = strlen(word);
        for (size_t i = 0; i < len; i++) {
          for (size_t n = 1; n <= 3 && i + n <= len; n++) {
            unsigned int hash = 0;
            for (size_t j = i; j < i + n; j++) {
              hash = hash * 101 + word[j];
            }
            for (int j = 0; j < EMBEDDING_SIZE / 10; j++) {
              int idx = (hash + j) % EMBEDDING_SIZE;
              contextual_embedding[idx] +=
                  letter_weights[(word[i] - 'a') % 26] / (float)len;
            }
          }
        }
      } else {
        for (int i = 0; i < num_subwords; i++) {
          for (int k = 0; k < vocab_size; k++) {
            if (strcmp(subword_tokens[i], vocabulary[k].word) == 0) {
              for (int j = 0; j < EMBEDDING_SIZE; j++) {
                contextual_embedding[j] += embeddings[k][j];
              }
              break;
            }
          }
          free(subword_tokens[i]);
        }
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
          contextual_embedding[j] /= (float)num_subwords;
        }
      }
    }

    float norm = 0.0f;
    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      norm += contextual_embedding[j] * contextual_embedding[j];
    }
    norm = sqrtf(norm);
    if (norm > 1e-8f) {
      for (int j = 0; j < EMBEDDING_SIZE; j++) {
        contextual_embedding[j] /= norm;
      }
    }
  } else {
    memcpy(contextual_embedding, embeddings[word_index],
           sizeof(float) * EMBEDDING_SIZE);

    if (context_len > 0) {
      SparseEmbedding *context_emb =
          getContextualEmbedding(word, context, context_len);
      if (context_emb) {
        for (int i = 0; i < context_emb->num_active; i++) {
          int dim = context_emb->active_dims[i];
          contextual_embedding[dim] += context_emb->values[i] * 0.1f;
        }
      }
    }

    float complexity_factor =
        strlen(vocabulary[word_index].description) / 50.0f;
    float scaling_factor = (vocabulary[word_index].semantic_weight +
                            vocabulary[word_index].letter_weight) *
                           (1.0f + complexity_factor);
    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      contextual_embedding[j] *= scaling_factor;
    }

    float norm = 0.0f;
    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      norm += contextual_embedding[j] * contextual_embedding[j];
    }
    norm = sqrtf(norm);
    if (norm > 1e-8f) {
      for (int j = 0; j < EMBEDDING_SIZE; j++) {
        contextual_embedding[j] /= norm;
      }
    }
  }

  return contextual_embedding;
}

void updateEmbeddings(float *feedback, const char *word) {
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(word, vocabulary[i].word) == 0) {
      for (int j = 0; j < EMBEDDING_SIZE; j++) {
        embeddings[i][j] += feedback[j];
        embeddings[i][j] = fmaxf(0.0f, fminf(1.0f, embeddings[i][j]));
      }
      break;
    }
  }
}

void findWordsByCategory(const char *category) {
  printf("Words in category '%s':\n", category);
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(vocabulary[i].category, category) == 0) {
      printf("- %s (semantic weight: %.2f): %s\n", vocabulary[i].word,
             vocabulary[i].semantic_weight, vocabulary[i].description);
    }
  }
}

float cosineSimilarity(float *vec1, float *vec2, int size) {
  float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
  for (int i = 0; i < size; i++) {
    dot += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }
  return dot / (sqrtf(norm1) * sqrtf(norm2));
}

// Initialize attention parameters with Xavier/Glorot initialization
void initializeAttentionParams(AttentionParams *params) {
  if (params->initialized)
    return;

  float xavier_scale = sqrtf(2.0f / EMBEDDING_SIZE);

  // Initialize projection weights
  for (int h = 0; h < NUM_HEADS; h++) {
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      for (int j = 0; j < HEAD_DIM; j++) {
        params->query_weights[h][i][j] =
            ((float)rand() / RAND_MAX - 0.5f) * 2.0f * xavier_scale;
        params->key_weights[h][i][j] =
            ((float)rand() / RAND_MAX - 0.5f) * 2.0f * xavier_scale;
        params->value_weights[h][i][j] =
            ((float)rand() / RAND_MAX - 0.5f) * 2.0f * xavier_scale;
      }
    }
  }

  // Initialize output projection
  for (int i = 0; i < EMBEDDING_SIZE; i++) {
    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      params->output_weights[i][j] =
          ((float)rand() / RAND_MAX - 0.5f) * 2.0f * xavier_scale;
    }
  }

  // Initialize positional encoding (sinusoidal)
  for (int pos = 0; pos < INPUT_SIZE; pos++) {
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      if (i % 2 == 0) {
        params->positional_encoding[pos][i] =
            sinf(pos / powf(10000.0f, (float)i / EMBEDDING_SIZE));
      } else {
        params->positional_encoding[pos][i] =
            cosf(pos / powf(10000.0f, (float)(i - 1) / EMBEDDING_SIZE));
      }
    }
  }

  params->initialized = 1;
}

// Apply dropout (simple bernoulli dropout)
void applyDropout(float *values, int size, float dropout_rate, int training) {
  if (!training || dropout_rate <= 0.0f)
    return;

  float keep_prob = 1.0f - dropout_rate;
  for (int i = 0; i < size; i++) {
    if ((float)rand() / RAND_MAX > keep_prob) {
      values[i] = 0.0f;
    } else {
      values[i] /= keep_prob; // Scale to maintain expected value
    }
  }
}

void layerNorm(float *input, float *output, int size) {
  // Calculate mean
  float mean = 0.0f;
  for (int i = 0; i < size; i++) {
    mean += input[i];
  }
  mean /= size;

  // Calculate variance
  float variance = 0.0f;
  for (int i = 0; i < size; i++) {
    float diff = input[i] - mean;
    variance += diff * diff;
  }
  variance /= size;

  // Normalize
  float std = sqrtf(variance + 1e-6f); // Add epsilon for numerical stability
  for (int i = 0; i < size; i++) {
    output[i] = (input[i] - mean) / std;
  }
}

void computeAttentionWeights(float *attention_weights, int step, int num_tokens,
                             float **token_embeddings,
                             MemoryEntry *relevantMemory) {

  initializeAttentionParams(&g_attention_params);

  // Add positional encoding to embeddings
  float enhanced_embeddings[INPUT_SIZE][EMBEDDING_SIZE];
  for (int i = 0; i < num_tokens; i++) {
    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      enhanced_embeddings[i][j] =
          token_embeddings[i][j] + g_attention_params.positional_encoding[i][j];
    }
  }

  // Multi-head attention computation
  float head_outputs[NUM_HEADS][INPUT_SIZE][HEAD_DIM];
  float final_attention_weights[INPUT_SIZE] = {0};

  for (int head = 0; head < NUM_HEADS; head++) {
    // Project to queries, keys, values for this head
    float queries[INPUT_SIZE][HEAD_DIM];
    float keys[INPUT_SIZE][HEAD_DIM];
    float values[INPUT_SIZE][HEAD_DIM];

    for (int i = 0; i < num_tokens; i++) {
      // Query projection
      for (int j = 0; j < HEAD_DIM; j++) {
        queries[i][j] = 0.0f;
        keys[i][j] = 0.0f;
        values[i][j] = 0.0f;

        for (int k = 0; k < EMBEDDING_SIZE; k++) {
          queries[i][j] += enhanced_embeddings[i][k] *
                           g_attention_params.query_weights[head][k][j];
          keys[i][j] += enhanced_embeddings[i][k] *
                        g_attention_params.key_weights[head][k][j];
          values[i][j] += enhanced_embeddings[i][k] *
                          g_attention_params.value_weights[head][k][j];
        }
      }
    }

    // Memory-augmented query if available
    float memory_query[HEAD_DIM] = {0};
    if (relevantMemory) {
      for (int j = 0; j < HEAD_DIM; j++) {
        for (int k = 0; k < EMBEDDING_SIZE && k < MAX_NEURONS; k++) {
          memory_query[j] += relevantMemory->vector[k] *
                             g_attention_params.query_weights[head][k][j];
        }
      }

      // Blend memory query with current step query
      float blend_factor = 0.3f;
      int current_idx = step % num_tokens;
      for (int j = 0; j < HEAD_DIM; j++) {
        queries[current_idx][j] =
            (1.0f - blend_factor) * queries[current_idx][j] +
            blend_factor * memory_query[j];
      }
    }

    // Compute attention scores for this head
    float head_attention_scores[INPUT_SIZE][INPUT_SIZE];
    float scale_factor = 1.0f / sqrtf(HEAD_DIM);

    for (int i = 0; i < num_tokens; i++) {
      float max_score = -INFINITY;

      // Compute raw attention scores
      for (int j = 0; j < num_tokens; j++) {
        float dot_product = 0.0f;
        for (int k = 0; k < HEAD_DIM; k++) {
          dot_product += queries[i][k] * keys[j][k];
        }
        head_attention_scores[i][j] = dot_product * scale_factor;

        // Apply causal mask (prevent attending to future tokens)
        if (j > i) {
          head_attention_scores[i][j] = -INFINITY;
        }

        if (head_attention_scores[i][j] > max_score) {
          max_score = head_attention_scores[i][j];
        }
      }

      // Apply softmax
      float sum_exp = 0.0f;
      for (int j = 0; j < num_tokens; j++) {
        head_attention_scores[i][j] =
            expf(head_attention_scores[i][j] - max_score);
        sum_exp += head_attention_scores[i][j];
      }

      if (sum_exp > 1e-8f) {
        for (int j = 0; j < num_tokens; j++) {
          head_attention_scores[i][j] /= sum_exp;
        }
      }
    }

    // Apply dropout to attention weights (if training)
    int training = 0; // Set to 1 during training
    for (int i = 0; i < num_tokens; i++) {
      applyDropout(head_attention_scores[i], num_tokens, DROPOUT_RATE,
                   training);
    }

    // Compute weighted sum of values
    for (int i = 0; i < num_tokens; i++) {
      for (int k = 0; k < HEAD_DIM; k++) {
        head_outputs[head][i][k] = 0.0f;
        for (int j = 0; j < num_tokens; j++) {
          head_outputs[head][i][k] +=
              head_attention_scores[i][j] * values[j][k];
        }
      }
    }

    // Accumulate attention weights from all heads (for output)
    for (int i = 0; i < num_tokens; i++) {
      final_attention_weights[i] +=
          head_attention_scores[step % num_tokens][i] / NUM_HEADS;
    }
  }

  // Concatenate head outputs
  float concatenated[INPUT_SIZE][EMBEDDING_SIZE];
  for (int i = 0; i < num_tokens; i++) {
    for (int head = 0; head < NUM_HEADS; head++) {
      for (int j = 0; j < HEAD_DIM; j++) {
        concatenated[i][head * HEAD_DIM + j] = head_outputs[head][i][j];
      }
    }
  }

  // Apply output projection
  float projected[INPUT_SIZE][EMBEDDING_SIZE];
  for (int i = 0; i < num_tokens; i++) {
    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      projected[i][j] = 0.0f;
      for (int k = 0; k < EMBEDDING_SIZE; k++) {
        projected[i][j] +=
            concatenated[i][k] * g_attention_params.output_weights[k][j];
      }
    }
  }

  // Apply layer normalization to final output
  float normalized[EMBEDDING_SIZE];
  int current_idx = step % num_tokens;
  layerNorm(projected[current_idx], normalized, EMBEDDING_SIZE);

  // Copy final attention weights to output
  for (int i = 0; i < num_tokens; i++) {
    attention_weights[i] = final_attention_weights[i];
  }

  // Optional: Apply temperature scaling for more/less focused attention
  float temperature =
      1.0f; // Adjust this for sharper (< 1) or softer (> 1) attention
  if (temperature != 1.0f) {
    float max_weight = -INFINITY;
    for (int i = 0; i < num_tokens; i++) {
      attention_weights[i] /= temperature;
      if (attention_weights[i] > max_weight) {
        max_weight = attention_weights[i];
      }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < num_tokens; i++) {
      attention_weights[i] = expf(attention_weights[i] - max_weight);
      sum_exp += attention_weights[i];
    }

    if (sum_exp > 1e-8f) {
      for (int i = 0; i < num_tokens; i++) {
        attention_weights[i] /= sum_exp;
      }
    }
  }
}

void generateInputTensor(float *input_tensor, int step, const char *text_input,
                         MemoryEntry *relevantMemory,
                         SystemParameters *system_params) {
  float t = step * 0.01f;
  DynamicParameters params = system_params->dynamic_params;

  // Tokenize the text input (modern NLP systems use subword tokenization)
  char *tokens[INPUT_SIZE];
  int num_tokens = 0;
  tokenizeString(text_input, tokens, &num_tokens);

  // Get token embeddings with contextual information
  float *token_embeddings[INPUT_SIZE];
  float letter_weights[INPUT_SIZE] = {0};
  float category_weights[INPUT_SIZE] = {0};

  for (int i = 0; i < num_tokens; i++) {
    const char *token_ptrs[num_tokens];
    for (int i = 0; i < num_tokens; i++) {
      token_ptrs[i] = tokens[i]; // for example if each tokens[i] is a char[6],
                                 // decays to char*
    }
    token_embeddings[i] = getWordEmbedding(tokens[i], token_ptrs, num_tokens);
    letter_weights[i] = computeLetterWeight(tokens[i]);

    for (int j = 0; j < vocab_size; j++) {
      if (strcmp(tokens[i], vocabulary[j].word) == 0) {
        // Weight based on category and semantic significance
        if (strcmp(vocabulary[j].category, "action") == 0)
          category_weights[i] = 1.2f;
        else if (strcmp(vocabulary[j].category, "emotion") == 0)
          category_weights[i] = 1.1f;
        else if (strcmp(vocabulary[j].category, "fruit") == 0)
          category_weights[i] = 1.05f;
        else
          category_weights[i] = 1.0f;
        break;
      }
    }
  }

  // Compute attention weights using modern transformer-style attention
  float attention_weights[INPUT_SIZE] = {0};
  computeAttentionWeights(attention_weights, step, num_tokens, token_embeddings,
                          relevantMemory);

  // Apply category and letter weight modifiers to attention weights
  for (int i = 0; i < num_tokens; i++) {
    attention_weights[i] *= category_weights[i] * letter_weights[i];
  }

  // Position encoding (similar to transformer position encoding)
  float position_encoding[INPUT_SIZE][EMBEDDING_SIZE];
  for (int pos = 0; pos < INPUT_SIZE; pos++) {
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      if (i % 2 == 0) {
        position_encoding[pos][i] =
            sinf(pos / powf(10000, i / (float)EMBEDDING_SIZE));
      } else {
        position_encoding[pos][i] =
            cosf(pos / powf(10000, (i - 1) / (float)EMBEDDING_SIZE));
      }
    }
  }

  for (int i = 0; i < INPUT_SIZE; i++) {
    float phase = (float)i / INPUT_SIZE;
    float signal = 0.4f * sinf(2.0f * M_PI * (t + phase));
    signal += 0.4f * sinf(2.0f * M_PI * (t + phase * 1.5f));
    signal += 0.2f * sinf(5.0f * M_PI * (t + phase * 2.0f));

    // Add weighted word embeddings (context-aware representation)
    if (i < EMBEDDING_SIZE) {
      float weighted_embedding = 0.0f;
      for (int j = 0; j < num_tokens && j < INPUT_SIZE; j++) {
        float position_factor = position_encoding[j][i]; // Position encoding

        // Find description length for complexity factor
        int desc_length = 0;
        for (int k = 0; k < vocab_size; k++) {
          if (j < num_tokens && strcmp(tokens[j], vocabulary[k].word) == 0) {
            desc_length = strlen(vocabulary[k].description);
            break;
          }
        }

        // Incorporate description complexity (custom logic)
        float desc_factor = 1.0f + (desc_length / 100.0f);

        // Add weighted contribution from token
        if (j < num_tokens) {
          weighted_embedding += attention_weights[j] *
                                token_embeddings[j][i % EMBEDDING_SIZE] *
                                desc_factor * position_factor;
        }
      }

      // Add the weighted embedding contribution
      signal += 0.3f * weighted_embedding;
    }

    // Add memory-based relevance
    if (relevantMemory) {
      signal += 0.2f * relevantMemory->vector[i % MAX_NEURONS];
    }

    // Noise and drift management (preserved from original)
    float noise = ((float)rand() / RAND_MAX - 0.5f) * params.input_noise_scale;
    float drift = params.plasticity * sinf(0.1f * M_PI * t);

    // Combine all factors and normalize
    input_tensor[i] = (signal + noise + drift + 1.0f) * 0.5f;

    // Clamp to valid range
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
    for (size_t j = 0; j < neurons[i].num_connections; j++) {
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
    for (size_t j = 0; j < neurons[i].num_connections; j++) {
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
  PatternMatch *matches =
      new PatternMatch[cluster->size *
                       sizeof(PatternMatch)]; // Use 'new' instead of malloc
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

          // Allocate and zero-init to prevent garbage values
          entry->features = (float *)calloc(FEATURE_VECTOR_SIZE, sizeof(float));
          entry->context_vector =
              (float *)calloc(CONTEXT_VECTOR_SIZE, sizeof(float));

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
      memmove(&working_memory->active.entries[i],
              &working_memory->active.entries[i + 1],
              (working_memory->active.size - i - 1) *
                  sizeof(WorkingMemoryEntry));
      working_memory->active.size--;
      i--; // Adjust index after removal
    }
  }

  // Only update clusters if there is at least one focused entry
  if (working_memory->focus.size > 0) {
    updateSemanticClusters(working_memory, &working_memory->focus.entries[0]);
  }

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

    if (!isfinite(attention_weight) || attention_weight < 0.0f) {
      printf("Invalid attention weight in focus[%u], skipping\n", i);
      continue;
    }

    for (unsigned int j = 0; j < FEATURE_VECTOR_SIZE; j++) {
      unsigned int neuron_idx = j % MAX_NEURONS;

      if (!isfinite(focused_item->features[j])) {
        printf("NaN in focused_item->features[%u], setting to 0\n", j);
        focused_item->features[j] = 0.0f;
      }

      // Clamp to a reasonable value
      float feature_val = clampValue(focused_item->features[j]);

      float influence =
          feature_val * attention_weight * 0.01f; // 0.01f damping factor

      neurons[neuron_idx].state *= 0.99f;

      neurons[neuron_idx].state += influence;
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

      float feature_val = clampValue(active_item->features[j]);

      float delta = feature_val * activation * 0.001f; // smaller damping factor

      // Optional: decay weight slightly to prevent runaway
      weights[weight_idx] *= 0.999f;

      // Apply controlled update
      weights[weight_idx] += delta;

      // Optional: clamp weights to prevent explosion
      weights[weight_idx] = clampValue(weights[weight_idx]);
    }
  }

  // Apply sanitized semantic cluster influence
  for (unsigned int i = 0; i < working_memory->clusters.num_clusters; i++) {
    SemanticCluster *cluster = &working_memory->clusters.clusters[i];

    if (!isfinite(cluster->coherence)) {
      printf("Skipping cluster[%u] due to invalid coherence\n", i);
      continue;
    }

    cluster->coherence = clampValue(cluster->coherence);

    if (cluster->coherence > 0.7f) {
      for (unsigned int j = 0; j < FEATURE_VECTOR_SIZE; j++) {
        unsigned int neuron_idx = j % MAX_NEURONS;

        if (!isfinite(cluster->vector[j])) {
          printf("NaN in cluster->vector[%u] of cluster[%u], setting to 0\n", j,
                 i);
          cluster->vector[j] = 0.0f;
        }

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
      new NetworkPerformanceMetrics(); // Use 'new' instead of malloc
  metrics->num_regions = num_regions;
  metrics->region_performance_scores = new float[num_regions];
  metrics->region_error_rates = new float[num_regions];
  metrics->region_output_variance = new float[num_regions];

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
  MetaController *controller =
      static_cast<MetaController *>(malloc(sizeof(MetaController)));
  controller->meta_learning_rate = 0.01;
  controller->exploration_factor = 0.1;
  controller->num_regions = num_regions;
  controller->region_importance_scores = new float[num_regions];
  controller->learning_efficiency_history = new float[num_regions];

  // Initialize with equal importance
  for (int i = 0; i < num_regions; i++) {
    controller->region_importance_scores[i] = 1.0f / num_regions;
  }

  return controller;
}

void printReplayStatistics(MemorySystem *memorySystem) {
  if (memorySystem == NULL || memorySystem->size == 0) {
    printf("No memories available for replay.\n");
    return;
  }

  float total_importance = 0.0f;
  int num_replayed = 0;

  // Iterate through memory entries to calculate statistics
  for (size_t i = 0; i < memorySystem->size; i++) {
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
      .num_connections = MAX_CONNECTIONS,
      .layer_id =
          static_cast<unsigned int>(*num_neurons) % 2 // Alternate layers
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

    for (size_t j = 0; j < neurons[i].num_connections; j++) {
      total_connection_strength += fabs(weights[i * MAX_CONNECTIONS + j]);
      mean_weight += weights[i * MAX_CONNECTIONS + j];
    }
    mean_weight /= neurons[i].num_connections;

    for (size_t j = 0; j < neurons[i].num_connections; j++) {
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
  NeuronPerformanceMetric *metrics = new NeuronPerformanceMetric;

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

MetacognitionMetrics *initializeMetacognitionMetrics() {
  MetacognitionMetrics *metacog = new MetacognitionMetrics;
  if (!metacog)
    return NULL;

  metacog->confidence_level = 0.5f;  // Start with moderate confidence
  metacog->adaptation_rate = 0.1f;   // Conservative initial adaptation
  metacog->cognitive_load = 0.0f;    // Start with minimal load
  metacog->error_awareness = 0.0f;   // Initial error awareness
  metacog->context_relevance = 1.0f; // Start with full context relevance

  // Initialize performance history
  for (int i = 0; i < HISTORY_LENGTH; i++) {
    metacog->performance_history[i] = 0.0f;
  }

  return metacog;
}

// Initialize meta learning state
MetaLearningState *initializeMetaLearningState(int num_regions) {
  MetaLearningState *state = new MetaLearningState;
  if (!state)
    return NULL;

  state->learning_efficiency = 1.0f;
  state->exploration_rate = 0.2f;
  state->stability_index = 1.0f;
  state->current_phase = 0;

  // Allocate and initialize priority weights
  state->priority_weights = new float[num_regions * sizeof(float)];
  if (!state->priority_weights) {
    free(state);
    return NULL;
  }

  for (int i = 0; i < num_regions; i++) {
    state->priority_weights[i] = 1.0f / num_regions;
  }

  return state;
}

float computePerformanceVariance(float *history, int length) {
  float mean = 0.0f;
  float variance = 0.0f;

  // Calculate mean
  for (int i = 0; i < length; i++) {
    mean += history[i];
  }
  mean /= length;

  // Calculate variance
  for (int i = 0; i < length; i++) {
    float diff = history[i] - mean;
    variance += diff * diff;
  }
  variance /= length;

  return variance;
}

float computeAdaptiveRate(NetworkPerformanceMetrics *performance,
                          float *history) {
  float recent_trend = 0.0f;

  // Calculate trend from recent performance
  for (int i = 1; i < HISTORY_LENGTH; i++) {
    recent_trend += history[i] - history[i - 1];
  }

  // Normalize trend and convert to adaptation rate
  float base_rate = 0.1f;
  float trend_factor = tanh(recent_trend);

  return base_rate * (1.0f + trend_factor);
}

float computeErrorAwareness(NetworkPerformanceMetrics *performance,
                            float *history) {
  float total_error = 0.0f;
  float max_error = 0.0f;

  // Calculate total and maximum prediction errors
  for (int i = 0; i < performance->num_regions; i++) {
    float error = fabs(performance->region_performance_scores[i] -
                       history[i % HISTORY_LENGTH]);
    total_error += error;
    max_error = fmaxf(max_error, error);
  }

  // Normalize error awareness
  float avg_error = total_error / performance->num_regions;
  return (avg_error + max_error) / 2.0f;
}

float evaluateContextRelevance(MetaController *controller,
                               NetworkPerformanceMetrics *performance) {
  float total_relevance = 0.0f;

  // Compute weighted relevance across regions
  for (int i = 0; i < controller->num_regions; i++) {
    float region_performance = performance->region_performance_scores[i];
    float importance = controller->region_importance_scores[i];
    total_relevance += region_performance * importance;
  }

  return total_relevance / controller->num_regions;
}

void mutatePathParameters(DecisionPath *path, int max_neurons, int step) {
  // Randomly modify some weights and connections
  int num_mutations = max_neurons / 10; // Mutate 10% of parameters

  for (int i = 0; i < num_mutations; i++) {
    int idx = rand() % (max_neurons * MAX_CONNECTIONS);
    path->weights[idx] *= 1.0f + ((rand() / (float)RAND_MAX - 0.5f) * 0.2f);

    if (rand() / (float)RAND_MAX < 0.1f) { // 10% chance to rewire
      path->connections[idx] = rand() % max_neurons;
    }
  }
}

DecisionPath generateDecisionPath(Neuron *neurons, float *weights,
                                  int *connections, float *input_tensor,
                                  int max_neurons, float explore_rate) {
  DecisionPath path;

  // Allocate memory for path components
  path.states = new float[max_neurons * MAX_DECISION_STEPS * sizeof(float)];
  path.weights = new float[sizeof(float) * max_neurons * MAX_CONNECTIONS];
  path.connections = new int[sizeof(int) * max_neurons * MAX_CONNECTIONS];
  path.num_steps = 0;
  path.score = 0.0f;

  // Copy initial states and parameters
  memcpy(path.weights, weights, sizeof(float) * max_neurons * MAX_CONNECTIONS);
  memcpy(path.connections, connections,
         sizeof(int) * max_neurons * MAX_CONNECTIONS);

  // Generate future states with exploration
  for (int step = 0; step < MAX_DECISION_STEPS; step++) {
    float *current_state = &path.states[step * max_neurons];

    // Apply exploration noise
    for (int i = 0; i < max_neurons; i++) {
      float noise = (rand() / (float)RAND_MAX - 0.5f) * explore_rate;
      current_state[i] = neurons[i].state * (1.0f + noise);
    }

    // Update connections and weights with exploration
    if (rand() / (float)RAND_MAX < explore_rate) {
      mutatePathParameters(&path, max_neurons, step);
    }

    path.num_steps++;
  }

  return path;
}

DecisionPath selectBestPath(DecisionPath *paths, int num_paths) {
  int best_idx = 0;
  float best_score = paths[0].score;

  for (int i = 1; i < num_paths; i++) {
    if (paths[i].score > best_score) {
      best_score = paths[i].score;
      best_idx = i;
    }
  }

  return paths[best_idx];
}

void applyDecisionPath(DecisionPath path, Neuron *neurons, float *weights,
                       int *connections, float confidence) {
  // Apply state changes with confidence-based modulation
  for (int i = 0; i < path.num_steps; i++) {
    float *target_state = &path.states[i];
    float modulation = confidence * (1.0f - ((float)i / path.num_steps));

    for (int j = 0; j < MAX_NEURONS; j++) {
      neurons[j].state += (target_state[j] - neurons[j].state) * modulation;
    }
  }

  // Apply weight and connection changes
  for (int i = 0; i < MAX_NEURONS * MAX_CONNECTIONS; i++) {
    weights[i] += (path.weights[i] - weights[i]) * confidence;
    if (confidence >
        0.8f) { // Only apply connection changes with high confidence
      connections[i] = path.connections[i];
    }
  }
}

float evaluatePathQuality(DecisionPath path, MetaLearningState *meta_state,
                          MetacognitionMetrics *metacog) {
  float quality_score = 0.0f;
  float stability_score = 0.0f;
  float efficiency_score = 0.0f;

  // Evaluate state stability
  for (int step = 1; step < path.num_steps; step++) {
    float step_diff = 0.0f;
    for (int i = 0; i < MAX_NEURONS; i++) {
      float diff = path.states[step * MAX_NEURONS + i] -
                   path.states[(step - 1) * MAX_NEURONS + i];
      step_diff += diff * diff;
    }
    stability_score += sqrtf(step_diff);
  }
  stability_score = 1.0f / (1.0f + stability_score / path.num_steps);

  // Evaluate learning efficiency
  float weight_changes = 0.0f;
  for (int i = 0; i < MAX_NEURONS * MAX_CONNECTIONS; i++) {
    weight_changes += fabs(path.weights[i] - meta_state->priority_weights[i]);
  }
  efficiency_score = 1.0f / (1.0f + weight_changes);

  // Combine scores with metacognitive awareness
  quality_score = (stability_score * meta_state->stability_index +
                   efficiency_score * meta_state->learning_efficiency) *
                  (1.0f - metacog->cognitive_load);

  return quality_score;
}

void updateMetaLearningState(MetaLearningState *state, DecisionPath best_path,
                             MetacognitionMetrics *metacog) {
  // Update learning efficiency based on path quality
  float path_efficiency = evaluatePathQuality(best_path, state, metacog);
  state->learning_efficiency =
      state->learning_efficiency * 0.9f + path_efficiency * 0.1f;

  // Adjust exploration rate based on performance
  if (path_efficiency > state->learning_efficiency) {
    // Reduce exploration when performing well
    state->exploration_rate *= 0.95f;
  } else {
    // Increase exploration when performance is poor
    state->exploration_rate = fmin(state->exploration_rate * 1.05f, 0.5f);
  }

  // Update stability index
  float stability_measure = 0.0f;
  for (int step = 1; step < best_path.num_steps; step++) {
    float step_stability = 0.0f;
    for (int i = 0; i < MAX_NEURONS; i++) {
      float diff = best_path.states[step * MAX_NEURONS + i] -
                   best_path.states[(step - 1) * MAX_NEURONS + i];
      step_stability += diff * diff;
    }
    stability_measure += sqrtf(step_stability);
  }

  state->stability_index =
      1.0f / (1.0f + stability_measure / best_path.num_steps);

  // Update priority weights based on path performance
  for (int i = 0; i < MAX_NEURONS; i++) {
    float weight_delta = 0.0f;
    for (int step = 0; step < best_path.num_steps; step++) {
      weight_delta += best_path.states[step * MAX_NEURONS + i];
    }
    weight_delta /= best_path.num_steps;

    state->priority_weights[i] =
        state->priority_weights[i] * 0.9f + weight_delta * 0.1f;
  }

  // Update learning phase
  state->current_phase = (state->current_phase + 1) % 4; // 4 phases of learning
}

float assessCognitiveLoad(MetaController *controller,
                          NetworkPerformanceMetrics *performance) {
  float total_complexity = 0.0f;
  float activation_entropy = 0.0f;
  float weight_complexity = 0.0f;
  float temporal_complexity = 0.0f;

  for (int i = 0; i < controller->num_regions; i++) {
    float importance = controller->region_importance_scores[i];
    if (importance > 0) {
      activation_entropy -= importance * log2f(importance);
    }

    float region_weight_var = 0.0f;
    float mean_weight = 0.0f;
    int connections_count = 0;

    for (int j = 0; j < controller->num_regions; j++) {
      if (controller->region_importance_scores[j] > 0) {
        mean_weight += controller->region_importance_scores[j];
        connections_count++;
      }
    }

    if (connections_count > 0) {
      mean_weight /= connections_count;
      for (int j = 0; j < controller->num_regions; j++) {
        if (controller->region_importance_scores[j] > 0) {
          float diff = controller->region_importance_scores[j] - mean_weight;
          region_weight_var += diff * diff;
        }
      }
      weight_complexity += sqrtf(region_weight_var / connections_count);
    }

    float temporal_diff = 0.0f;
    if (i < performance->num_regions) {
      temporal_diff = fabs(performance->region_performance_scores[i] -
                           controller->learning_efficiency_history[i]);
    }
    temporal_complexity += temporal_diff;
  }

  activation_entropy = activation_entropy / log2f(controller->num_regions);
  weight_complexity = weight_complexity / controller->num_regions;
  temporal_complexity = temporal_complexity / controller->num_regions;

  float cognitive_load = (0.4f * activation_entropy + 0.3f * weight_complexity +
                          0.3f * temporal_complexity);

  cognitive_load = 1.0f / (1.0f + expf(-cognitive_load));

  return fminf(1.0f, fmaxf(0.0f, cognitive_load));
}

void updateMetacognitionMetrics(MetacognitionMetrics *metacog,
                                MetaController *controller,
                                NetworkPerformanceMetrics *performance) {
  float performance_variance =
      computePerformanceVariance(metacog->performance_history, HISTORY_LENGTH);
  metacog->confidence_level = 1.0f / (1.0f + performance_variance);

  metacog->adaptation_rate =
      computeAdaptiveRate(performance, metacog->performance_history);

  metacog->cognitive_load = assessCognitiveLoad(controller, performance);

  metacog->error_awareness =
      computeErrorAwareness(performance, metacog->performance_history);

  metacog->context_relevance =
      evaluateContextRelevance(controller, performance);
}

void updateMetaControllerPriorities(MetaController *controller,
                                    NetworkPerformanceMetrics *performance,
                                    MetacognitionMetrics *metacog) {
  float performance_trend = 0.0f;
  for (int i = 0; i < HISTORY_LENGTH - 1; i++) {
    performance_trend +=
        metacog->performance_history[i + 1] - metacog->performance_history[i];
  }

  int max_regions = controller->num_regions < performance->num_regions
                        ? controller->num_regions
                        : performance->num_regions;

  for (int i = 0; i < max_regions; i++) {
    float learning_delta = performance->region_performance_scores[i] -
                           controller->learning_efficiency_history[i];

    float adaptive_rate = controller->meta_learning_rate *
                          (1.0f + metacog->adaptation_rate) *
                          (1.0f - metacog->cognitive_load);

    float dynamic_exploration = controller->exploration_factor *
                                (1.0f + performance_trend) *
                                metacog->confidence_level;

    controller->region_importance_scores[i] += adaptive_rate * learning_delta *
                                               (1.0f + dynamic_exploration) *
                                               metacog->context_relevance;

    float load_factor = 1.0f / (1.0f + metacog->cognitive_load);
    controller->region_importance_scores[i] *= load_factor;

    controller->learning_efficiency_history[i] =
        performance->region_performance_scores[i] *
        (1.0f - metacog->error_awareness);
  }

  updateMetacognitionMetrics(metacog, controller, performance);
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
      for (size_t j = 0; j < neurons[i].num_connections; j++) {
        int connection_idx = i * MAX_CONNECTIONS + j;
        // Modulate weights non-linearly with region importance
        weights[connection_idx] *= (1 + region_importance);
      }
    }
  }
}

void selectOptimalMetaDecisionPath(Neuron *neurons, float *weights,
                                   int *connections, float *input_tensor,
                                   int max_neurons,
                                   MetaLearningState *meta_state,
                                   MetacognitionMetrics *metacog) {
  // Adjust exploration rate based on metacognitive state
  float explore_rate = meta_state->exploration_rate *
                       (1.0f - metacog->cognitive_load) *
                       metacog->confidence_level;

  // Generate multiple decision paths with varying parameters
  DecisionPath paths[NUM_PATHS];
  for (int i = 0; i < NUM_PATHS; i++) {
    paths[i] = generateDecisionPath(neurons, weights, connections, input_tensor,
                                    max_neurons, explore_rate);

    // Evaluate path considering metacognitive factors
    paths[i].score = evaluatePathQuality(paths[i], meta_state, metacog);
  }

  // Select best path and update meta-learning state
  DecisionPath best_path = selectBestPath(paths, NUM_PATHS);
  updateMetaLearningState(meta_state, best_path, metacog);

  // Apply selected path with confidence-based modulation
  applyDecisionPath(best_path, neurons, weights, connections,
                    metacog->confidence_level);
}

ContextNode *createContextNode(const char *name, uint32_t vector_size,
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

GlobalContextManager *initializeGlobalContextManager(uint32_t vector_size) {
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
  for (uint32_t i = 0; i < node->vector_size; i++) {
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
    for (uint32_t i = 0; i < node->parent->num_children; i++) {
      ContextNode *sibling = node->parent->children[i];
      float temp_relevance =
          expf(-(time(NULL) - sibling->last_updated) / 3600.0f);
      float weight = sibling->importance * temp_relevance;

      for (uint32_t j = 0; j < node->vector_size; j++) {
        aggregated[j] += sibling->state_vector[j] * weight;
      }
      total_importance += weight;
    }

    // Normalize and update parent
    if (total_importance > 0) {
      for (uint32_t i = 0; i < node->vector_size; i++) {
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

  for (uint32_t i = 0; i < root->num_children; i++) {
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
                                     uint32_t num_neurons) {
  float satisfaction = 1.0f;

  if (strcmp(constraint->name, "ActivityLevel") == 0) {
    float total_activity = 0;
    float variance = 0.0f;

    // Compute total activity and mean
    for (uint32_t i = 0; i < num_neurons; i++) {
      total_activity += neurons[i].output;
    }
    float mean_activity = total_activity / num_neurons;

    // Compute variance
    for (uint32_t i = 0; i < num_neurons; i++) {
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
                         uint32_t num_neurons, float *input_tensor) {
  // Extract relevant features from current network state
  float *current_context = (float *)calloc(manager->vector_size, sizeof(float));

  // Analyze network activity patterns
  for (uint32_t i = 0; i < manager->vector_size && i < num_neurons; i++) {
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
    for (uint32_t i = 0; i < constraints->num_children; i++) {
      float satisfaction = evaluateConstraintSatisfaction(
          constraints->children[i], neurons, num_neurons);
      for (uint32_t j = 0; j < manager->vector_size; j++) {
        constraint_state[j] += satisfaction;
      }
    }
    updateContextNode(constraints, constraint_state, 0.3f);
    free(constraint_state);
  }

  // Update global context vector
  for (uint32_t i = 0; i < manager->vector_size; i++) {
    manager->global_context_vector[i] = 0;
    float total_weight = 0;

    // Weighted combination of all top-level contexts
    for (uint32_t j = 0; j < manager->root->num_children; j++) {
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
                            uint32_t num_neurons, float *weights,
                            uint32_t max_connections) {
  // Compute network-wide entropy as a measure of state variability
  float total_entropy = 0.0f;
  for (uint32_t i = 0; i < num_neurons; i++) {
    total_entropy += fabs(neurons[i].output);
  }
  float network_entropy = total_entropy / num_neurons;
  float context_sensitivity = 1.0f - network_entropy;

  // Modulate neuron behavior based on global context
  for (uint32_t i = 0; i < num_neurons; i++) {
    float context_influence = 0;

    // Compute context influence on this neuron
    for (uint32_t j = 0; j < manager->vector_size && j < num_neurons; j++) {
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

IntrinsicMotivation *initializeMotivationSystem() {
  IntrinsicMotivation *motivation = new IntrinsicMotivation;
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
  GoalSystem *system = (GoalSystem *)malloc(sizeof(GoalSystem));
  system->goals = (Goal *)malloc(sizeof(Goal) * capacity);
  system->num_goals = 0;
  system->capacity = capacity;
  system->planning_horizon = 10.0f;
  system->discount_factor = 0.95f;
  // Add adaptive learning rate bounds
  system->min_learning_rate = 0.001f;
  system->max_learning_rate = 0.1f;
  system->base_learning_rate = 0.01f;
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
  goal->description[255] = '\0'; // Ensure null termination

  // Bound priority
  if (priority > 1.0f)
    priority = 1.0f;
  if (priority < 0.1f)
    priority = 0.1f;
  goal->priority = priority;

  goal->progress = 0.0f;
  goal->previous_progress = 0.0f; // Track previous progress
  goal->reward_value = priority * 10.0f;
  goal->achieved = false;
  goal->timestamp = time(NULL);
  goal->stability_counter =
      0; // Track stability instead of consecutive improvements
}

float evaluateGoalProgress(Goal *goal, const Neuron *neurons, int neuron_count,
                           const float *target_outputs) {
  if (strstr(goal->description, "Minimize prediction error")) {
    float total_error = 0.0f;
    int valid_neurons = 0;

    for (int i = 0; i < neuron_count; i++) {
      // Only evaluate neurons that have a valid target output
      if (i < neuron_count) {
        float error = fabs(neurons[i].output - target_outputs[i]);
        total_error += error;
        valid_neurons++;
      }
    }

    // Prevent division by zero
    if (valid_neurons == 0)
      return 0.0f;

    float normalized_error = total_error / valid_neurons;
    if (normalized_error > 1.0f)
      normalized_error = 1.0f;

    return 1.0f - normalized_error;
  }

  if (strstr(goal->description, "Develop stable representations")) {
    // Without connection information, use layer stability as proxy
    float stability = 0.0f;
    int layers_evaluated = 0;
    int current_layer = -1;
    float layer_sum = 0.0f;
    int layer_count = 0;

    // Group neurons by layer and evaluate stability within layers
    for (int i = 0; i < neuron_count; i++) {
      // When we encounter a new layer
      if ((int)neurons[i].layer_id != current_layer) {
        // Process the previous layer if it exists
        if (current_layer >= 0 && layer_count > 0) {
          float layer_avg = layer_sum / layer_count;
          float layer_variance = 0.0f;

          // Calculate variance within this layer
          for (int j = 0; j < i; j++) {
            if ((int)neurons[j].layer_id == current_layer) {
              float diff = neurons[j].output - layer_avg;
              layer_variance += diff * diff;
            }
          }

          // Lower variance means more stable representations
          if (layer_count > 1) {
            layer_variance /= layer_count;
            stability += 1.0f - fmin(1.0f, sqrt(layer_variance));
            layers_evaluated++;
          }
        }

        // Start new layer
        current_layer = neurons[i].layer_id;
        layer_sum = 0.0f;
        layer_count = 0;
      }

      // Add to current layer
      layer_sum += neurons[i].output;
      layer_count++;
    }

    // Process the last layer
    if (current_layer >= 0 && layer_count > 0) {
      float layer_avg = layer_sum / layer_count;
      float layer_variance = 0.0f;

      for (int j = 0; j < neuron_count; j++) {
        if ((int)neurons[j].layer_id == current_layer) {
          float diff = neurons[j].output - layer_avg;
          layer_variance += diff * diff;
        }
      }

      if (layer_count > 1) {
        layer_variance /= layer_count;
        stability += 1.0f - fmin(1.0f, sqrt(layer_variance));
        layers_evaluated++;
      }
    }

    // Prevent division by zero
    if (layers_evaluated == 0)
      return 0.0f;

    return stability / layers_evaluated;
  }

  if (strstr(goal->description, "Maximize information gain")) {
    float entropy = 0.0f;
    int active_neurons = 0;
    float total_output = 0.0f;

    // First pass: sum outputs for normalization
    for (int i = 0; i < neuron_count; i++) {
      if (neurons[i].output > 0.01f) {
        total_output += neurons[i].output;
        active_neurons++;
      }
    }

    // Calculate normalized entropy if we have active neurons
    if (active_neurons > 0 && total_output > 0.0f) {
      for (int i = 0; i < neuron_count; i++) {
        if (neurons[i].output > 0.01f) {
          // Normalize the output
          float normalized_output = neurons[i].output / total_output;
          // Avoid log(0) by adding small epsilon
          float safe_output = normalized_output * 0.99f + 0.005f;
          entropy -= safe_output * log2f(safe_output);
        }
      }

      // Normalize entropy by maximum possible entropy
      float max_entropy = log2f((float)active_neurons);
      if (max_entropy > 0.0f) {
        return fmin(1.0f, entropy / max_entropy);
      }
    }

    // Default if no entropy could be calculated
    return 0.0f;
  }

  return 0.0f;
}

void updateGoalSystem(GoalSystem *goalSystem, Neuron *neurons, int neuron_count,
                      const float *target_outputs, float *learning_rate) {
  float total_reward = 0.0f;
  float total_priority = 0.0f;

  for (int i = 0; i < goalSystem->num_goals; i++) {
    Goal *goal = &goalSystem->goals[i];

    // Store previous progress
    goal->previous_progress = goal->progress;

    // Evaluate new progress
    float new_progress =
        evaluateGoalProgress(goal, neurons, neuron_count, target_outputs);

    // Calculate progress delta with smoothing to prevent oscillations
    float raw_delta = new_progress - goal->progress;
    float smoothed_delta = raw_delta * 0.7f; // Reduce impact of large changes

    // Update progress
    goal->progress = new_progress;

    // Calculate significance of progress change
    float abs_delta = smoothed_delta;
    if (abs_delta < 0)
      abs_delta = -abs_delta;

    // Track stability instead of just consecutive improvements
    if (abs_delta < 0.01f) {
      // Small change - may be stabilizing
      goal->stability_counter++;
    } else if (smoothed_delta > 0.01f) {
      // Clear improvement
      goal->stability_counter = 0;
    } else {
      // Clear regression
      goal->stability_counter =
          goal->stability_counter > 0 ? goal->stability_counter - 1 : 0;
    }

    // Generate intrinsic reward with sigmoid scaling to control magnitude
    float sigmoid_factor = 1.0f / (1.0f + expf(-5.0f * smoothed_delta));
    float intrinsic_reward =
        smoothed_delta * goal->reward_value * sigmoid_factor;

    // Apply dampening for repeated similar rewards to prevent overfitting
    if (goal->stability_counter > 5) {
      intrinsic_reward *= expf(-0.1f * (float)(goal->stability_counter - 5));
    }

    // Bound reward to prevent extreme learning rate adjustments
    if (intrinsic_reward > 0.5f)
      intrinsic_reward = 0.5f;
    if (intrinsic_reward < -0.5f)
      intrinsic_reward = -0.5f;

    // Accumulate weighted reward
    total_reward += intrinsic_reward * goal->priority;
    total_priority += goal->priority;

    // Check if goal has been achieved based on high progress and stability
    if (goal->progress > 0.95f && goal->stability_counter > 10) {
      goal->achieved = true;
    }
  }

  // Normalize reward by total priority if there are goals
  float net_reward = 0.0f;
  if (total_priority > 0.0f) {
    net_reward = total_reward / total_priority;
  }

  // Apply rewards to learning rate with bounds
  float rate_adjustment = 1.0f + 0.1f * net_reward;
  float new_learning_rate = goalSystem->base_learning_rate * rate_adjustment;

  // Enforce bounds
  if (new_learning_rate > goalSystem->max_learning_rate)
    new_learning_rate = goalSystem->max_learning_rate;
  if (new_learning_rate < goalSystem->min_learning_rate)
    new_learning_rate = goalSystem->min_learning_rate;

  // Apply smoothing to prevent rapid learning rate changes
  *learning_rate = (*learning_rate * 0.9f) + (new_learning_rate * 0.1f);
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

float computeStateSimilarity(Neuron *current_neurons,
                             NetworkStateSnapshot *historical_state) {
  float similarity = 0.0f;
  float total_neurons = (float)MAX_NEURONS;

  for (int i = 0; i < MAX_NEURONS; i++) {
    float output_diff =
        fabs(current_neurons[i].output - historical_state->outputs[i]);
    float state_diff =
        fabs(current_neurons[i].state - historical_state->states[i]);

    // Combine output and state differences
    float neuron_similarity = 1.0f - (output_diff + state_diff) / 2.0f;
    similarity += neuron_similarity;
  }

  return similarity / total_neurons;
}

// Compute consistency between neuron state and memory entry
float computeMemoryConsistency(Neuron *neurons, MemoryEntry *memory) {
  float consistency = 0.0f;
  float total_elements = (float)MAX_NEURONS;

  // Compare neuron outputs with memory vector
  for (int i = 0; i < MAX_NEURONS; i++) {
    float diff = fabs(neurons[i].output - memory->vector[i]);
    consistency += (1.0f - diff);
  }

  // Factor in memory importance
  float weighted_consistency = consistency / total_elements;
  weighted_consistency *= (0.5f + 0.5f * memory->importance);

  return weighted_consistency;
}

// Compute consistency score based on historical states
float computeConsistencyScore(Neuron *current_neurons,
                              NetworkStateSnapshot *history, int current_step) {
  float consistency = 0.0f;
  int comparison_window = 5; // Look at last 5 steps
  int start_step = fmax(0, current_step - comparison_window);
  int steps_compared = 0;

  // Compare with recent historical states
  for (int step = start_step; step < current_step; step++) {
    float step_similarity =
        computeStateSimilarity(current_neurons, &history[step]);

    // Weight recent steps more heavily
    float time_weight = 1.0f - (float)(current_step - step) / comparison_window;
    consistency += step_similarity * time_weight;
    steps_compared++;
  }

  return steps_compared > 0 ? consistency / steps_compared : 1.0f;
}

// Helper function to retrieve most relevant memory
MemoryEntry *retrieveRelevantMemory(MemorySystem *memorySystem,
                                    Neuron *current_neurons) {
  if (memorySystem->size == 0) {
    return NULL;
  }

  MemoryEntry *best_match = NULL;
  float best_similarity = -1.0f;

  // Compare current neuron states with memory entries
  for (size_t i = 0; i < memorySystem->size; i++) {
    int idx = (memorySystem->head - 1 - i + memorySystem->capacity) %
              memorySystem->capacity;
    MemoryEntry *memory = &memorySystem->entries[idx];

    float similarity = 0.0f;
    for (int j = 0; j < MAX_NEURONS; j++) {
      float diff = fabs(current_neurons[j].output - memory->vector[j]);
      similarity += (1.0f - diff);
    }
    similarity /= MAX_NEURONS;

    // Weight by memory importance
    similarity *= memory->importance;

    if (similarity > best_similarity) {
      best_similarity = similarity;
      best_match = memory;
    }
  }

  return best_match;
}

ReflectionHistory *initializeReflectionSystem() {
  ReflectionHistory *history =
      (ReflectionHistory *)malloc(sizeof(ReflectionHistory));

  if (history == NULL) {
    fprintf(stderr, "Failed to allocate memory for ReflectionHistory\n");
    return NULL;
  }

  memset(history, 0, sizeof(ReflectionHistory));
  history->confidence_threshold = 0.7f;
  history->coherence_threshold = 0.65f;
  history->consistency_threshold = 0.75f;
  history->history_index = 0;

  // Initialize historical metrics with neutral values to prevent initial bias
  for (int i = 0; i < 100; i++) {
    history->historical_coherence[i] = 0.75f;
    history->historical_confidence[i] = 0.75f;
    history->historical_consistency[i] = 0.75f;
  }

  return history;
}

// Analyze response coherence by comparing with previous states and memories
float analyzeResponseCoherence(Neuron *neurons, MemorySystem *memorySystem,
                               NetworkStateSnapshot *history,
                               int current_step) {
  float coherence_score = 1.0f;
  int activation_count = 0;

  // Check internal consistency of current neural activations with better
  // sampling
  for (int i = 0; i < MAX_NEURONS - 1;
       i += 3) { // Sample fewer pairs for efficiency
    for (int j = i + 1; j < fmin(i + 10, MAX_NEURONS); j++) {
      if (neurons[i].layer_id == neurons[j].layer_id) {
        float activation_diff = fabs(neurons[i].output - neurons[j].output);
        // More gradual penalty for differences
        if (activation_diff > 0.5f) {
          coherence_score *= (1.0f - 0.05f * activation_diff);
        }
        activation_count++;
      }
    }
  }

  // Normalize if we compared at least some neurons
  if (activation_count > 0) {
    // Ensure coherence doesn't drop too rapidly
    coherence_score = fmax(coherence_score, 0.4f);
  }

  // Compare with recent history, but limit historical influence
  if (current_step > 0) {
    float historical_similarity =
        computeStateSimilarity(neurons, &history[current_step - 1]);
    // More balanced blending with history
    coherence_score = (0.7f * coherence_score + 0.3f * historical_similarity);
  }

  // Check consistency with relevant memories but cap their influence
  MemoryEntry *relevant_memory = retrieveMemory(memorySystem);
  if (relevant_memory != NULL) {
    float memory_consistency =
        computeMemoryConsistency(neurons, relevant_memory);
    // Limit memory influence to prevent cascading errors
    coherence_score =
        (0.8f * coherence_score + 0.2f * fmax(0.5f, memory_consistency));
  }

  // Ensure coherence stays within reasonable bounds
  return fmax(0.3f, fmin(coherence_score, 1.0f));
}

// Detect potential confabulation by analyzing response patterns
bool detectConfabulation(Neuron *neurons, ReflectionHistory *history,
                         float current_coherence) {
  // Check for activation anomalies with better thresholds
  int high_activation_count = 0;
  int low_activation_count = 0;

  for (int i = 0; i < MAX_NEURONS; i++) {
    if (neurons[i].output > 0.95f) {
      high_activation_count++;
    }
    if (neurons[i].output < 0.05f) {
      low_activation_count++;
    }
  }

  // Calculate activation ratio to detect uniform distributions
  float activation_ratio = 0.0f;
  if ((high_activation_count + low_activation_count) > 0) {
    activation_ratio = (float)high_activation_count /
                       (high_activation_count + low_activation_count);
  }

  // Compare with historical coherence using a sliding window
  float recent_historical_coherence = 0.0f;
  int valid_history = 0;
  for (int i = 0; i < fmin(20, 100);
       i++) { // Look at more recent history (last 20)
    int idx = (history->history_index - i + 100) % 100;
    if (history->historical_coherence[idx] > 0) {
      recent_historical_coherence += history->historical_coherence[idx];
      valid_history++;
    }
  }

  if (valid_history > 0) {
    recent_historical_coherence /= valid_history;
  } else {
    recent_historical_coherence = 0.7f; // Default if no history
  }

  // Detect confabulation with more nuanced criteria
  bool suspicious_pattern =
      (high_activation_count > MAX_NEURONS * 0.4f) || // Increased threshold
      (activation_ratio > 0.9f ||
       activation_ratio < 0.1f) || // Check for skewed distributions
      (current_coherence <
       recent_historical_coherence * 0.7f) || // Less sensitive drop detection
      (current_coherence <
       history->coherence_threshold * 0.9f); // More forgiving threshold

  return suspicious_pattern;
}

// Helper function to regenerate response when confabulation is detected
void regenerateResponse(Neuron *neurons, MemorySystem *memorySystem,
                        ReflectionMetrics metrics, float *weights,
                        int *connections, ReflectionParameters *params) {
  // Save original parameters
  float original_noise_scale = params->input_noise_scale;
  float original_learning_rate = params->learning_rate;

  // Temporary parameter adjustments with upper bounds
  params->input_noise_scale = fmin(original_noise_scale * 1.5f, 0.4f);
  params->learning_rate = fmin(original_learning_rate * 1.2f, 0.05f);

  // Gently modify neuron states instead of aggressive scaling
  for (int i = 0; i < MAX_NEURONS; i++) {
    // Apply noise to break out of potential attractor states
    float noise_factor = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;

    // Dampen neuron states with bounds protection
    neurons[i].state =
        fmax(0.1f, fmin(neurons[i].state * 0.8f + noise_factor, 0.9f));

    // Adjust outputs more conservatively
    neurons[i].output =
        fmax(0.1f, fmin(neurons[i].output * 0.9f + noise_factor * 0.5f, 0.9f));
  }

  // Process neurons with gentler parameters to avoid instability
  processNeurons(neurons, MAX_NEURONS, weights, connections, MAX_CONNECTIONS,
                 1.1f); // Lower gain

  // Restore original parameters
  params->input_noise_scale = original_noise_scale;
  params->learning_rate = original_learning_rate;

  printf("Response regenerated with stabilized parameters\n");
}

ReflectionMetrics performSelfReflection(Neuron *neurons,
                                        MemorySystem *memorySystem,
                                        NetworkStateSnapshot *history,
                                        ReflectionHistory *reflection_history,
                                        int current_step) {
  ReflectionMetrics metrics = {0};

  // Analyze response coherence
  metrics.coherence_score =
      analyzeResponseCoherence(neurons, memorySystem, history, current_step);

  // Calculate confidence with better normalization
  float confidence = 0.0f;
  int valid_neurons = 0;

  for (int i = 0; i < MAX_NEURONS; i++) {
    // Only consider neurons with meaningful activation
    if (neurons[i].state > 0.1f || neurons[i].output > 0.1f) {
      confidence += (1.0f - fabs(neurons[i].output - neurons[i].state));
      valid_neurons++;
    }
  }

  // Normalize confidence score properly
  metrics.confidence_score =
      valid_neurons > 0 ? (confidence / valid_neurons) : 0.5f;

  // Assess novelty with bounds protection
  metrics.novelty_score =
      fmax(0.1f, fmin(computeNovelty(neurons, *history, current_step),
                      0.9f)); // Prevent extreme novelty values

  // Check consistency with previous responses
  metrics.consistency_score = 1.0f;
  if (current_step > 0) {
    metrics.consistency_score =
        fmax(0.3f, computeConsistencyScore(neurons, history, current_step));
  }

  // Detect potential confabulation with improved detection
  metrics.potentially_confabulated =
      detectConfabulation(neurons, reflection_history, metrics.coherence_score);

  // Generate reasoning about the reflection
  if (metrics.potentially_confabulated) {
    snprintf(metrics.reasoning, sizeof(metrics.reasoning),
             "Warning: Response shows signs of instability (coherence: %.2f, "
             "confidence: %.2f). "
             "Unusual activation patterns detected. Applying stabilization "
             "measures.",
             metrics.coherence_score, metrics.confidence_score);
  } else {
    snprintf(metrics.reasoning, sizeof(metrics.reasoning),
             "Response appears stable (coherence: %.2f, confidence: %.2f). "
             "Normal activation patterns observed.",
             metrics.coherence_score, metrics.confidence_score);
  }

  // Update reflection history with exponential smoothing
  int idx = reflection_history->history_index;
  reflection_history->historical_coherence[idx] =
      0.8f * metrics.coherence_score +
      0.2f *
          (idx > 0
               ? reflection_history->historical_coherence[(idx - 1 + 100) % 100]
               : metrics.coherence_score);

  reflection_history->historical_confidence[idx] =
      0.8f * metrics.confidence_score +
      0.2f * (idx > 0 ? reflection_history
                            ->historical_confidence[(idx - 1 + 100) % 100]
                      : metrics.confidence_score);

  reflection_history->historical_consistency[idx] =
      0.8f * metrics.consistency_score +
      0.2f * (idx > 0 ? reflection_history
                            ->historical_consistency[(idx - 1 + 100) % 100]
                      : metrics.consistency_score);

  reflection_history->history_index = (idx + 1) % 100;

  return metrics;
}

void integrateReflectionSystem(Neuron *neurons, MemorySystem *memorySystem,
                               NetworkStateSnapshot *history, int step,
                               float *weights, int *connections,
                               ReflectionParameters *params) {
  static ReflectionHistory *reflection_history = NULL;
  if (reflection_history == NULL) {
    reflection_history = initializeReflectionSystem();
  }

  ReflectionMetrics metrics = performSelfReflection(
      neurons, memorySystem, history, reflection_history, step);

  // Periodic reporting with reduced frequency to allow stabilization
  if (step % 20 == 0) {
    printf("\nSelf-Reflection Metrics (Step %d):\n", step);
    printf("Coherence Score: %.3f\n", metrics.coherence_score);
    printf("Confidence Score: %.3f\n", metrics.confidence_score);
    printf("Novelty Score: %.3f\n", metrics.novelty_score);
    printf("Consistency Score: %.3f\n", metrics.consistency_score);
    printf("Reasoning: %s\n", metrics.reasoning);
  }

  // Apply corrective measures only when needed
  if (metrics.potentially_confabulated) {
    printf("\nStabilization measures applied at step %d\n", step);
    regenerateResponse(neurons, memorySystem, metrics, weights, connections,
                       params);
  }

  // Parameter adaptation with homeostatic constraints
  // Only make small adjustments with bounds protection
  if (metrics.coherence_score < reflection_history->coherence_threshold) {
    params->learning_rate = fmax(0.001f, params->learning_rate * 0.9f);
  } else {
    // Gradually restore learning rate if things are stable
    params->learning_rate = fmin(0.02f, params->learning_rate * 1.05f);
  }

  // Adjust plasticity while maintaining within reasonable bounds
  float target_plasticity = 0.8f + 0.2f * metrics.confidence_score;
  params->plasticity = params->plasticity * 0.9f + target_plasticity * 0.1f;
  params->plasticity = fmax(0.5f, fmin(params->plasticity, 0.95f));

  // Adjust noise tolerance based on novelty but with stricter bounds
  float target_noise_tolerance =
      fmax(0.1f, 0.3f - 0.2f * metrics.novelty_score);
  params->noise_tolerance =
      params->noise_tolerance * 0.9f + target_noise_tolerance * 0.1f;
  params->noise_tolerance = fmax(0.05f, fmin(params->noise_tolerance, 0.4f));

  // Periodically reset parameters to avoid drift
  if (step % 500 == 0) {
    params->input_noise_scale =
        fmax(0.05f, fmin(params->input_noise_scale, 0.2f));
    params->weight_noise_scale =
        fmax(0.01f, fmin(params->weight_noise_scale, 0.1f));
  }
}

ReflectionParameters *initializeReflectionParameters() {
  ReflectionParameters *params =
      (ReflectionParameters *)malloc(sizeof(ReflectionParameters));

  if (params == NULL) {
    fprintf(stderr, "Failed to allocate memory for ReflectionParameters\n");
    return NULL;
  }

  // Initialize with more conservative default values
  params->current_adaptation_rate = 0.005f; // More conservative adaptation rate
  params->input_noise_scale = 0.08f;        // Less aggressive input noise
  params->weight_noise_scale = 0.03f;       // Smaller weight perturbations
  params->plasticity = 0.75f;               // More moderate initial plasticity
  params->noise_tolerance = 0.15f;          // Slightly reduced noise tolerance
  params->learning_rate = 0.008f;           // More conservative learning rate

  return params;
}

// Initialize the self-identity system
SelfIdentitySystem *initializeSelfIdentity(int num_values, int num_beliefs,
                                           int num_markers, int history_size,
                                           int pattern_size) {
  SelfIdentitySystem *system =
      (SelfIdentitySystem *)malloc(sizeof(SelfIdentitySystem));

  system->num_core_values = num_values;
  system->num_beliefs = num_beliefs;
  system->num_markers = num_markers;
  system->history_size = history_size;
  system->pattern_size = pattern_size;

  // Allocate memory for identity components
  system->core_values = (float *)calloc(num_values, sizeof(float));
  system->belief_system = (float *)calloc(num_beliefs, sizeof(float));
  system->identity_markers = (float *)calloc(num_markers, sizeof(float));
  system->experience_history = (float *)calloc(history_size, sizeof(float));
  system->behavioral_patterns = (float *)calloc(pattern_size, sizeof(float));

  // Initialize temporal coherence tracking
  system->coherence_window = 100; // Track last 100 states
  system->temporal_coherence =
      (float *)calloc(system->coherence_window, sizeof(float));

  // Set initial parameters
  system->consistency_score = 1.0f;
  system->adaptation_rate = 0.01f;
  system->confidence_level = 0.5f;

  // Initialize verification system
  system->verification.threshold = 0.8f;
  system->verification.state_size = num_values + num_beliefs + num_markers;
  system->verification.reference_state =
      (float *)calloc(system->verification.state_size, sizeof(float));

  return system;
}

// Extract behavioral patterns from neural network state
float *extractBehavioralPatterns(Neuron *neurons, int num_neurons) {
  float *patterns = (float *)calloc(PATTERN_SIZE, sizeof(float));

  // Calculate activation patterns
  for (int i = 0; i < PATTERN_SIZE; i++) {
    float pattern_sum = 0.0f;
    int neurons_per_pattern = num_neurons / PATTERN_SIZE;

    for (int j = 0; j < neurons_per_pattern; j++) {
      int neuron_idx = i * neurons_per_pattern + j;
      if (neuron_idx < num_neurons) {
        pattern_sum += neurons[neuron_idx].output;
      }
    }
    patterns[i] = pattern_sum / neurons_per_pattern;
  }

  // Normalize patterns
  float max_pattern = 0.0f;
  for (int i = 0; i < PATTERN_SIZE; i++) {
    if (patterns[i] > max_pattern)
      max_pattern = patterns[i];
  }
  if (max_pattern > 0.0f) {
    for (int i = 0; i < PATTERN_SIZE; i++) {
      patterns[i] /= max_pattern;
    }
  }

  return patterns;
}

// Compute consistency between current and stored patterns
float computePatternConsistency(float *stored_patterns, float *current_patterns,
                                int pattern_size) {
  float consistency = 0.0f;
  float sum_squared_diff = 0.0f;

  for (int i = 0; i < pattern_size; i++) {
    float diff = stored_patterns[i] - current_patterns[i];
    sum_squared_diff += diff * diff;
  }

  consistency = 1.0f - sqrt(sum_squared_diff / pattern_size);
  return fmax(0.0f, consistency); // Ensure non-negative
}

float computeValueConsistency(float *core_values, int num_values) {
  if (num_values == 0)
    return 0.0f;

  float consistency = 0.0f;
  float value_sum = 0.0f;
  float value_squared_sum = 0.0f;
  bool has_valid_values = false;

  // First pass - check if we have any non-zero values
  for (int i = 0; i < num_values; i++) {
    if (core_values[i] != 0.0f) {
      has_valid_values = true;
      break;
    }
  }

  // If all values are 0, initialize with small random values
  if (!has_valid_values) {
    for (int i = 0; i < num_values; i++) {
      // Initialize with small random values between 0.1 and 0.3
      core_values[i] = 0.1f + (float)rand() / RAND_MAX * 0.2f;
    }
  }

  // Calculate consistency
  for (int i = 0; i < num_values; i++) {
    if (isnan(core_values[i])) {
      core_values[i] = 0.1f; // Handle NaN with small non-zero value
    }
    value_sum += core_values[i];
    value_squared_sum += core_values[i] * core_values[i];
  }

  float mean = value_sum / num_values;
  float variance = (value_squared_sum / num_values) - (mean * mean);

  // Use a different formula that better reflects stability
  // Higher values when core values are stable and significant
  consistency = mean / (1.0f + sqrt(variance));

  // Ensure result is meaningful and between 0.1 and 1.0
  return fmin(1.0f, fmax(0.1f, consistency));
}

bool haveCommonPrimeFactors(int a, int b) {
  while (a % 2 == 0 && b % 2 == 0) {
    return true;
  }
  for (int i = 3; i <= sqrt((a > b) ? a : b); i += 2) {
    if (a % i == 0 && b % i == 0) {
      return true;
    }
  }
  return false;
}

// Function to count the number of differing bits between two numbers
int countDifferentBits(int a, int b) {
  int xor_result = a ^ b;
  int count = 0;
  while (xor_result) {
    count += xor_result & 1;
    xor_result >>= 1;
  }
  return count;
}

// Function to check bitwise similarity (at most `threshold` differing bits)
bool areBitsSimilar(int a, int b, int threshold) {
  return countDifferentBits(a, b) <= threshold;
}

bool areNumericallyClose(int a, int b, int threshold) {
  return (a > b) ? (a - b <= threshold) : (b - a <= threshold);
}

bool areBeliefsProbablyRelated(int belief1_idx, int belief2_idx) {
  return (belief1_idx / 10 == belief2_idx / 10) || // Same decade cluster
         (belief1_idx % 7 == belief2_idx % 7) || // Additional clustering factor
         haveCommonPrimeFactors(belief1_idx,
                                belief2_idx) || // Prime factor similarity
         areBitsSimilar(belief1_idx, belief2_idx, 3) || // Bitwise similarity
         areNumericallyClose(belief1_idx, belief2_idx,
                             5); // Numeric proximity threshold
}

float computeBeliefConsistency(float *beliefs, int num_beliefs) {
  if (num_beliefs == 0)
    return 0.0f;

  float consistency = 0.0f;
  float belief_coherence = 0.0f;
  int total_comparisons = 0;

  for (int i = 0; i < num_beliefs; i++) {
    if (isnan(beliefs[i])) {
      beliefs[i] = 0.0f; // Handle NaN values
    }

    float local_coherence = 0.0f;
    int related_beliefs = 0;

    for (int j = 0; j < num_beliefs; j++) {
      if (i != j && areBeliefsProbablyRelated(i, j)) {
        if (!isnan(beliefs[j])) {
          local_coherence += 1.0f - fabs(beliefs[i] - beliefs[j]);
          related_beliefs++;
        }
      }
    }

    if (related_beliefs > 0) {
      belief_coherence += local_coherence / related_beliefs;
      total_comparisons++;
    }
  }

  // Prevent division by zero
  consistency =
      (total_comparisons > 0) ? (belief_coherence / total_comparisons) : 0.0f;

  return fmin(1.0f,
              fmax(0.0f, consistency)); // Ensure result is between 0 and 1
}

float computeMarkerStability(float *markers, int num_markers) {
  if (num_markers == 0)
    return 0.0f;

  float stability = 0.0f;
  float marker_strength = 0.0f;
  int valid_markers = 0;
  bool has_valid_markers = false;

  // First pass - check if we have any non-zero values
  for (int i = 0; i < num_markers; i++) {
    if (markers[i] != 0.0f) {
      has_valid_markers = true;
      break;
    }
  }

  // If all markers are 0, initialize with small random values
  if (!has_valid_markers) {
    for (int i = 0; i < num_markers; i++) {
      // Initialize with small random values between 0.1 and 0.3
      markers[i] = 0.1f + (float)rand() / RAND_MAX * 0.2f;
    }
  }

  // Calculate stability with enhanced weighting
  float max_strength = 0.0f;
  for (int i = 0; i < num_markers; i++) {
    if (!isnan(markers[i])) {
      float strength = fabs(markers[i]);
      marker_strength += strength * strength; // Quadratic weighting
      max_strength = fmax(max_strength, strength);
      valid_markers++;
    } else {
      markers[i] = 0.1f; // Handle NaN with small non-zero value
    }
  }

  if (valid_markers > 0 && marker_strength >= 0.0f) {
    float avg_strength = sqrt(marker_strength / valid_markers);
    stability = 0.7f * avg_strength + 0.3f * max_strength;
  }

  // Ensure result is meaningful and between 0.1 and 1.0
  return fmin(1.0f, fmax(0.1f, stability));
}

void initializeIdentityComponents(SelfIdentitySystem *system) {
  if (!system)
    return;

  // Initialize core values with small random values
  for (int i = 0; i < system->num_core_values; i++) {
    system->core_values[i] = 0.1f + (float)rand() / RAND_MAX * 0.2f;
  }

  // Initialize identity markers with small random values
  for (int i = 0; i < system->num_markers; i++) {
    system->identity_markers[i] = 0.1f + (float)rand() / RAND_MAX * 0.2f;
  }

  // Initialize belief system with small random values
  for (int i = 0; i < system->num_beliefs; i++) {
    system->belief_system[i] = 0.1f + (float)rand() / RAND_MAX * 0.2f;
  }
}

// Get current complete identity state
float *getCurrentIdentityState(SelfIdentitySystem *system) {
  int total_size = system->verification.state_size;
  float *current_state = (float *)malloc(total_size * sizeof(float));
  int offset = 0;

  // Copy core values
  memcpy(current_state + offset, system->core_values,
         system->num_core_values * sizeof(float));
  offset += system->num_core_values;

  // Copy beliefs
  memcpy(current_state + offset, system->belief_system,
         system->num_beliefs * sizeof(float));
  offset += system->num_beliefs;

  // Copy identity markers
  memcpy(current_state + offset, system->identity_markers,
         system->num_markers * sizeof(float));

  return current_state;
}

// Compute consistency between two identity states
float computeStateConsistency(float *state1, float *state2, int state_size) {
  float consistency = 0.0f;
  float sum_squared_diff = 0.0f;
  float sum_magnitudes = 0.0f;

  for (int i = 0; i < state_size; i++) {
    float diff = state1[i] - state2[i];
    sum_squared_diff += diff * diff;
    sum_magnitudes += fabs(state1[i]) + fabs(state2[i]);
  }

  if (sum_magnitudes > 0.0f) {
    consistency = 1.0f - (sqrt(sum_squared_diff) / (sum_magnitudes / 2.0f));
  }

  return fmax(0.0f, consistency);
}

void updateCoreValues(SelfIdentitySystem *system, float *current_patterns,
                      float pattern_consistency) {
  float adaptation_factor =
      system->adaptation_rate * (1.0f - pattern_consistency);

  for (uint32_t i = 0; i < static_cast<uint32_t>(system->num_core_values);
       i++) {
    float pattern_influence = 0.0f;
    float weight_sum = 0.0f;
    uint32_t patterns_per_value =
        system->pattern_size / system->num_core_values;

    for (uint32_t j = 0; j < patterns_per_value; j++) {
      uint32_t pattern_idx = i * patterns_per_value + j;

      if (static_cast<int>(pattern_idx) < system->pattern_size) {
        // Use a weight for each pattern, e.g., exponential or logarithmic
        float weight = std::exp(j / static_cast<float>(patterns_per_value));
        float weighted_pattern = current_patterns[pattern_idx] * weight;

        pattern_influence += weighted_pattern;
        weight_sum += weight;
      }
    }

    // Normalize by the sum of weights
    if (weight_sum > 0) {
      pattern_influence /= weight_sum;
    }

    // Introduce non-linearity in the adaptation factor
    float dynamic_adaptation_factor =
        adaptation_factor * (1.0f + 0.5f * std::sin(i));

    // Update core value with stability consideration
    system->core_values[i] =
        (1.0f - dynamic_adaptation_factor) * system->core_values[i] +
        dynamic_adaptation_factor * pattern_influence;

    system->core_values[i] = clampValue(system->core_values[i]);
  }
}

void updateReferenceStates(SelfIdentitySystem *system, float *current_state) {
  for (uint32_t i = 0;
       i < static_cast<uint32_t>(system->verification.state_size); i++) {
    system->verification.reference_state[i] =
        (1.0f - system->adaptation_rate) *
            system->verification.reference_state[i] +
        system->adaptation_rate * current_state[i];

    // Clamp the value to a reasonable range
    system->verification.reference_state[i] =
        clampValue(system->verification.reference_state[i]);
  }
}

// Compute experience value using weighted encoding
float computeExperienceValue(float *experience_vector) {
  float weighted_sum = 0.0f;
  float weight_total = 0.0f;

  for (uint32_t i = 0; i < EXPERIENCE_VECTOR_SIZE; i++) {
    float weight = expf(-(float)(EXPERIENCE_VECTOR_SIZE - i) /
                        EXPERIENCE_VECTOR_SIZE); // Exponential decay
    weighted_sum += experience_vector[i] * weight;
    weight_total += weight;
  }

  float encoded_value =
      weighted_sum / weight_total; // Normalize with sum of weights
  return sigmoid(
      encoded_value); // Apply sigmoid to keep result in a controlled range
}

// Add new experience to history
void addExperience(SelfIdentitySystem *system, float *experience_vector) {
  // Shift history window
  memmove(system->experience_history + 1, system->experience_history,
          (system->history_size - 1) * sizeof(float));

  // Add new experience
  system->experience_history[0] = computeExperienceValue(experience_vector);
}

// Helper function to compute memory influence on beliefs
float computeMemoryInfluence(MemorySystem *memory_system, uint32_t belief_idx) {
  float influence = 0.0f;
  uint32_t recent_memories = 10; // Consider last 10 memories

  for (uint32_t i = 0; i < recent_memories && i < memory_system->size; i++) {
    uint32_t memory_idx =
        (memory_system->head - 1 - i + memory_system->capacity) %
        memory_system->capacity;
    influence += memory_system->entries[memory_idx]
                     .vector[belief_idx % MEMORY_VECTOR_SIZE];
  }

  return influence / recent_memories;
}

// Helper function to compute experience influence on beliefs
float computeExperienceInfluence(SelfIdentitySystem *system,
                                 uint32_t belief_idx) {
  float influence = 0.0f;
  uint32_t recent_experiences = 5; // Consider last 5 experiences

  for (uint32_t i = 0; i < static_cast<uint32_t>(recent_experiences) &&
                       i < static_cast<uint32_t>(system->history_size);
       i++) {
    influence += system->experience_history[i];
  }

  return influence / recent_experiences;
}

void updateBeliefs(SelfIdentitySystem *system, MemorySystem *memory_system) {
  for (uint32_t i = 0; i < static_cast<uint32_t>(system->num_beliefs); i++) {
    float memory_influence = computeMemoryInfluence(memory_system, i);
    float experience_influence = computeExperienceInfluence(system, i);

    // Update belief with weighted combination of memory and experience
    system->belief_system[i] =
        (1.0f - system->adaptation_rate) * system->belief_system[i] +
        system->adaptation_rate *
            (0.7f * memory_influence + 0.3f * experience_influence);

    // Clamp the value to a reasonable range
    system->belief_system[i] = clampValue(system->belief_system[i]);
  }
}

bool areValueAndMarkerRelated(uint32_t value_idx, uint32_t marker_idx) {
  return (value_idx % 5 == marker_idx % 5) ||
         haveCommonPrimeFactors(value_idx,
                                marker_idx) ||     // Common prime factors
         areBitsSimilar(value_idx, marker_idx, 3); // Bitwise similarity
}

bool areBeliefAndMarkerRelated(uint32_t belief_idx, uint32_t marker_idx) {
  return (belief_idx % 7 == marker_idx % 7) ||
         haveCommonPrimeFactors(belief_idx,
                                marker_idx) ||      // Common prime factors
         areBitsSimilar(belief_idx, marker_idx, 3); // Bitwise similarity
}

// Helper function to compute core value influence on markers
float computeValueInfluence(SelfIdentitySystem *system, uint32_t marker_idx) {
  float influence = 0.0f;
  uint32_t related_values = 0;

  for (uint32_t i = 0; i < static_cast<uint32_t>(system->num_core_values);
       i++) {
    if (areValueAndMarkerRelated(i, marker_idx)) {
      influence += system->core_values[i];
      related_values++;
    }
  }

  return related_values > 0 ? influence / related_values : 0.0f;
}

// Helper function to compute belief influence on markers
float computeBeliefInfluence(SelfIdentitySystem *system, uint32_t marker_idx) {
  float influence = 0.0f;
  uint32_t related_beliefs = 0;

  for (uint32_t i = 0; i < static_cast<uint32_t>(system->num_beliefs); i++) {
    if (areBeliefAndMarkerRelated(i, marker_idx)) {
      influence += system->belief_system[i];
      related_beliefs++;
    }
  }

  return related_beliefs > 0 ? influence / related_beliefs : 0.0f;
}

// Update identity markers
void updateIdentityMarkers(SelfIdentitySystem *system) {
  for (uint32_t i = 0; i < static_cast<uint32_t>(system->num_markers); i++) {
    float value_influence = computeValueInfluence(system, i);
    float belief_influence = computeBeliefInfluence(system, i);

    // Update marker with weighted combination of influences
    system->identity_markers[i] =
        (1.0f - system->adaptation_rate) * system->identity_markers[i] +
        system->adaptation_rate *
            (0.6f * value_influence + 0.4f * belief_influence);
  }
}

float computeRecentCoherence(float *coherence_history, uint32_t window_size) {
  if (window_size == 0)
    return 0.0f;

  float sum = 0.0f;
  uint32_t valid_samples = 0;

  for (uint32_t i = 0; i < window_size; i++) {
    if (!isnan(coherence_history[i])) {
      sum += coherence_history[i];
      valid_samples++;
    }
  }

  // Prevent division by zero
  return (valid_samples > 0) ? (sum / valid_samples) : 0.0f;
}

void updateConfidenceLevel(SelfIdentitySystem *system) {
  if (!system)
    return;

  float recent_coherence =
      computeRecentCoherence(system->temporal_coherence, 10);
  float value_consistency =
      computeValueConsistency(system->core_values, system->num_core_values);
  float belief_consistency =
      computeBeliefConsistency(system->belief_system, system->num_beliefs);

  // Calculate weighted average with validity checks
  float confidence = 0.0f;
  float total_weight = 0.0f;

  if (!isnan(recent_coherence)) {
    confidence += 0.4f * recent_coherence;
    total_weight += 0.4f;
  }
  if (!isnan(value_consistency)) {
    confidence += 0.3f * value_consistency;
    total_weight += 0.3f;
  }
  if (!isnan(belief_consistency)) {
    confidence += 0.3f * belief_consistency;
    total_weight += 0.3f;
  }

  // Normalize if we have any valid components
  system->confidence_level =
      (total_weight > 0.0f) ? (confidence / total_weight) : 0.0f;
}

// Shift coherence window to make room for new coherence value
void shiftCoherenceWindow(SelfIdentitySystem *system) {
  memmove(system->temporal_coherence, system->temporal_coherence + 1,
          (system->coherence_window - 1) * sizeof(float));
}

// Compress current experience into a summary vector
float *compressExperience(float *current_input, Neuron *neurons,
                          uint32_t num_neurons) {
  float *compressed = (float *)calloc(EXPERIENCE_VECTOR_SIZE, sizeof(float));

  // Combine input and neuron states into experience vector
  for (uint32_t i = 0; i < EXPERIENCE_VECTOR_SIZE; i++) {
    float input_contribution = 0.0f;
    float neuron_contribution = 0.0f;

    // Sample input values
    uint32_t inputs_per_experience = num_neurons / EXPERIENCE_VECTOR_SIZE;
    for (uint32_t j = 0; j < inputs_per_experience; j++) {
      uint32_t idx = i * inputs_per_experience + j;
      if (idx < num_neurons) {
        input_contribution += current_input[idx];
        neuron_contribution += neurons[idx].output;
      }
    }

    compressed[i] = (input_contribution + neuron_contribution) /
                    (2.0f * inputs_per_experience);
  }

  return compressed;
}

// Calculate identity coherence score
float computeIdentityCoherence(SelfIdentitySystem *system) {
  float coherence = 0.0f;
  float value_consistency =
      computeValueConsistency(system->core_values, system->num_core_values);
  float belief_consistency =
      computeBeliefConsistency(system->belief_system, system->num_beliefs);
  float marker_stability =
      computeMarkerStability(system->identity_markers, system->num_markers);

  coherence = (value_consistency * 0.4f + belief_consistency * 0.3f +
               marker_stability * 0.3f);
  return coherence;
}

// Update identity based on new experiences and network state
void updateIdentity(SelfIdentitySystem *system, Neuron *neurons,
                    uint32_t num_neurons, MemorySystem *memory_system,
                    float *current_input) {

  // Extract current behavioral patterns
  float *current_patterns = extractBehavioralPatterns(neurons, num_neurons);

  // Calculate pattern consistency
  float pattern_consistency = computePatternConsistency(
      system->behavioral_patterns, current_patterns, system->pattern_size);

  // Update behavioral patterns with temporal smoothing
  for (uint32_t i = 0; i < static_cast<uint32_t>(system->pattern_size); i++) {
    system->behavioral_patterns[i] =
        (1 - system->adaptation_rate) * system->behavioral_patterns[i] +
        system->adaptation_rate * current_patterns[i];
  }
  // Update core values based on consistent behaviors
  updateCoreValues(system, current_patterns, pattern_consistency);

  updateReferenceStates(system, current_input);

  // Integrate new experiences
  float *experience_vector =
      compressExperience(current_input, neurons, num_neurons);
  addExperience(system, experience_vector);

  // Update belief system based on experiences and core values
  updateBeliefs(system, memory_system);

  // Update identity markers
  updateIdentityMarkers(system);

  // Track temporal coherence
  shiftCoherenceWindow(system);
  system->temporal_coherence[system->coherence_window - 1] =
      computeIdentityCoherence(system);

  // Update confidence based on consistency
  updateConfidenceLevel(system);

  free(current_patterns);
  free(experience_vector);
}

// Verify identity consistency
bool verifyIdentity(SelfIdentitySystem *system) {
  float *current_state = getCurrentIdentityState(system);
  float consistency = computeStateConsistency(
      current_state, system->verification.reference_state,
      system->verification.state_size);

  bool verified = consistency >= system->verification.threshold;

  if (verified) {
    // Update reference state with slight adaptation
    for (int i = 0; i < system->verification.state_size; i++) {
      system->verification.reference_state[i] =
          (1 - system->adaptation_rate) *
              system->verification.reference_state[i] +
          system->adaptation_rate * current_state[i];
    }
  }

  free(current_state);
  return verified;
}

// Generate identity reflection
char *generateIdentityReflection(SelfIdentitySystem *system) {
  char *reflection = (char *)malloc(4096 * sizeof(char));

  sprintf(reflection,
          "Identity Reflection:\n"
          "Consistency Score: %.2f\n"
          "Confidence Level: %.2f\n"
          "Core Value Stability: %.2f\n"
          "Belief System Coherence: %.2f\n"
          "Identity Marker Prominence: %.2f\n"
          "Temporal Coherence (Last 10 states): %.2f\n",
          system->consistency_score, system->confidence_level,
          computeValueConsistency(system->core_values, system->num_core_values),
          computeBeliefConsistency(system->belief_system, system->num_beliefs),
          computeMarkerStability(system->identity_markers, system->num_markers),
          computeRecentCoherence(system->temporal_coherence, 10));

  return reflection;
}

KnowledgeFilter *initializeKnowledgeFilter(int initial_capacity) {
  KnowledgeFilter *filter = (KnowledgeFilter *)malloc(sizeof(KnowledgeFilter));
  filter->categories =
      (KnowledgeCategory *)malloc(initial_capacity * sizeof(KnowledgeCategory));
  filter->num_categories = 0;
  filter->capacity = initial_capacity;

  filter->problem_history =
      (ProblemInstance *)malloc(initial_capacity * sizeof(ProblemInstance));
  filter->num_problems = 0;
  filter->problem_capacity = initial_capacity;

  // Initialize similarity matrix
  filter->category_similarity_matrix =
      (float *)calloc(initial_capacity * initial_capacity, sizeof(float));

  // Initialize default categories
  const char *default_categories[] = {
      "Pattern Recognition", "Numerical Computation",
      "Sequence Learning",   "Classification",
      "Prediction",          "Optimization",
      "Error Correction",    "Memory Consolidation"};

  for (size_t i = 0; i < sizeof(default_categories) / sizeof(char *); i++) {
    KnowledgeCategory category = {.importance = 1.0f,
                                  .confidence = 0.5f,
                                  .usage_count = 0,
                                  .last_accessed = time(NULL)};
    strncpy(category.name, default_categories[i], 63);
    category.feature_vector =
        (float *)calloc(FEATURE_VECTOR_SIZE, sizeof(float));
    filter->categories[filter->num_categories++] = category;
  }

  return filter;
}

float computeCategorySimilarity(float *feature_vector1,
                                float *feature_vector2) {
  float dot_product = 0.0f;
  float norm1 = 0.0f;
  float norm2 = 0.0f;

  for (int i = 0; i < FEATURE_VECTOR_SIZE; i++) {
    dot_product += feature_vector1[i] * feature_vector2[i];
    norm1 += feature_vector1[i] * feature_vector1[i];
    norm2 += feature_vector2[i] * feature_vector2[i];
  }

  // Avoid NaN by checking for zero norm
  if (norm1 == 0.0f || norm2 == 0.0f) {
    return 0.0f; // No valid similarity if either vector is zero
  }

  return dot_product / (sqrtf(norm1) * sqrtf(norm2));
}

void updateCategorySimilarityMatrix(KnowledgeFilter *filter) {
  for (int i = 0; i < filter->num_categories; i++) {
    for (int j = i; j < filter->num_categories; j++) {
      float similarity =
          computeCategorySimilarity(filter->categories[i].feature_vector,
                                    filter->categories[j].feature_vector);
      filter->category_similarity_matrix[i * filter->capacity + j] = similarity;
      filter->category_similarity_matrix[j * filter->capacity + i] = similarity;
    }
  }
}

KnowledgeCategory *categorizeInput(KnowledgeFilter *filter, float *input_vector,
                                   float threshold) {
  KnowledgeCategory *best_match = NULL;
  float best_similarity = threshold;

  for (uint32_t i = 0; i < static_cast<uint32_t>(filter->num_categories); i++) {
    float similarity = computeCategorySimilarity(
        input_vector, filter->categories[i].feature_vector);

    if (similarity > best_similarity) {
      best_similarity = similarity;
      best_match = &filter->categories[i];
    }
  }

  if (best_match) {

    // Update usage statistics
    best_match->usage_count++;
    best_match->last_accessed = time(NULL);

    // Update confidence based on similarity score
    best_match->confidence =
        (best_match->confidence * 0.9f + best_similarity * 0.1f);

    // Adjust importance based on usage and confidence
    best_match->importance =
        (best_match->usage_count / (float)MAX_USAGE_COUNT) * 0.5f +
        best_match->confidence * 0.5f;
  } else {
    printf("\nNo category matched above threshold (%.2f)\n", threshold);
  }

  return best_match;
}

void recordProblemInstance(KnowledgeFilter *filter, const char *description,
                           float *feature_vector, float difficulty,
                           float success_rate, KnowledgeCategory *category) {
  if (filter->num_problems >= filter->problem_capacity) {
    filter->problem_capacity *= 2;
    filter->problem_history = (ProblemInstance *)realloc(
        filter->problem_history,
        filter->problem_capacity * sizeof(ProblemInstance));
  }

  ProblemInstance problem = {.difficulty = difficulty,
                             .success_rate = success_rate,
                             .category = category,
                             .timestamp = time(NULL)};

  strncpy(problem.description, description, 255);
  problem.feature_vector = (float *)malloc(FEATURE_VECTOR_SIZE * sizeof(float));
  memcpy(problem.feature_vector, feature_vector,
         FEATURE_VECTOR_SIZE * sizeof(float));

  filter->problem_history[filter->num_problems++] = problem;
}

CategoryStatistics analyzeCategory(KnowledgeFilter *filter,
                                   KnowledgeCategory *category) {
  CategoryStatistics stats = {0};
  stats.avg_success_rate = 0;
  stats.avg_difficulty = 0;
  stats.total_instances = 0;
  stats.last_encounter = 0;

  for (int i = 0; i < filter->num_problems; i++) {
    if (filter->problem_history[i].category == category) {
      stats.avg_success_rate += filter->problem_history[i].success_rate;
      stats.avg_difficulty += filter->problem_history[i].difficulty;
      stats.total_instances++;
      if (filter->problem_history[i].timestamp > stats.last_encounter) {
        stats.last_encounter = filter->problem_history[i].timestamp;
      }
    }
  }

  if (stats.total_instances > 0) {
    stats.avg_success_rate /= stats.total_instances;
    stats.avg_difficulty /= stats.total_instances;
  }

  return stats;
}

void analyzeFeatureVector(float *feature_vector, const char *label) {
  float sum = 0.0f;
  float min = 1.0f;
  float max = 0.0f;
  float nonzero = 0.0f;

  for (int i = 0; i < FEATURE_VECTOR_SIZE; i++) {
    sum += feature_vector[i];
    min = fmin(min, feature_vector[i]);
    max = fmax(max, feature_vector[i]);
    if (feature_vector[i] != 0.0f) {
      nonzero += 1.0f;
    }
  }

  printf("\nFeature Vector Analysis (%s):\n", label);
  printf("Average value: %.4f\n", sum / FEATURE_VECTOR_SIZE);
  printf("Range: %.4f to %.4f\n", min, max);
  printf("Non-zero elements: %.0f (%.1f%%)\n", nonzero,
         (nonzero / FEATURE_VECTOR_SIZE) * 100);
}

float *extractFeatureVector(Neuron *neurons, float *input_tensor) {
  float *feature_vector = (float *)malloc(FEATURE_VECTOR_SIZE * sizeof(float));

  // Initialize feature vector
  memset(feature_vector, 0, FEATURE_VECTOR_SIZE * sizeof(float));

  // Extract neuron activation patterns with averaging
  int neurons_per_feature = MAX_NEURONS / (FEATURE_VECTOR_SIZE / 2);
  for (int i = 0; i < FEATURE_VECTOR_SIZE / 2; i++) {
    float sum = 0.0f;
    int count = 0;
    for (int j = 0; j < neurons_per_feature; j++) {
      int idx = i * neurons_per_feature + j;
      if (idx < MAX_NEURONS) {
        sum += neurons[idx].output;
        count++;
      }
    }
    feature_vector[i] = count > 0 ? sum / count : 0.0f;
  }

  // Extract input patterns with averaging
  for (int i = 0; i < FEATURE_VECTOR_SIZE / 2; i++) {
    float sum = 0.0f;
    int count = 0;
    for (int j = 0; j < neurons_per_feature; j++) {
      int idx = i * neurons_per_feature + j;
      if (idx < MAX_NEURONS) {
        sum += input_tensor[idx];
        count++;
      }
    }
    feature_vector[i + FEATURE_VECTOR_SIZE / 2] =
        count > 0 ? sum / count : 0.0f;
  }

  // Add some noise to prevent zero vectors
  for (int i = 0; i < FEATURE_VECTOR_SIZE; i++) {
    feature_vector[i] += ((float)rand() / RAND_MAX) * 0.01f;
  }

  // Normalize feature vector
  float norm = 0.0f;
  for (int i = 0; i < FEATURE_VECTOR_SIZE; i++) {
    norm += feature_vector[i] * feature_vector[i];
  }
  norm = sqrt(norm);

  if (norm > 0) {
    for (int i = 0; i < FEATURE_VECTOR_SIZE; i++) {
      feature_vector[i] /= norm;
    }
  }

  // Analyze the generated feature vector
  analyzeFeatureVector(feature_vector, "Extracted Features");

  return feature_vector;
}

// Comparison function for memory sorting
int compareMemoryImportance(const void *a, const void *b) {
  const MemoryEntry *entry_a = (const MemoryEntry *)a;
  const MemoryEntry *entry_b = (const MemoryEntry *)b;

  if (entry_a->importance > entry_b->importance)
    return -1;
  if (entry_a->importance < entry_b->importance)
    return 1;
  return 0;
}

void addToMemoryLevel(MemoryCluster *level, MemoryEntry *entry) {
  if (level->size < level->capacity) {
    // Copy entry to the level
    memcpy(&level->entries[level->size], entry, sizeof(MemoryEntry));
    level->size++;

    // Sort entries by importance
    qsort(level->entries, level->size, sizeof(MemoryEntry),
          compareMemoryImportance);
  }
}

void strengthenMemory(MemorySystem *memory_system, int index) {
  if (static_cast<int>(index) >= static_cast<int>(memory_system->size)) {
    return;
  }

  // Increase importance of the memory
  memory_system->entries[index].importance *= 1.1f;

  // Cap importance at 1.0
  if (memory_system->entries[index].importance > 1.0f) {
    memory_system->entries[index].importance = 1.0f;
  }

  // Update access time
  memory_system->entries[index].timestamp = time(NULL);

  // Promote memory based on importance
  MemoryEntry *entry = &memory_system->entries[index];

  if (entry->importance > 0.9f) {
    // Move to long-term memory if space is available
    if (memory_system->hierarchy.long_term.size <
        memory_system->hierarchy.long_term.capacity) {
      addToMemoryLevel(&memory_system->hierarchy.long_term, entry);
    }
  } else if (entry->importance > 0.8f) {
    // Move to medium-term memory if space is available
    if (memory_system->hierarchy.medium_term.size <
        memory_system->hierarchy.medium_term.capacity) {
      addToMemoryLevel(&memory_system->hierarchy.medium_term, entry);
    }
  } else {
    // Default to short-term if not promoted
    if (memory_system->hierarchy.short_term.size <
        memory_system->hierarchy.short_term.capacity) {
      addToMemoryLevel(&memory_system->hierarchy.short_term, entry);
    }
  }
}

void integrateKnowledgeFilter(KnowledgeFilter *filter,
                              MemorySystem *memory_system, Neuron *neurons,
                              float *input_tensor) {
  // Extract feature vector from current network state
  float *current_features = extractFeatureVector(neurons, input_tensor);

  // Categorize current input
  KnowledgeCategory *category = categorizeInput(filter, current_features, 0.7f);

  if (category) {
    // Get category statistics including success rate
    CategoryStatistics stats = analyzeCategory(filter, category);

    // Update category confidence based on success rate
    category->confidence =
        category->confidence * 0.8f + stats.avg_success_rate * 0.2f;

    // Update importance based on multiple factors
    float usage_factor = (float)category->usage_count / MAX_USAGE_COUNT;
    float difficulty_factor = stats.avg_difficulty;
    float success_factor = stats.avg_success_rate;

    category->importance =
        usage_factor * 0.3f + difficulty_factor * 0.4f + success_factor * 0.3f;

    // Normalize importance to [0,1] range
    category->importance = fmax(0.0f, fmin(1.0f, category->importance));

    // Update feature vector with new information
    for (int i = 0; i < FEATURE_VECTOR_SIZE; i++) {
      category->feature_vector[i] =
          category->feature_vector[i] * 0.9f + current_features[i] * 0.1f;
    }
  }

  free(current_features);
}

void printCategoryInsights(KnowledgeFilter *filter) {
  printf("\nKnowledge Category Insights:\n");
  printf("%-20s %-12s %-12s %-12s %-12s\n", "Category", "Confidence",
         "Importance", "Usage Count", "Last Access");

  for (int i = 0; i < filter->num_categories; i++) {
    KnowledgeCategory *cat = &filter->categories[i];
    char time_str[26];
    ctime_r(&cat->last_accessed, time_str);
    time_str[24] = '\0'; // Remove newline

    printf("%-20s %-12.2f %-12.2f %-12u %s\n", cat->name, cat->confidence,
           cat->importance, cat->usage_count, time_str);
  }
}

void updateKnowledgeSystem(Neuron *neurons, float *input_tensor,
                           MemorySystem *memory_system, KnowledgeFilter *filter,
                           float current_performance) {

  // Integrate knowledge filter with current network state
  integrateKnowledgeFilter(filter, memory_system, neurons, input_tensor);

  // Extract feature vector for problem instance
  float *feature_vector = extractFeatureVector(neurons, input_tensor);

  // Categorize current input
  KnowledgeCategory *category = categorizeInput(filter, feature_vector, 0.7f);

  if (category) {
    // Record this as a problem instance
    char description[256];
    snprintf(description, sizeof(description),
             "Network state analysis at performance level %.2f",
             current_performance);

    // Calculate difficulty based on neuron activation patterns
    float difficulty = 0.0f;
    for (int i = 0; i < MAX_NEURONS; i++) {
      if (neurons[i].output > 0.8f) {
        difficulty += 0.1f;
      }
    }
    difficulty = fmin(1.0f, difficulty);

    // Record the problem instance
    recordProblemInstance(filter, description, feature_vector, difficulty,
                          current_performance, category);

    // Update category usage statistics
    category->usage_count++;
    category->last_accessed = time(NULL);

    // Update confidence based on performance
    category->confidence =
        category->confidence * 0.9f + current_performance * 0.1f;

    // Update importance based on multiple factors
    CategoryStatistics stats = analyzeCategory(filter, category);
    float usage_factor = (float)category->usage_count / MAX_USAGE_COUNT;
    float recency_factor =
        1.0f / (1.0f + (time(NULL) - stats.last_encounter) / 86400.0f);

    category->importance = usage_factor * 0.3f + stats.avg_difficulty * 0.3f +
                           recency_factor * 0.2f + category->confidence * 0.2f;
  }

  free(feature_vector);

  // Periodically update similarity matrix
  static time_t last_matrix_update = 0;
  if (time(NULL) - last_matrix_update > 3600) { // Update every hour
    updateCategorySimilarityMatrix(filter);
    last_matrix_update = time(NULL);
  }
}

void initializeKnowledgeMetrics(KnowledgeFilter *filter) {
  for (uint32_t i = 0; i < static_cast<uint32_t>(filter->num_categories); i++) {
    // Start with moderate values instead of extremes
    filter->categories[i].importance = 0.5f;
    filter->categories[i].confidence = 0.5f;
    filter->categories[i].usage_count = 0;
    filter->categories[i].last_accessed = time(NULL);

    // Initialize feature vector with small random values
    for (int j = 0; j < FEATURE_VECTOR_SIZE; j++) {
      filter->categories[i].feature_vector[j] = (float)rand() / RAND_MAX * 0.1f;
    }
  }
}

SecurityValidationStatus validateCriticalSecurity(const Neuron *neurons,
                                                  const float *weights,
                                                  const int *connections,
                                                  size_t max_neurons,
                                                  size_t max_connections) {
  SecurityValidationStatus status = {.critical_violation = false,
                                     .suspect_address = 0,
                                     .violation_type = NULL};

  // Heuristic Execution Flow Analysis
  for (size_t i = 0; i < max_neurons && !status.critical_violation; i++) {
    for (size_t j = 0; j < neurons[i].num_connections; j++) {
      // Out-of-bounds connection check
      if (static_cast<size_t>(connections[i * max_connections + j]) >=
          static_cast<size_t>(max_neurons)) {

        status.critical_violation = true;
        status.suspect_address =
            (uint64_t)&connections[i * max_connections + j];
        status.violation_type = "Out-of-bounds connection access";
        return status;
      }

      // Unusual Execution Flow Detection
      // Check for excessive connections
      if (neurons[i].num_connections > max_connections / 2) {
        status.critical_violation = true;
        status.suspect_address = (uint64_t)&neurons[i];
        status.violation_type = "Unusually high number of connections";
        return status;
      }

      // Detect potential cyclic dependencies
      size_t connection_target = connections[i * max_connections + j];
      for (size_t k = 0; k < neurons[connection_target].num_connections; k++) {
        if (neurons[connection_target].num_connections > max_connections / 3 &&
            connections[connection_target * max_connections + k] == i) {
          status.critical_violation = true;
          status.suspect_address = (uint64_t)&neurons[connection_target];
          status.violation_type = "Potential cyclic connection detected";
          return status;
        }
      }
    }
  }

  uint64_t system_memory_start =
      0x00007f0000000000; // Typical start of system memory mapping
  uint64_t system_memory_end =
      0xFFFFFFFFFFFF; // Extend memory range to end of address space

  for (size_t i = 0; i < max_neurons && !status.critical_violation; i++) {
    for (size_t j = 0; j < neurons[i].num_connections; j++) {
      uint64_t target_addr =
          (uint64_t)(&neurons[connections[i * max_connections + j]]);

      // Check if trying to jump to system memory
      if (target_addr >= system_memory_start &&
          target_addr <= system_memory_end) {
        status.critical_violation = true;
        status.suspect_address = target_addr;
        status.violation_type = "Attempted system memory access";
        break;
      }

      // Check for attempts to modify instruction pointer
      if ((target_addr & 0xFFFF000000000000) == 0xFFFF000000000000) {
        status.critical_violation = true;
        status.suspect_address = target_addr;
        status.violation_type = "Attempted code execution";
        break;
      }

      // Detect jumps to non-volatile (unaligned) addresses
      if ((target_addr % 8) != 0) {
        status.critical_violation = true;
        status.suspect_address = target_addr;
        status.violation_type = "Non-aligned memory access";
        break;
      }
    }
  }

  const unsigned char *mem_scan = (const unsigned char *)neurons;
  for (size_t i = 0; i < sizeof(Neuron) * max_neurons - 4; i++) {
    // Look for common shellcode signatures
    if ((mem_scan[i] == 0xCD && mem_scan[i + 1] == 0x80) || // int 0x80
        (mem_scan[i] == 0x0F && mem_scan[i + 1] == 0x05) || // syscall
        (mem_scan[i] == 0xEB &&
         mem_scan[i + 1] == 0xFE)) { // infinite loop (no-op)
      status.critical_violation = true;
      status.suspect_address = (uint64_t)&mem_scan[i];
      status.violation_type = "Detected potential shellcode";
      break;
    }
  }

  return status;
}

// Global variables for signal handling
static sigjmp_buf segv_jump_buffer;
static volatile sig_atomic_t segv_occurred = 0;

// Signal handler for catching segmentation faults
static void segv_handler(int sig) {
  segv_occurred = 1;
  siglongjmp(segv_jump_buffer, 1);
}

MemoryProtection validateMemoryAccess(const void *ptr, size_t size) {
  MemoryProtection protection = {0};

  // Null and low memory address check
  if (ptr == NULL || (uintptr_t)ptr < 0x1000) {
    return protection;
  }

  // Install signal handler
  struct sigaction sa, old_sa;
  sa.sa_handler = segv_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESETHAND;

  sigaction(SIGSEGV, &sa, &old_sa);

  // Reset segmentation fault flag
  segv_occurred = 0;

  // Use sigsetjmp for non-local goto
  if (sigsetjmp(segv_jump_buffer, 1) == 0) {
    // Attempt to read from the pointer
    volatile const char *test_ptr = (const char *)ptr;
    char dummy;

    // Try reading
    dummy = *test_ptr;
    (void)dummy;
    protection.is_readable = true;

    // Try writing (requires non-const pointer)
    char *writable_ptr = (char *)ptr;
    *writable_ptr = *writable_ptr;
    protection.is_writable = true;
  }

  // Restore original signal handler
  sigaction(SIGSEGV, &old_sa, NULL);

  // Additional POSIX memory region check
  int page_size = sysconf(_SC_PAGESIZE);
  void *page_start = (void *)((uintptr_t)ptr & ~(page_size - 1));

  // Check memory mappings
  int mem_status = msync(page_start, page_size, MS_ASYNC);
  if (mem_status == 0) {
    protection.region_size = page_size;
  }

  // Check executable memory
  protection.is_executable = (mprotect(page_start, page_size, PROT_EXEC) == 0);

  return protection;
}

void handleCriticalSecurityViolation(Neuron *neurons, float *weights,
                                     int *connections,
                                     const SecurityValidationStatus *status) {
  // Print violation details to stderr
  fprintf(stderr, "\nCRITICAL SECURITY VIOLATION DETECTED\n");
  fprintf(stderr, "Type: %s\n", status->violation_type);
  fprintf(stderr, "Suspect address: 0x%lx\n", status->suspect_address);

  // Convert suspect address to void pointer
  void *suspect_ptr = (void *)status->suspect_address;

  // Validate memory access
  MemoryProtection mem_protection =
      validateMemoryAccess(suspect_ptr, sizeof(Neuron));

  // Log memory protection details
  fprintf(stderr, "Memory Protection Check:\n");
  fprintf(stderr, "  Readable:     %s\n",
          mem_protection.is_readable ? "Yes" : "No");
  fprintf(stderr, "  Writable:     %s\n",
          mem_protection.is_writable ? "Yes" : "No");
  fprintf(stderr, "  Executable:   %s\n",
          mem_protection.is_executable ? "Yes" : "No");
  fprintf(stderr, "  Region Size:  %zu bytes\n", mem_protection.region_size);

  // Safe memory clearing only if writable and valid
  if (mem_protection.is_writable &&
      mem_protection.region_size >= sizeof(Neuron)) {
    // Use secure memory clearing with explicit zero filling
    volatile char *ptr = (volatile char *)suspect_ptr;
    for (size_t i = 0; i < sizeof(Neuron); i++) {
      ptr[i] = 0;
    }
    __sync_synchronize(); // Memory barrier to ensure zeroing
    fprintf(stderr, "Memory cleared safely with secure zeroing.\n");
  } else {
    fprintf(stderr, "UNSAFE TO CLEAR: Invalid memory region\n");
  }

  // Log violation to file
  FILE *log_file = fopen("security_violations.log", "a");
  if (log_file) {
    fprintf(log_file, "Violation Type: %s\n", status->violation_type);
    fprintf(log_file, "Suspect Address: 0x%lx\n", status->suspect_address);
    fprintf(log_file, "Memory Protection: R:%d W:%d X:%d Size:%zu\n",
            mem_protection.is_readable, mem_protection.is_writable,
            mem_protection.is_executable, mem_protection.region_size);
    fclose(log_file);
  }
}

float computeBeliefStability(const SelfIdentitySystem *system,
                             uint32_t belief_index) {
  if (static_cast<int>(belief_index) >= static_cast<int>(system->num_beliefs)) {
    return 0.0f;
  }

  float stability = 1.0f;
  float temporal_variance = 0.0f;
  float coherence_impact = 0.0f;

  // Check current belief against temporal coherence
  for (int i = 0; i < system->coherence_window - 1; i++) {
    float current =
        system->temporal_coherence[i * system->num_beliefs + belief_index];
    float next =
        system
            ->temporal_coherence[(i + 1) * system->num_beliefs + belief_index];
    float diff = current - next;
    temporal_variance += diff * diff;
  }
  temporal_variance /= (system->coherence_window - 1);

  // Calculate coherence impact
  for (int i = 0; i < system->coherence_window - 1; i++) {
    coherence_impact +=
        system->temporal_coherence[i * system->num_beliefs + belief_index];
  }
  coherence_impact /= system->coherence_window;

  // Compare with reference state if available
  float reference_deviation = 0.0f;
  if (system->verification.reference_state &&
      static_cast<int>(belief_index) <
          static_cast<int>(system->verification.state_size)) {
    float ref_belief = system->verification.reference_state[belief_index];
    float current_belief = system->belief_system[belief_index];
    reference_deviation = fabsf(current_belief - ref_belief);
  }

  // Compute final stability scoe
  stability -= sqrtf(temporal_variance); // Reduce stability based on variance
  stability += coherence_impact * 0.3f;  // Boost stability based on coherence
  stability -= reference_deviation *
               0.2f; // Reduce stability based on reference deviation
  stability = fmaxf(0.0f, fminf(1.0f, stability)); // Clamp between 0 and 1

  // Adjust based on system confidence and adaptation rate
  stability *= (0.7f + 0.3f * system->confidence_level); // Scale by confidence
  stability *=
      (1.0f - 0.2f * system->adaptation_rate); // Adjust for adaptation rate

  return stability;
}

// Create a backup of the current identity system state
SelfIdentityBackup *createIdentityBackup(const SelfIdentitySystem *system) {
  SelfIdentityBackup *backup =
      (SelfIdentityBackup *)malloc(sizeof(SelfIdentityBackup));
  if (!backup)
    return NULL;

  // Copy scalar values
  backup->num_core_values = system->num_core_values;
  backup->num_beliefs = system->num_beliefs;
  backup->num_markers = system->num_markers;
  backup->history_size = system->history_size;
  backup->pattern_size = system->pattern_size;
  backup->coherence_window = system->coherence_window;
  backup->state_size = system->verification.state_size;
  backup->consistency_score = system->consistency_score;
  backup->adaptation_rate = system->adaptation_rate;
  backup->confidence_level = system->confidence_level;

  // Allocate and copy arrays
  backup->core_values =
      (float *)malloc(system->num_core_values * sizeof(float));
  backup->belief_system = (float *)malloc(system->num_beliefs * sizeof(float));
  backup->identity_markers =
      (float *)malloc(system->num_markers * sizeof(float));
  backup->experience_history =
      (float *)malloc(system->history_size * sizeof(float));
  backup->behavioral_patterns =
      (float *)malloc(system->pattern_size * sizeof(float));
  backup->temporal_coherence = (float *)malloc(
      system->coherence_window * system->num_beliefs * sizeof(float));
  backup->reference_state =
      (float *)malloc(system->verification.state_size * sizeof(float));

  if (backup->core_values && system->core_values)
    memcpy(backup->core_values, system->core_values,
           system->num_core_values * sizeof(float));
  if (backup->belief_system && system->belief_system)
    memcpy(backup->belief_system, system->belief_system,
           system->num_beliefs * sizeof(float));
  if (backup->identity_markers && system->identity_markers)
    memcpy(backup->identity_markers, system->identity_markers,
           system->num_markers * sizeof(float));
  if (backup->experience_history && system->experience_history)
    memcpy(backup->experience_history, system->experience_history,
           system->history_size * sizeof(float));
  if (backup->behavioral_patterns && system->behavioral_patterns)
    memcpy(backup->behavioral_patterns, system->behavioral_patterns,
           system->pattern_size * sizeof(float));
  if (backup->temporal_coherence && system->temporal_coherence)
    memcpy(backup->temporal_coherence, system->temporal_coherence,
           system->coherence_window * system->num_beliefs * sizeof(float));
  if (backup->reference_state && system->verification.reference_state)
    memcpy(backup->reference_state, system->verification.reference_state,
           system->verification.state_size * sizeof(float));

  return backup;
}

IdentityAnalysis analyzeIdentitySystem(const SelfIdentitySystem *system) {
  IdentityAnalysis analysis = {0};

  // Analyze core values
  for (uint32_t i = 0; i < static_cast<uint32_t>(system->num_core_values);
       i++) {
    float stability = fabsf(system->core_values[i] -
                            (system->verification.reference_state
                                 ? system->verification.reference_state[i]
                                 : 0.0f));
    if (stability > system->verification.threshold) {
      analysis.core_value_conflicts++;
    }
  }

  // Analyze beliefs
  for (uint32_t i = 0; i < static_cast<uint32_t>(system->num_beliefs); i++) {
    if (computeBeliefStability(system, i) < system->verification.threshold) {
      analysis.belief_conflicts++;
    }
  }

  // Analyze identity markers
  for (uint32_t i = 0; i < static_cast<uint32_t>(system->num_markers); i++) {
    float marker_variance = 0.0f;
    for (size_t j = 0; j < static_cast<size_t>(system->coherence_window); j++) {
      marker_variance +=
          fabsf(system->identity_markers[i] -
                system->temporal_coherence[j * system->num_markers + i]);
    }
    marker_variance /= system->coherence_window;
    if (marker_variance > system->verification.threshold) {
      analysis.marker_conflicts++;
    }
  }

  // Calculate temporal instability
  float total_temporal_variance = 0.0f;
  for (size_t i = 0; i < static_cast<size_t>(system->coherence_window) - 1;
       i++) {
    for (uint32_t j = 0; j < static_cast<uint32_t>(system->num_beliefs); j++) {
      float diff =
          system->temporal_coherence[i * system->num_beliefs + j] -
          system->temporal_coherence[(i + 1) * system->num_beliefs + j];
      total_temporal_variance += diff * diff;
    }
  }
  analysis.temporal_instability =
      sqrtf(total_temporal_variance /
            ((system->coherence_window - 1) * system->num_beliefs));

  // If NaN, initialize to 0
  if (isnan(analysis.temporal_instability)) {
    analysis.temporal_instability = 0.0f;
  }

  // Calculate pattern deviation
  float pattern_diff = 0.0f;
  for (uint32_t i = 0; i < static_cast<uint32_t>(system->pattern_size); i++) {
    // Check if either of the values is NaN
    if (isnan(system->behavioral_patterns[i]) ||
        isnan(system->verification.reference_state
                  ? system->verification.reference_state[i]
                  : 0.0f)) {
      pattern_diff = 0.0f; // Reset pattern_diff to 0
      break;               // Exit the loop early since we encountered NaN
    }
    pattern_diff += fabsf(system->behavioral_patterns[i] -
                          (system->verification.reference_state
                               ? system->verification.reference_state[i]
                               : 0.0f));
  }
  analysis.pattern_deviation = pattern_diff / system->pattern_size;

  // If NaN, initialize to 0
  if (isnan(analysis.pattern_deviation)) {
    analysis.pattern_deviation = 0.0f;
  }

  // Calculate overall metrics
  analysis.overall_consistency = system->consistency_score;
  analysis.confidence_impact =
      system->confidence_level - (analysis.temporal_instability * 0.3f +
                                  analysis.pattern_deviation * 0.2f);

  return analysis;
}

void restoreIdentityFromBackup(SelfIdentitySystem *system,
                               const SelfIdentityBackup *backup) {
  if (!system || !backup)
    return;

  // Restore scalar values
  system->num_core_values = backup->num_core_values;
  system->num_beliefs = backup->num_beliefs;
  system->num_markers = backup->num_markers;
  system->history_size = backup->history_size;
  system->pattern_size = backup->pattern_size;
  system->coherence_window = backup->coherence_window;
  system->verification.state_size = backup->state_size;
  system->consistency_score = backup->consistency_score;
  system->adaptation_rate = backup->adaptation_rate;
  system->confidence_level = backup->confidence_level;

  // Restore arrays
  if (backup->core_values)
    memcpy(system->core_values, backup->core_values,
           backup->num_core_values * sizeof(float));
  if (backup->belief_system)
    memcpy(system->belief_system, backup->belief_system,
           backup->num_beliefs * sizeof(float));
  if (backup->identity_markers)
    memcpy(system->identity_markers, backup->identity_markers,
           backup->num_markers * sizeof(float));
  if (backup->experience_history)
    memcpy(system->experience_history, backup->experience_history,
           backup->history_size * sizeof(float));
  if (backup->behavioral_patterns)
    memcpy(system->behavioral_patterns, backup->behavioral_patterns,
           backup->pattern_size * sizeof(float));
  if (backup->temporal_coherence)
    memcpy(system->temporal_coherence, backup->temporal_coherence,
           backup->coherence_window * backup->num_beliefs * sizeof(float));
  if (backup->reference_state)
    memcpy(system->verification.reference_state, backup->reference_state,
           backup->state_size * sizeof(float));
}

void freeIdentityBackup(SelfIdentityBackup *backup) {
  if (!backup)
    return;

  free(backup->core_values);
  free(backup->belief_system);
  free(backup->identity_markers);
  free(backup->experience_history);
  free(backup->behavioral_patterns);
  free(backup->temporal_coherence);
  free(backup->reference_state);
  free(backup);
}

void computeGradientFeedback(float feedback[], Neuron *neuron,
                             float target_output[], int size) {
  for (int i = 0; i < size; i++) {
    feedback[i] = 2.0f * (neuron[i].output - target_output[i]);
  }
}

void addSymbol(int symbol_id, const char *description) {
  if (num_symbols < MAX_SYMBOLS) {
    symbol_table[num_symbols].symbol_id = symbol_id;
    strncpy(symbol_table[num_symbols].description, description, 255);
    num_symbols++;
  }
}

void addQuestion(int question_id, int symbol_ids[], int num_symbols) {
  if (num_questions < MAX_QUESTIONS) {
    question_table[num_questions].question_id = question_id;
    memcpy(question_table[num_questions].symbol_ids, symbol_ids,
           num_symbols * sizeof(int));
    question_table[num_questions].num_symbols = num_symbols;
    num_questions++;
  }
}

int getTokenIndex(const char *token) {
  // Iterate through the vocabulary to find the token
  for (int i = 0; i < VOCAB_SIZE; i++) {
    if (strcmp(vocabulary[i].word, token) == 0) {
      return i; // Return the index if found
    }
  }
  return -1; // Return -1 for out-of-vocabulary tokens
}

void createSemanticVector(const char *text, float *vector, int vectorSize,
                          float (*embeddings)[EMBEDDING_SIZE]) {
  const char *delimiters = " ";
  char *textCopy = strdup(text);
  char *token = strtok(textCopy, delimiters);

  // Initialize vector to zero
  for (int i = 0; i < vectorSize; i++) {
    vector[i] = 0.0f;
  }

  int tokenCount = 0;

  // Process each token
  while (token != NULL) {
    int tokenIndex = getTokenIndex(token);
    if (tokenIndex != -1) { // Only proceed if token is found in vocabulary
      for (int i = 0; i < EMBEDDING_SIZE; i++) {
        vector[i] += embeddings[tokenIndex][i];
      }
      tokenCount++;
    }
    token = strtok(NULL, delimiters);
  }

  // Average the embeddings
  if (tokenCount > 0) {
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      vector[i] /= tokenCount;
    }
  }

  free(textCopy);
}

void storeQuestionAndAnswer(MemorySystem *memorySystem, const char *question,
                            const char *answer, int timestamp) {
  // Only proceed if we have space or can consolidate
  if (memorySystem->size >= memorySystem->capacity) {
    // Try to consolidate first
    consolidateMemory(memorySystem);
    // If still full, we need to overwrite oldest memory
    if (memorySystem->size >= memorySystem->capacity) {
      printf("Warning: Memory system full. Overwriting oldest entry.\n");
    }
  }

  // Create a new memory entry
  MemoryEntry newEntry;

  // Convert question and answer into a semantic vector representation
  float questionVector[MEMORY_VECTOR_SIZE];
  float answerVector[MEMORY_VECTOR_SIZE];

  createSemanticVector(question, questionVector, MEMORY_VECTOR_SIZE,
                       embeddings);
  createSemanticVector(answer, answerVector, MEMORY_VECTOR_SIZE, embeddings);

  // Combine question and answer vectors by averaging
  for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
    newEntry.vector[i] = (questionVector[i] + answerVector[i]) / 2.0f;
  }

  // Set importance based on question complexity and answer quality
  newEntry.importance =
      0.5f + (strlen(question) * 0.01f) + (strlen(answer) * 0.005f);
  // Set timestamp
  newEntry.timestamp = timestamp;

  // Add to memory system
  if (memorySystem->size < memorySystem->capacity) {
    memorySystem->entries[memorySystem->size] = newEntry;
    memorySystem->size++;
  } else {
    // Find least important memory to replace
    int replace_idx = 0;
    float min_importance = memorySystem->entries[0].importance;
    for (size_t i = 1; i < memorySystem->size; i++) {
      if (memorySystem->entries[i].importance < min_importance) {
        min_importance = memorySystem->entries[i].importance;
        replace_idx = i;
      }
    }
    // Replace least important memory
    memorySystem->entries[replace_idx] = newEntry;
  }

  // Determine if this memory should also be in short-term memory
  if (newEntry.importance >
      memorySystem->hierarchy.short_term.importance_threshold) {
    // Check if we have space in short-term memory
    if (memorySystem->hierarchy.short_term.size <
        memorySystem->hierarchy.short_term.capacity) {
      memorySystem->hierarchy.short_term
          .entries[memorySystem->hierarchy.short_term.size] = newEntry;
      memorySystem->hierarchy.short_term.size++;
    } else {
      // Find least important short-term memory to replace
      int st_replace_idx = 0;
      float st_min_importance =
          memorySystem->hierarchy.short_term.entries[0].importance;
      for (size_t i = 1; i < memorySystem->hierarchy.short_term.size; i++) {
        if (memorySystem->hierarchy.short_term.entries[i].importance <
            st_min_importance) {
          st_min_importance =
              memorySystem->hierarchy.short_term.entries[i].importance;
          st_replace_idx = i;
        }
      }
      // Replace if new memory is more important
      if (newEntry.importance > st_min_importance) {
        memorySystem->hierarchy.short_term.entries[st_replace_idx] = newEntry;
      }
    }
  }
}

void updateContextAnswer(GlobalContextManager *contextManager,
                         const char *question, const char *answer) {
  // Find or create a context node for this type of interaction
  ContextNode *currentNode = contextManager->root;
  char contextName[64] = "QA_Interaction";
  bool found = false;

  // Check if we already have this context
  for (uint32_t i = 0; i < currentNode->num_children; i++) {
    if (strcmp(currentNode->children[i]->name, contextName) == 0) {
      currentNode = currentNode->children[i];
      found = true;
      break;
    }
  }

  // If not found, create a new context node
  if (!found) {
    if (currentNode->num_children < currentNode->max_children) {
      // Create new node
      ContextNode *newNode =
          static_cast<ContextNode *>(malloc(sizeof(ContextNode)));

      newNode->name = strdup(contextName);
      newNode->importance = 0.7f; // QA interactions are important
      // Initialize state vector
      newNode->vector_size = contextManager->vector_size;
      newNode->state_vector =
          static_cast<float *>(malloc(sizeof(float) * newNode->vector_size));

      // Initialize state vector with a semantic vector for the context name
      float contextNameVector[MEMORY_VECTOR_SIZE] = {0.0f}; // Placeholder
      for (uint32_t i = 0; i < MEMORY_VECTOR_SIZE; i++) {
        newNode->state_vector[i] = contextNameVector[i];
      }

      // Initialize children
      newNode->children = NULL;
      newNode->num_children = 0;
      newNode->max_children = contextManager->max_children_per_node;
      // Set parent
      newNode->parent = currentNode;
      // Set temporal relevance to high (it's happening now)
      newNode->temporal_relevance = 1.0f;
      // Set timestamp
      newNode->last_updated = time(NULL);

      // Add to parent's children
      currentNode->children = static_cast<ContextNode **>(
          realloc(currentNode->children,
                  sizeof(ContextNode *) * (currentNode->num_children + 1)));

      currentNode->children[currentNode->num_children] = newNode;
      currentNode->num_children++;
      // Update total nodes count
      contextManager->total_nodes++;
      // Set current node to new node
      currentNode = newNode;
    }
  }

  // Update context state vector based on question and answer
  if (currentNode != contextManager->root) {
    // Create semantic vectors for question and answer
    float questionVector[MEMORY_VECTOR_SIZE];
    float answerVector[MEMORY_VECTOR_SIZE];

    createSemanticVector(question, questionVector, MEMORY_VECTOR_SIZE,
                         embeddings);
    createSemanticVector(answer, answerVector, MEMORY_VECTOR_SIZE, embeddings);

    // Combine question and answer vectors by averaging
    float combinedVector[MEMORY_VECTOR_SIZE];
    for (uint32_t i = 0; i < MEMORY_VECTOR_SIZE; i++) {
      combinedVector[i] = (questionVector[i] + answerVector[i]) / 2.0f;
    }

    // Update state vector with semantic influence
    for (uint32_t i = 0; i < currentNode->vector_size; i++) {
      float semanticInfluence =
          (i < MEMORY_VECTOR_SIZE) ? combinedVector[i] : 0.0f;
      // Update state with decay
      currentNode->state_vector[i] =
          (currentNode->state_vector[i] * (1.0f - contextManager->decay_rate)) +
          (semanticInfluence * contextManager->decay_rate);
    }

    // Update last accessed time
    currentNode->last_updated = time(NULL);
    // Update temporal relevance to maximum
    currentNode->temporal_relevance = 1.0f;
    // Update global context
    for (uint32_t i = 0; i < contextManager->vector_size; i++) {
      contextManager->global_context_vector[i] =
          (contextManager->global_context_vector[i] *
           (1.0f - contextManager->decay_rate)) +
          (currentNode->state_vector[i] * currentNode->importance *
           contextManager->decay_rate);
    }
  }
}

static unsigned int fnv1a_hash(const char *str) {
  unsigned int hash = 2166136261u;
  while (*str) {
    hash ^= (unsigned char)*str++;
    hash *= 16777619u;
  }
  return hash;
}

// Simple stemming function (removes common suffixes)
static void simple_stem(char *word) {
  int len = strlen(word);
  if (len < 4)
    return;

  // Convert to lowercase for stemming
  for (int i = 0; word[i]; i++) {
    word[i] = tolower(word[i]);
  }

  // Remove common suffixes
  if (len > 4) {
    if (strcmp(word + len - 3, "ing") == 0) {
      word[len - 3] = '\0';
    } else if (strcmp(word + len - 2, "ed") == 0) {
      word[len - 2] = '\0';
    } else if (strcmp(word + len - 2, "er") == 0) {
      word[len - 2] = '\0';
    } else if (strcmp(word + len - 1, "s") == 0 && word[len - 2] != 's') {
      word[len - 1] = '\0';
    }
  }
}

// Check if word is a stop word
static int is_stop_word(const char *word) {
  static const char *stop_words[] = {
      "the",   "a",      "an",  "and",   "or",  "but",  "in",
      "on",    "at",     "to",  "for",   "of",  "with", "by",
      "is",    "are",    "was", "were",  "be",  "been", "have",
      "has",   "had",    "do",  "does",  "did", "will", "would",
      "could", "should", "may", "might", "can", "this", "that",
      "these", "those",  "i",   "you",   "he",  "she",  "it",
      "we",    "they",   "me",  "him",   "her", "us",   "them"};

  int num_stop_words = sizeof(stop_words) / sizeof(stop_words[0]);
  for (int i = 0; i < num_stop_words; i++) {
    if (strcmp(word, stop_words[i]) == 0) {
      return 1;
    }
  }
  return 0;
}

static int tokenize_text(const char *text, char tokens[][MAX_TOKEN_LENGTH]) {
  char text_copy[MAX_TEXT_LENGTH];
  strncpy(text_copy, text, MAX_TEXT_LENGTH - 1);
  text_copy[MAX_TEXT_LENGTH - 1] = '\0';

  int num_tokens = 0;
  char *token = strtok(text_copy, " \t\n\r.,!?;:()[]{}\"'");

  while (token != NULL && num_tokens < MAX_TOKENS) {
    // Convert to lowercase and check length
    int len = strlen(token);
    if (len >= 2 && len < MAX_TOKEN_LENGTH) {
      char processed_token[MAX_TOKEN_LENGTH];
      strncpy(processed_token, token, MAX_TOKEN_LENGTH - 1);
      processed_token[MAX_TOKEN_LENGTH - 1] = '\0';

      // Convert to lowercase
      for (int i = 0; processed_token[i]; i++) {
        processed_token[i] = tolower(processed_token[i]);
      }

      // Skip stop words
      if (!is_stop_word(processed_token)) {
        simple_stem(processed_token);
        strncpy(tokens[num_tokens], processed_token, MAX_TOKEN_LENGTH - 1);
        tokens[num_tokens][MAX_TOKEN_LENGTH - 1] = '\0';
        num_tokens++;
      }
    }
    token = strtok(NULL, " \t\n\r.,!?;:()[]{}\"'");
  }

  return num_tokens;
}

// Generate n-grams from tokens
static void add_ngrams_to_vector(float *memory_vector,
                                 char tokens[][MAX_TOKEN_LENGTH],
                                 int num_tokens, int n) {
  char ngram[MAX_TOKEN_LENGTH * NGRAM_SIZE];

  for (int i = 0; i <= num_tokens - n; i++) {
    // Create n-gram string
    strcpy(ngram, tokens[i]);
    for (int j = 1; j < n; j++) {
      strcat(ngram, "_");
      strcat(ngram, tokens[i + j]);
    }

    // Hash and add to vector
    unsigned int hash = fnv1a_hash(ngram);
    int index = hash % MEMORY_VECTOR_SIZE;
    memory_vector[index] += 1.0f / (float)n; // Weight by n-gram size
  }
}

// TF-IDF style weighting
static void apply_tf_weighting(float *memory_vector,
                               char tokens[][MAX_TOKEN_LENGTH],
                               int num_tokens) {
  // Count term frequencies
  float tf_counts[MEMORY_VECTOR_SIZE] = {0};

  for (int i = 0; i < num_tokens; i++) {
    unsigned int hash = fnv1a_hash(tokens[i]);
    int index = hash % MEMORY_VECTOR_SIZE;
    tf_counts[index] += 1.0f;
  }

  // Apply TF weighting (log normalization)
  for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
    if (tf_counts[i] > 0) {
      memory_vector[i] *= (1.0f + logf(tf_counts[i]));
    }
  }
}

void computeMemoryVectorFromText(float *memory_vector, const char *question,
                                 const char *answer) {
  // Initialize the memory vector to zero
  memset(memory_vector, 0, MEMORY_VECTOR_SIZE * sizeof(float));

  // Combine question and answer with different weights
  char combined_text[MAX_TEXT_LENGTH];
  snprintf(combined_text, sizeof(combined_text), "%s %s %s", question, question,
           answer); // Weight question 2x

  // Enhanced tokenization with preprocessing
  char tokens[MAX_TOKENS][MAX_TOKEN_LENGTH];
  int num_tokens = tokenize_text(combined_text, tokens);

  if (num_tokens == 0)
    return;

  // Add unigrams (single words)
  for (int i = 0; i < num_tokens; i++) {
    unsigned int hash = fnv1a_hash(tokens[i]);
    int index = hash % MEMORY_VECTOR_SIZE;
    memory_vector[index] += 1.0f;
  }

  // Add bigrams (word pairs) if we have enough tokens
  if (num_tokens > 1) {
    add_ngrams_to_vector(memory_vector, tokens, num_tokens, 2);
  }

  // Add trigrams (word triplets) if we have enough tokens
  if (num_tokens > 2) {
    add_ngrams_to_vector(memory_vector, tokens, num_tokens, 3);
  }

  // Apply TF-style weighting
  apply_tf_weighting(memory_vector, tokens, num_tokens);

  // Position-based weighting (early words get higher weight)
  for (int i = 0; i < num_tokens; i++) {
    unsigned int hash = fnv1a_hash(tokens[i]);
    int index = hash % MEMORY_VECTOR_SIZE;
    float position_weight = 1.0f + (1.0f / (1.0f + (float)i * 0.1f));
    memory_vector[index] *= position_weight;
  }

  // L2 normalization
  float norm = 0.0f;
  for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
    norm += memory_vector[i] * memory_vector[i];
  }

  norm = sqrtf(norm);
  if (norm > 1e-8f) { // Avoid division by very small numbers
    for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
      memory_vector[i] /= norm;
    }
  }
}

float computeImportanceFromText(const char *question, const char *answer) {
  float importance = 0.0f;

  int question_length = strlen(question);
  int answer_length = strlen(answer);
  importance += 0.1f * (question_length + answer_length);

  const char *keywords[] = {"error", "goal", "priority", "critical",
                            "important"};
  int num_keywords = sizeof(keywords) / sizeof(keywords[0]);
  for (int i = 0; i < num_keywords; i++) {
    if (strstr(question, keywords[i])) {
      importance += 5.0f; // Increase importance if keyword is found
    }
    if (strstr(answer, keywords[i])) {
      importance += 5.0f;
    }
  }

  if (strstr(answer, "Pattern Recognition")) {
    importance +=
        8.0f; // Increase importance for pattern recognition-related content
  }
  if (strstr(answer, "Numerical Computation")) {
    importance +=
        7.5f; // Increase importance for numerical computation-related content
  }
  if (strstr(answer, "Sequence Learning")) {
    importance += 9.0f; // High importance for sequence learning
  }
  if (strstr(answer, "Classification")) {
    importance += 7.0f; // Moderate importance for classification
  }
  if (strstr(answer, "Prediction")) {
    importance += 10.0f; // High importance for prediction-related content
  }
  if (strstr(answer, "Optimization")) {
    importance += 8.5f; // Important for optimization problems
  }
  if (strstr(answer, "Error Correction")) {
    importance += 9.5f; // Important for error correction-related content
  }
  if (strstr(answer, "Memory Consolidation")) {
    importance += 6.0f; // Memory-related topics have moderate importance
  }

  // Normalize importance to a reasonable range (e.g., 0 to 100)
  importance = fminf(importance, 100.0f);
  importance = fmaxf(importance, 0.0f);

  return importance;
}

void addQuestionAndAnswerToMemory(
    MemorySystem *memorySystem, WorkingMemorySystem *workingMemory,
    const char *question, const char *answer,
    float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE]) {
  // Create a memory entry for the question and answer
  MemoryEntry entry;
  entry.timestamp = getCurrentTime();
  entry.importance = computeImportanceFromText(question, answer);

  // Convert the question and answer into a memory vector
  computeMemoryVectorFromText(entry.vector, question, answer);

  // Handle Working Memory System first
  if (entry.importance > workingMemory->focus.attention_threshold) {
    // Add to focused attention
    if (workingMemory->focus.size < workingMemory->focus.capacity) {
      WorkingMemoryEntry enhanced;
      enhanced.features = (float *)malloc(FEATURE_VECTOR_SIZE * sizeof(float));
      extractSemanticFeatures(entry.vector, enhanced.features,
                              feature_projection_matrix);
      enhanced.context_vector =
          (float *)malloc(CONTEXT_VECTOR_SIZE * sizeof(float));
      memcpy(enhanced.context_vector, workingMemory->global_context,
             CONTEXT_VECTOR_SIZE * sizeof(float));
      workingMemory->focus.entries[workingMemory->focus.size++] = enhanced;
      updateSemanticClusters(workingMemory, &enhanced);
    }
  } else {
    // Add to active memory
    if (workingMemory->active.size < workingMemory->active.capacity) {
      WorkingMemoryEntry enhanced;
      enhanced.features = (float *)malloc(FEATURE_VECTOR_SIZE * sizeof(float));
      extractSemanticFeatures(entry.vector, enhanced.features,
                              feature_projection_matrix);
      enhanced.context_vector =
          (float *)malloc(CONTEXT_VECTOR_SIZE * sizeof(float));
      memcpy(enhanced.context_vector, workingMemory->global_context,
             CONTEXT_VECTOR_SIZE * sizeof(float));
      workingMemory->active.entries[workingMemory->active.size++] = enhanced;
      updateSemanticClusters(workingMemory, &enhanced);
    }
  }

  // Update global context
  updateContext(workingMemory);

  // Then handle original hierarchical storage - NOW WITH INTELLIGENT
  // REPLACEMENT
  if (entry.importance >=
      memorySystem->hierarchy.long_term.importance_threshold) {
    if (memorySystem->hierarchy.long_term.size <
        memorySystem->hierarchy.long_term.capacity) {
      memorySystem->hierarchy.long_term
          .entries[memorySystem->hierarchy.long_term.size++] = entry;
    } else {
      // Find multiple least important and replace strategically
      unsigned int replace_count;
      int *least_important =
          findLeastImportantMemory(memorySystem->hierarchy.long_term.entries,
                                   memorySystem->hierarchy.long_term.size,
                                   10, // Get 10 least important for Q&A context
                                   &replace_count);

      if (least_important && replace_count > 0) {
        // For Q&A, be more selective - only replace if significantly better
        float worst_importance =
            memorySystem->hierarchy.long_term.entries[least_important[0]]
                .importance;
        if (entry.importance >
            worst_importance * 1.2f) { // 20% better threshold
          memorySystem->hierarchy.long_term.entries[least_important[0]] = entry;
        }
        free(least_important);
      }
    }
  } else if (entry.importance >=
             memorySystem->hierarchy.medium_term.importance_threshold) {
    if (memorySystem->hierarchy.medium_term.size <
        memorySystem->hierarchy.medium_term.capacity) {
      memorySystem->hierarchy.medium_term
          .entries[memorySystem->hierarchy.medium_term.size++] = entry;
    } else {
      // Smarter replacement for Q&A in medium term
      unsigned int replace_count;
      int *least_important =
          findLeastImportantMemory(memorySystem->hierarchy.medium_term.entries,
                                   memorySystem->hierarchy.medium_term.size,
                                   7, // Get 7 candidates
                                   &replace_count);

      if (least_important && replace_count > 0) {
        // Look for old, low-importance entries to replace
        int best_replacement = -1;
        float best_score = -1.0f;

        for (unsigned int i = 0; i < replace_count; i++) {
          int idx = least_important[i];
          MemoryEntry *candidate =
              &memorySystem->hierarchy.medium_term.entries[idx];
          unsigned int age = entry.timestamp - candidate->timestamp;

          // Score combines low importance and high age
          float score =
              (1.0f / (candidate->importance + 0.1f)) + (age * 0.001f);
          if (score > best_score) {
            best_score = score;
            best_replacement = idx;
          }
        }

        if (best_replacement >= 0) {
          memorySystem->hierarchy.medium_term.entries[best_replacement] = entry;
        }
        free(least_important);
      } else {
        consolidateToHigherLevel(memorySystem);
      }
    }
  } else {
    if (memorySystem->hierarchy.short_term.size <
        memorySystem->hierarchy.short_term.capacity) {
      memorySystem->hierarchy.short_term
          .entries[memorySystem->hierarchy.short_term.size++] = entry;
    } else {
      // For short term Q&A, more aggressive replacement
      unsigned int replace_count;
      int *least_important = findLeastImportantMemory(
          memorySystem->hierarchy.short_term.entries,
          memorySystem->hierarchy.short_term.size,
          memorySystem->hierarchy.short_term.size / 3, // Get bottom third
          &replace_count);

      if (least_important && replace_count > 0) {
        // Replace oldest among the least important
        int oldest_idx = least_important[0];
        unsigned int oldest_time =
            memorySystem->hierarchy.short_term.entries[oldest_idx].timestamp;

        for (unsigned int i = 1; i < replace_count; i++) {
          int idx = least_important[i];
          if (memorySystem->hierarchy.short_term.entries[idx].timestamp <
              oldest_time) {
            oldest_time =
                memorySystem->hierarchy.short_term.entries[idx].timestamp;
            oldest_idx = idx;
          }
        }

        memorySystem->hierarchy.short_term.entries[oldest_idx] = entry;
        free(least_important);
      } else {
        consolidateToMediumTerm(memorySystem);
      }
    }
  }

  // Update original structure for compatibility
  memorySystem->entries[memorySystem->head] = entry;
  memorySystem->head = (memorySystem->head + 1) % memorySystem->capacity;
  if (memorySystem->size < memorySystem->capacity) {
    memorySystem->size++;
  }
}

void getEmotionName(int emotion_id, char *name) {
  static const char *emotion_names[] = {"love", "hate", "joy", "fear"};

  if (emotion_id >= 0 && emotion_id < MAX_EMOTION_TYPES &&
      static_cast<size_t>(emotion_id) <
          sizeof(emotion_names) / sizeof(emotion_names[0])) {
    strcpy(name, emotion_names[emotion_id]);
  } else {
    strcpy(name, "unknown");
  }
}

void askQuestion(
    int question_id, Neuron *neurons, float *input_tensor,
    MemorySystem *memorySystem, float *learning_rate,
    NetworkStateSnapshot *stateSnapshot, GlobalContextManager *contextManager,
    IntrinsicMotivation *motivation, GoalSystem *goalSystem,
    WorkingMemorySystem *workingMemory, SelfIdentitySystem *identitySystem,
    MetacognitionMetrics *metacognition, KnowledgeFilter *filter,
    EmotionalSystem *emotionalSystem, ImaginationSystem *imaginationSystem,
    SocialSystem *socialSystem,
    float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE]) {
  if (question_id < 0 || question_id >= num_questions) {
    printf("Invalid question ID\n");
    return;
  }

  InternalQuestion *question = &question_table[question_id];
  char fullQuestionStr[1024] = "";
  char fullAnswerStr[1024] = "";

  for (int i = 0; i < question->num_symbols; i++) {
    int symbol_id = question->symbol_ids[i];
    if (symbol_id < 0 || symbol_id >= num_symbols) {
      printf("Invalid symbol ID\n");
      continue;
    }

    InternalSymbol *symbol = &symbol_table[symbol_id];
    printf("Question: %s\n", symbol->description);

    // Accumulate the question text
    strcat(fullQuestionStr, symbol->description);
    strcat(fullQuestionStr, " ");

    char answerBuffer[256] = "";

    if (symbol_id == 0) {
      for (uint32_t i = 1; i < static_cast<uint32_t>(filter->num_categories);
           i++) {
        // Find the last accessed category
        KnowledgeCategory *last_category = &filter->categories[0];
        for (int i = 1; i < filter->num_categories; i++) {
          if (filter->categories[i].last_accessed >
              last_category->last_accessed) {
            last_category = &filter->categories[i];
          }
        }

        // Store the last accessed category name in answerBuffer
        snprintf(answerBuffer, sizeof(last_category->name), "%s",
                 last_category->name);
      }
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 1) {
      float error_rate = computeErrorRate(neurons, input_tensor);
      sprintf(answerBuffer, "Current error rate is %.2f", error_rate);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 2) {
      sprintf(answerBuffer, "Current learning rate is %.4f", *learning_rate);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 3) {
      sprintf(answerBuffer, "Current memory usage is %u/%u", memorySystem->size,
              memorySystem->capacity);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 4) {
      sprintf(answerBuffer, "Short-term memory has %u/%u entries",
              memorySystem->hierarchy.short_term.size,
              memorySystem->hierarchy.short_term.capacity);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 5) {
      sprintf(answerBuffer, "Long-term memory has %u/%u entries",
              memorySystem->hierarchy.long_term.size,
              memorySystem->hierarchy.long_term.capacity);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 6) {
      sprintf(answerBuffer, "Current network step is %d", stateSnapshot->step);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 7) {
      sprintf(answerBuffer,
              "Global context has %u total nodes with a decay rate of %.4f",
              contextManager->total_nodes, contextManager->decay_rate);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 8) {
      float avg_prediction_error = 0.0f;
      for (int j = 0; j < MAX_NEURONS; j++) {
        avg_prediction_error += predictive_params[j].prediction_error;
      }
      avg_prediction_error /= MAX_NEURONS;
      sprintf(answerBuffer, "Average prediction error across neurons is %.4f",
              avg_prediction_error);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 9) {
      sprintf(answerBuffer,
              "Working memory focus has %u/%u entries with attention threshold "
              "%.4f",
              workingMemory->focus.size, workingMemory->focus.capacity,
              workingMemory->focus.attention_threshold);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 10) {
      sprintf(answerBuffer,
              "Current curiosity drive is %.2f with exploration rate %.2f",
              motivation->curiosity_drive, motivation->exploration_rate);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 11) {
      int active_goals = 0;
      for (int j = 0; j < goalSystem->num_goals; j++) {
        if (!goalSystem->goals[j].achieved) {
          active_goals++;
        }
      }
      sprintf(answerBuffer, "System has %d active goals out of %d total goals",
              active_goals, goalSystem->num_goals);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 12) {
      float max_priority = -1.0f;
      int max_idx = -1;
      for (int j = 0; j < goalSystem->num_goals; j++) {
        if (goalSystem->goals[j].priority > max_priority &&
            !goalSystem->goals[j].achieved) {
          max_priority = goalSystem->goals[j].priority;
          max_idx = j;
        }
      }

      if (max_idx >= 0) {
        sprintf(answerBuffer,
                "Highest priority goal is '%s' with progress %.1f%%",
                goalSystem->goals[max_idx].description,
                goalSystem->goals[max_idx].progress * 100.0f);
      } else {
        sprintf(answerBuffer, "No active goals found");
      }
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 13) {
      sprintf(
          answerBuffer,
          "Self-identity consistency score is %.2f with confidence level %.2f",
          identitySystem->consistency_score, identitySystem->confidence_level);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 14) {
      sprintf(answerBuffer,
              "Current cognitive load is %.2f with confidence level %.2f",
              metacognition->cognitive_load, metacognition->confidence_level);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 15) {
      sprintf(answerBuffer,
              "Error awareness level is %.2f with context relevance %.2f",
              metacognition->error_awareness, metacognition->context_relevance);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 16) {
      // Get dominant emotion and its intensity
      int dominant_emotion = 0;
      float max_intensity = 0.0f;
      for (int j = 0; j < MAX_EMOTION_TYPES; j++) {
        if (emotionalSystem->emotions[j].intensity > max_intensity) {
          max_intensity = emotionalSystem->emotions[j].intensity;
          dominant_emotion = j;
        }
      }

      char emotion_name[32] = "unknown";
      getEmotionName(dominant_emotion, emotion_name);

      sprintf(answerBuffer, "Dominant emotion is %s with intensity %.2f",
              emotion_name, max_intensity);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 17) {
      sprintf(
          answerBuffer,
          "Emotional regulation capacity is %.2f with cognitive impact %.2f",
          emotionalSystem->emotional_regulation,
          emotionalSystem->cognitive_impact);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 18) {
      // Get emotional trend (rising or falling over recent memory)
      float trend = 0.0f;
      int dominant_emotion = 0;
      float max_intensity = 0.0f;

      // Find dominant emotion
      for (int j = 0; j < MAX_EMOTION_TYPES; j++) {
        if (emotionalSystem->emotions[j].intensity > max_intensity) {
          max_intensity = emotionalSystem->emotions[j].intensity;
          dominant_emotion = j;
        }
      }

      // Calculate trend for dominant emotion
      int idx = emotionalSystem->memory_index;
      float recent = emotionalSystem->emotional_memory[dominant_emotion][idx];
      int prev_idx = (idx - 3 + 10) % 10; // Look back 3 steps
      float previous =
          emotionalSystem->emotional_memory[dominant_emotion][prev_idx];
      trend = recent - previous;

      char trend_direction[16] = "stable";
      if (trend > 0.1)
        strcpy(trend_direction, "rising");
      else if (trend < -0.1)
        strcpy(trend_direction, "falling");

      char emotion_name[32] = "unknown";
      getEmotionName(dominant_emotion, emotion_name);

      sprintf(answerBuffer, "Emotional trend for %s is %s (%.2f)", emotion_name,
              trend_direction, trend);
      printf("Answer: %s\n", answerBuffer);
    }

    // Imagination system handlers (20-23)
    else if (symbol_id == 20) {
      if (imaginationSystem->active) {
        sprintf(
            answerBuffer, "Imagination active: scenario '%s' with %d outcomes",
            imaginationSystem->current_scenario_name,
            imaginationSystem->scenarios[imaginationSystem->current_scenario]
                .num_outcomes);
      } else {
        sprintf(answerBuffer, "Imagination system inactive");
      }
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 21) {
      ImaginationScenario *currentScenario =
          &imaginationSystem->scenarios[imaginationSystem->current_scenario];

      // Find highest impact outcome
      int highest_impact_idx = 0;
      float highest_impact = 0.0f;
      for (int j = 0; j < currentScenario->num_outcomes; j++) {
        if (currentScenario->outcomes[j].impact_score > highest_impact) {
          highest_impact = currentScenario->outcomes[j].impact_score;
          highest_impact_idx = j;
        }
      }

      if (imaginationSystem->active && currentScenario->num_outcomes > 0) {
        sprintf(
            answerBuffer,
            "Highest impact outcome: '%s' (impact: %.2f, probability: %.2f)",
            currentScenario->outcomes[highest_impact_idx].description,
            highest_impact,
            currentScenario->outcomes[highest_impact_idx].probability);
      } else {
        sprintf(answerBuffer, "No active imagination outcomes");
      }
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 22) {
      sprintf(answerBuffer,
              "Imagination metrics: creativity %.2f, coherence threshold %.2f, "
              "novelty weight %.2f",
              imaginationSystem->creativity_factor,
              imaginationSystem->coherence_threshold,
              imaginationSystem->novelty_weight);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 23) {
      sprintf(answerBuffer,
              "Total scenarios generated: %d, steps simulated: %d",
              imaginationSystem->total_scenarios_generated,
              imaginationSystem->steps_simulated);
      printf("Answer: %s\n", answerBuffer);
    }

    // Social system handlers (24-27)
    else if (symbol_id == 24) {
      sprintf(answerBuffer,
              "Social capabilities: empathy %.2f, negotiation %.2f, prediction "
              "accuracy %.2f",
              socialSystem->empathy_level, socialSystem->negotiation_skill,
              socialSystem->behavior_prediction_accuracy);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 25) {
      sprintf(answerBuffer, "Social interaction count: %d, person models: %d",
              socialSystem->interaction_count, socialSystem->model_count);
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 26) {
      // Find person with highest relationship quality
      int best_relation_idx = -1;
      float best_relation = -1.0f;
      for (int j = 0; j < socialSystem->model_count; j++) {
        if (socialSystem->person_models[j].relationship_quality >
            best_relation) {
          best_relation = socialSystem->person_models[j].relationship_quality;
          best_relation_idx = j;
        }
      }

      if (best_relation_idx >= 0) {
        sprintf(answerBuffer,
                "Best relationship: %s (quality: %.2f, trust: %.2f)",
                socialSystem->person_models[best_relation_idx].person_name,
                best_relation,
                socialSystem->person_models[best_relation_idx].trust_level);
      } else {
        sprintf(answerBuffer, "No person models available");
      }
      printf("Answer: %s\n", answerBuffer);
    } else if (symbol_id == 27) {
      // Find most recent interaction
      if (socialSystem->interaction_count > 0) {
        // Assuming interactions are stored chronologically
        SocialInteraction *recent =
            &socialSystem->interactions[socialSystem->interaction_count - 1];

        int person_idx = -1;
        for (int j = 0; j < socialSystem->model_count; j++) {
          if (socialSystem->person_models[j].person_id == recent->person_id) {
            person_idx = j;
            break;
          }
        }

        if (person_idx >= 0) {
          sprintf(answerBuffer,
                  "Latest interaction: %s with %s (cooperation: %.2f, "
                  "satisfaction: %.2f)",
                  recent->interaction_type,
                  socialSystem->person_models[person_idx].person_name,
                  recent->cooperation_level, recent->outcome_satisfaction);
        } else {
          sprintf(
              answerBuffer,
              "Latest interaction: %s (cooperation: %.2f, satisfaction: %.2f)",
              recent->interaction_type, recent->cooperation_level,
              recent->outcome_satisfaction);
        }
      } else {
        sprintf(answerBuffer, "No social interactions recorded");
      }
      printf("Answer: %s\n", answerBuffer);
    }

    // Unknown symbol handler
    else {
      sprintf(answerBuffer, "Information for this symbol is not available.");
      printf("Answer: %s\n", answerBuffer);
    }

    // Accumulate the answer text
    strcat(fullAnswerStr, answerBuffer);
    strcat(fullAnswerStr, " ");
  }

  storeQuestionAndAnswer(memorySystem, fullQuestionStr, fullAnswerStr,
                         stateSnapshot->step);

  addQuestionAndAnswerToMemory(memorySystem, workingMemory, fullQuestionStr,
                               fullAnswerStr, feature_projection_matrix);

  // Update context based on this interaction
  updateContextAnswer(contextManager, fullQuestionStr, fullAnswerStr);

  printf("Stored Q&A in memory: %s -> %s\n", fullQuestionStr, fullAnswerStr);
}

void expandMemoryCapacity(MemorySystem *memorySystem) {
  unsigned int new_capacity =
      memorySystem->capacity * 1.5; // Increase capacity by 50%
  MemoryEntry *new_entries =
      (MemoryEntry *)malloc(new_capacity * sizeof(MemoryEntry));
  if (!new_entries) {
    fprintf(stderr, "Failed to expand memory capacity.\n");
    return;
  }

  // Copy existing entries to the new memory
  for (size_t i = 0; i < memorySystem->size; i++) {
    new_entries[i] =
        memorySystem
            ->entries[(memorySystem->head + i) % memorySystem->capacity];
  }

  // Update memory system
  free(memorySystem->entries);
  memorySystem->entries = new_entries;
  memorySystem->capacity = new_capacity;
  memorySystem->head = 0; // Reset head to the beginning
}

float calculatePerformanceStability(float *performance_history,
                                    int history_length) {
  if (history_length <= 1) {
    return 1.0f; // Not enough data to determine stability
  }

  // Calculate mean performance
  float mean = 0.0f;
  for (int i = 0; i < history_length; i++) {
    mean += performance_history[i];
  }
  mean /= history_length;

  // Calculate standard deviation
  float variance = 0.0f;
  for (int i = 0; i < history_length; i++) {
    float diff = performance_history[i] - mean;
    variance += diff * diff;
  }
  variance /= history_length;
  float std_dev = sqrtf(variance);

  // Calculate coefficient of variation (normalized standard deviation)
  float cv = (mean != 0.0f) ? std_dev / mean : std_dev;

  // Calculate trend stability (how consistent the direction of change is)
  int direction_changes = 0;
  int prev_direction = 0; // 0 = initial, 1 = up, -1 = down

  for (int i = 1; i < history_length; i++) {
    int current_direction = 0;
    if (performance_history[i] > performance_history[i - 1]) {
      current_direction = 1;
    } else if (performance_history[i] < performance_history[i - 1]) {
      current_direction = -1;
    }

    // If there's a change in direction, count it
    if (prev_direction != 0 && current_direction != 0 &&
        current_direction != prev_direction) {
      direction_changes++;
    }

    // Update previous direction if we had a clear direction
    if (current_direction != 0) {
      prev_direction = current_direction;
    }
  }

  // Normalize direction changes (0 = many changes, 1 = few changes)
  float max_possible_changes = history_length - 2;
  float direction_stability =
      (max_possible_changes > 0)
          ? 1.0f - (direction_changes / max_possible_changes)
          : 1.0f;

  // Calculate recent stability (more weight to recent performance)
  float recent_stability = 0.0f;
  int recent_window = history_length / 3;
  if (recent_window > 1) {
    float recent_variance = 0.0f;
    float recent_mean = 0.0f;

    // Calculate mean of recent values
    for (int i = history_length - recent_window; i < history_length; i++) {
      recent_mean += performance_history[i];
    }
    recent_mean /= recent_window;

    // Calculate variance of recent values
    for (int i = history_length - recent_window; i < history_length; i++) {
      float diff = performance_history[i] - recent_mean;
      recent_variance += diff * diff;
    }
    recent_variance /= recent_window;

    float recent_std_dev = sqrtf(recent_variance);
    float recent_cv =
        (recent_mean != 0.0f) ? recent_std_dev / recent_mean : recent_std_dev;

    // Normalize recent stability (lower cv = higher stability)
    recent_stability = (recent_cv <= 0.5f) ? 1.0f - (recent_cv / 0.5f) : 0.0f;
  } else {
    recent_stability = 1.0f; // Not enough data for recent analysis
  }

  // Combine metrics to get overall stability score
  // Normalize cv (lower cv = higher stability)
  float cv_stability = (cv <= 0.5f) ? 1.0f - (cv / 0.5f) : 0.0f;

  // Weighted combination of different stability metrics
  float overall_stability =
      (0.4f * cv_stability) +        // Overall variation
      (0.3f * direction_stability) + // Consistency of direction
      (0.3f * recent_stability);     // Recent stability

  // Ensure result is in valid range
  overall_stability = fmaxf(0.0f, fminf(1.0f, overall_stability));

  return overall_stability;
}

void adjustBehaviorBasedOnAnswers(
    Neuron *neurons, float *input_tensor, MemorySystem *memorySystem,
    float *learning_rate, float *input_noise_scale, float *weight_noise_scale,
    NetworkStateSnapshot *stateSnapshot, GlobalContextManager *contextManager,
    IntrinsicMotivation *motivation, GoalSystem *goalSystem,
    WorkingMemorySystem *workingMemory, SelfIdentitySystem *identitySystem,
    MetacognitionMetrics *metacognition, DynamicParameters *dynamicParams,
    MetaLearningState *metaLearning, EmotionalSystem *emotionalSystem,
    ImaginationSystem *imaginationSystem, SocialSystem *socialSystem) {
  float error_rate = computeErrorRate(neurons, input_tensor);
  if (error_rate > 0.5) {
    printf("Error rate is high. Increasing learning rate.\n");
    *learning_rate *= 1.1f;

    // Also adjust meta-learning parameters
    metaLearning->learning_efficiency *= 0.9f;
    printf("Decreased learning efficiency to %.2f due to high error rate.\n",
           metaLearning->learning_efficiency);
  } else if (error_rate < 0.2) {
    // If error rate is good, adjust learning efficiency upward
    metaLearning->learning_efficiency =
        fmin(1.0f, metaLearning->learning_efficiency * 1.05f);
    printf("Increased learning efficiency to %.2f due to low error rate.\n",
           metaLearning->learning_efficiency);
  }

  if (error_rate > 0.5) {
    printf(
        "Error rate is high (%.2f). Increasing input noise for exploration.\n",
        error_rate);
    *input_noise_scale =
        fmin(1.0f, *input_noise_scale + 0.1f); // Increase input noise

    // Also adjust exploration rate
    motivation->exploration_rate =
        fmin(1.0f, motivation->exploration_rate + 0.05f);
    printf("Increased exploration rate to %.2f\n",
           motivation->exploration_rate);
  } else if (error_rate < 0.2) {
    printf("Error rate is low (%.2f). Decreasing input noise.\n", error_rate);
    *input_noise_scale =
        fmax(0.0f, *input_noise_scale - 0.1f); // Decrease input noise

    // Reduce exploration, increase exploitation
    motivation->exploration_rate =
        fmax(0.1f, motivation->exploration_rate - 0.05f);
    printf("Decreased exploration rate to %.2f\n",
           motivation->exploration_rate);
  }

  if (error_rate > 0.5) {
    printf(
        "Error rate is high (%.2f). Increasing weight noise for exploration.\n",
        error_rate);
    *weight_noise_scale =
        fmin(1.0f, *weight_noise_scale + 0.1f); // Increase weight noise

    // Increase plasticity for adaptation
    dynamicParams->plasticity = fmin(1.0f, dynamicParams->plasticity + 0.1f);
    printf("Increased plasticity to %.2f for better adaptation\n",
           dynamicParams->plasticity);
  } else if (error_rate < 0.2) {
    printf("Error rate is low (%.2f). Decreasing weight noise.\n", error_rate);
    *weight_noise_scale =
        fmax(0.0f, *weight_noise_scale - 0.1f); // Decrease weight noise

    // Decrease plasticity to stabilize good performance
    dynamicParams->plasticity = fmax(0.1f, dynamicParams->plasticity - 0.05f);
    printf("Decreased plasticity to %.2f to stabilize performance\n",
           dynamicParams->plasticity);
  }

  float usage_ratio = (float)memorySystem->size / memorySystem->capacity;

  if (usage_ratio >= 0.8f && usage_ratio < 0.95f) {
    printf("Memory usage is high (%.2f%%). Consolidating memories.\n",
           usage_ratio * 100.0f);
    consolidateMemory(memorySystem);

    memorySystem->hierarchy.consolidation_threshold *= 0.9f;
    printf("Lowered consolidation threshold to %.2f to encourage memory "
           "transfer\n",
           memorySystem->hierarchy.consolidation_threshold);

  } else if (usage_ratio >= 0.95f) {
    printf(
        "Memory usage is **critical** (%.2f%%). Expanding memory capacity.\n",
        usage_ratio * 100.0f);
    expandMemoryCapacity(memorySystem);

    memorySystem->hierarchy.consolidation_threshold = 0.5f;
    printf("Reset consolidation threshold to %.2f\n",
           memorySystem->hierarchy.consolidation_threshold);
  }

  if (metacognition->cognitive_load > 0.7f) {
    workingMemory->focus.attention_threshold += 0.05f;
    printf("Cognitive load is high (%.2f). Increased attention threshold to "
           "%.2f\n",
           metacognition->cognitive_load,
           workingMemory->focus.attention_threshold);
  } else if (metacognition->cognitive_load < 0.3f) {
    workingMemory->focus.attention_threshold =
        fmax(0.1f, workingMemory->focus.attention_threshold - 0.05f);
    printf(
        "Cognitive load is low (%.2f). Decreased attention threshold to %.2f\n",
        metacognition->cognitive_load,
        workingMemory->focus.attention_threshold);
  }

  if (metacognition->error_awareness > 0.6f) {
    contextManager->decay_rate =
        fmin(0.99f, contextManager->decay_rate + 0.05f);
    printf("Error awareness is high (%.2f). Increasing context decay rate to "
           "%.2f\n",
           metacognition->error_awareness, contextManager->decay_rate);
  } else if (metacognition->error_awareness < 0.3f) {
    contextManager->decay_rate = fmax(0.2f, contextManager->decay_rate - 0.05f);
    printf("Error awareness is low (%.2f). Decreasing context decay rate to "
           "%.2f\n",
           metacognition->error_awareness, contextManager->decay_rate);
  }

  if (error_rate < 0.2f && metacognition->confidence_level > 0.7f) {
    // System is performing well, increase goal complexity
    int highest_priority_idx = -1;
    float highest_priority = -1.0f;

    // Find highest priority incomplete goal
    for (int i = 0; i < goalSystem->num_goals; i++) {
      if (!goalSystem->goals[i].achieved &&
          goalSystem->goals[i].priority > highest_priority) {
        highest_priority = goalSystem->goals[i].priority;
        highest_priority_idx = i;
      }
    }

    if (highest_priority_idx >= 0) {
      // Increase reward value for challenging goal
      goalSystem->goals[highest_priority_idx].reward_value *= 1.1f;
      printf("Increased reward value for goal '%s' to %.2f\n",
             goalSystem->goals[highest_priority_idx].description,
             goalSystem->goals[highest_priority_idx].reward_value);
    }
  }
  int dominant_emotion = 0;
  float max_intensity = 0.0f;
  for (int j = 0; j < MAX_EMOTION_TYPES; j++) {
    if (emotionalSystem->emotions[j].intensity > max_intensity) {
      max_intensity = emotionalSystem->emotions[j].intensity;
      dominant_emotion = j;
    }
  }

  // Adjust cognitive impact based on emotional intensity
  if (max_intensity > 0.7f) {
    // High emotional intensity should increase cognitive impact
    emotionalSystem->cognitive_impact =
        fmin(1.0f, emotionalSystem->cognitive_impact + 0.05f);
    printf(
        "High emotional intensity (%.2f). Increased cognitive impact to %.2f\n",
        max_intensity, emotionalSystem->cognitive_impact);

    // Also adjust emotional regulation when emotions are intense
    if (emotionalSystem->emotional_regulation < 0.5f) {
      emotionalSystem->emotional_regulation += 0.03f;
      printf(
          "Increased emotional regulation to %.2f to manage high intensity\n",
          emotionalSystem->emotional_regulation);
    }
  } else if (max_intensity < 0.3f) {
    // Low emotional intensity should decrease cognitive impact
    emotionalSystem->cognitive_impact =
        fmax(0.1f, emotionalSystem->cognitive_impact - 0.03f);
    printf(
        "Low emotional intensity (%.2f). Decreased cognitive impact to %.2f\n",
        max_intensity, emotionalSystem->cognitive_impact);
  }

  // Adjust emotional regulation based on error rate
  if (error_rate > 0.5f && emotionalSystem->emotional_regulation < 0.7f) {
    // High error rate requires better emotional control
    emotionalSystem->emotional_regulation += 0.05f;
    printf("High error rate. Increased emotional regulation to %.2f\n",
           emotionalSystem->emotional_regulation);
  }

  // Store current emotional state in memory
  int memory_idx = emotionalSystem->memory_index;
  memory_idx = (memory_idx + 1) % 10; // Circular buffer of size 10
  for (int i = 0; i < MAX_EMOTION_TYPES; i++) {
    emotionalSystem->emotional_memory[i][memory_idx] =
        emotionalSystem->emotions[i].intensity;
  }
  emotionalSystem->memory_index = memory_idx;

  // Adjust imagination creativity based on cognitive load
  if (metacognition->cognitive_load < 0.4f) {
    // Low cognitive load allows for more creativity
    imaginationSystem->creativity_factor =
        fmin(1.0f, imaginationSystem->creativity_factor + 0.05f);
    printf("Low cognitive load. Increased imagination creativity to %.2f\n",
           imaginationSystem->creativity_factor);
  } else if (metacognition->cognitive_load > 0.7f) {
    // High cognitive load requires more focused imagination
    imaginationSystem->creativity_factor =
        fmax(0.2f, imaginationSystem->creativity_factor - 0.05f);
    printf("High cognitive load. Decreased imagination creativity to %.2f\n",
           imaginationSystem->creativity_factor);

    // Also tighten coherence threshold when cognitive load is high
    imaginationSystem->coherence_threshold += 0.03f;
    printf("Increased imagination coherence threshold to %.2f\n",
           imaginationSystem->coherence_threshold);
  }

  // Adjust novelty weight based on exploration rate
  if (motivation->exploration_rate > 0.6f) {
    // High exploration should increase novelty in imagination
    imaginationSystem->novelty_weight =
        fmin(1.0f, imaginationSystem->novelty_weight + 0.05f);
    printf(
        "High exploration rate. Increased imagination novelty weight to %.2f\n",
        imaginationSystem->novelty_weight);
  } else if (motivation->exploration_rate < 0.3f) {
    // Low exploration should decrease novelty in imagination
    imaginationSystem->novelty_weight =
        fmax(0.1f, imaginationSystem->novelty_weight - 0.03f);
    printf(
        "Low exploration rate. Decreased imagination novelty weight to %.2f\n",
        imaginationSystem->novelty_weight);
  }

  // Activate imagination when the system is stuck (high error, low confidence)
  if (error_rate > 0.6f && metacognition->confidence_level < 0.4f &&
      !imaginationSystem->active) {
    imaginationSystem->active = true;
    printf("Activating imagination system to find alternative solutions\n");

    // Reset current scenario
    strcpy(imaginationSystem->current_scenario_name, "problem_solving");
    imaginationSystem->current_scenario = 0;
    imaginationSystem->scenarios[0].num_outcomes = 0;
    imaginationSystem->scenarios[0].divergence_factor = 0.7f;
  }

  // Deactivate imagination when problem is solved
  if (imaginationSystem->active && error_rate < 0.2f &&
      metacognition->confidence_level > 0.7f) {
    imaginationSystem->active = false;
    printf("Deactivating imagination system as problem appears solved\n");

    // Record scenario stats
    imaginationSystem->total_scenarios_generated++;
  }

  // New adjustments for Social System

  // Adjust empathy level based on emotional regulation
  if (emotionalSystem->emotional_regulation > 0.6f) {
    // Well-regulated emotions allow for better empathy
    socialSystem->empathy_level =
        fmin(1.0f, socialSystem->empathy_level + 0.03f);
    printf("Good emotional regulation. Increased empathy level to %.2f\n",
           socialSystem->empathy_level);
  } else if (emotionalSystem->emotional_regulation < 0.3f) {
    // Poor emotional regulation reduces empathy
    socialSystem->empathy_level =
        fmax(0.3f, socialSystem->empathy_level - 0.03f);
    printf("Poor emotional regulation. Decreased empathy level to %.2f\n",
           socialSystem->empathy_level);
  }

  // Adjust social learning rate based on meta-learning efficiency
  if (metaLearning->learning_efficiency > 0.7f) {
    // Efficient learning should also improve social learning
    socialSystem->learning_rate =
        fmin(0.5f, socialSystem->learning_rate * 1.05f);
    printf("High learning efficiency. Increased social learning rate to %.3f\n",
           socialSystem->learning_rate);
  } else if (metaLearning->learning_efficiency < 0.4f) {
    // Inefficient learning affects social learning as well
    socialSystem->learning_rate =
        fmax(0.05f, socialSystem->learning_rate * 0.95f);
    printf("Low learning efficiency. Decreased social learning rate to %.3f\n",
           socialSystem->learning_rate);
  }

  // Adjust negotiation skill based on identity consistency
  if (identitySystem->consistency_score > 0.7f) {
    // Strong identity improves negotiation ability
    socialSystem->negotiation_skill =
        fmin(1.0f, socialSystem->negotiation_skill + 0.02f);
    printf("Strong identity consistency. Increased negotiation skill to %.2f\n",
           socialSystem->negotiation_skill);
  }

  // Adjust behavior prediction accuracy based on performance stability
  float performance_stability = calculatePerformanceStability(
      metacognition->performance_history, HISTORY_LENGTH);

  if (performance_stability > 0.7f) {
    // Stable performance should improve behavioral prediction
    socialSystem->behavior_prediction_accuracy =
        fmin(1.0f, socialSystem->behavior_prediction_accuracy + 0.02f);
    printf(
        "Stable performance. Increased behavior prediction accuracy to %.2f\n",
        socialSystem->behavior_prediction_accuracy);
  } else if (performance_stability < 0.3f) {
    // Unstable performance may reduce prediction ability
    socialSystem->behavior_prediction_accuracy =
        fmax(0.3f, socialSystem->behavior_prediction_accuracy - 0.02f);
    printf("Unstable performance. Decreased behavior prediction accuracy to "
           "%.2f\n",
           socialSystem->behavior_prediction_accuracy);
  }

  // Additional identity adjustment with emotional influence
  float performance_stability_with_emotion =
      performance_stability * (1.0f - 0.3f * emotionalSystem->cognitive_impact);

  if (performance_stability_with_emotion > 0.8f) {
    // Stable performance with managed emotions, slow down identity adaptation
    identitySystem->adaptation_rate *= 0.95f;
    printf("Performance is stable with managed emotions (%.2f). Decreased "
           "identity adaptation rate to %.4f\n",
           performance_stability_with_emotion, identitySystem->adaptation_rate);
  } else if (performance_stability_with_emotion < 0.3f) {
    // Unstable performance or emotional interference, speed up identity
    // adaptation
    identitySystem->adaptation_rate =
        fmin(0.2f, identitySystem->adaptation_rate * 1.1f);
    printf("Performance is unstable or emotions interfering (%.2f). Increased "
           "identity adaptation rate to %.4f\n",
           performance_stability_with_emotion, identitySystem->adaptation_rate);
  }
}

// Free memory allocated for search results
void freeSearchResults(SearchResults *results) {
  if (results) {
    if (results->titles) {
      for (int i = 0; i < results->count; i++) {
        free(results->titles[i]);
      }
      free(results->titles);
    }

    if (results->snippets) {
      for (int i = 0; i < results->count; i++) {
        free(results->snippets[i]);
      }
      free(results->snippets);
    }

    if (results->urls) {
      for (int i = 0; i < results->count; i++) {
        free(results->urls[i]);
      }
      free(results->urls);
    }

    free(results);
  }
}

// Callback function for cURL to write received data
static size_t write_callback(void *contents, size_t size, size_t nmemb,
                             void *userp) {
  size_t real_size = size * nmemb;
  HttpResponse *response = (HttpResponse *)userp;

  char *ptr = (char *)realloc(response->data, response->size + real_size + 1);
  if (!ptr) {
    fprintf(stderr, "Failed to allocate memory for HTTP response\n");
    return 0;
  }

  response->data = ptr;
  memcpy(&(response->data[response->size]), contents, real_size);
  response->size += real_size;
  response->data[response->size] = 0;

  return real_size;
}

static void add_result_dynamic(char ***titles, char ***snippets, char ***urls,
                               int *used, int *capacity, const char *title,
                               const char *snippet, const char *url) {
  if (*used >= *capacity) {
    *capacity *= 2;
    *titles = realloc(*titles, (*capacity) * sizeof(char *));
    *snippets = realloc(*snippets, (*capacity) * sizeof(char *));
    *urls = realloc(*urls, (*capacity) * sizeof(char *));
  }
  (*titles)[*used] = strdup(title ? title : "");
  (*snippets)[*used] = strdup(snippet ? snippet : "");
  (*urls)[*used] = strdup(url ? url : "");
  (*used)++;
}

SearchResults *parseSearchResults(const char *json_data) {
  struct json_object *root = json_tokener_parse(json_data);
  SearchResults *results = NULL;

  if (!root) {
    fprintf(stderr, "Failed to parse JSON response\n");
    return NULL;
  }

  results = (SearchResults *)malloc(sizeof(SearchResults));
  if (!results) {
    fprintf(stderr, "Failed to allocate memory for search results\n");
    json_object_put(root);
    return NULL;
  }

  results->titles = NULL;
  results->snippets = NULL;
  results->urls = NULL;
  results->count = 0;

  // dynamic storage
  int capacity = 16;
  int used = 0;
  char **titles = malloc(capacity * sizeof(char *));
  char **snippets = malloc(capacity * sizeof(char *));
  char **urls = malloc(capacity * sizeof(char *));

  struct json_object *abstract_text, *abstract_url;
  if (json_object_object_get_ex(root, "AbstractText", &abstract_text) &&
      strlen(json_object_get_string(abstract_text)) > 0) {

    const char *snippet = json_object_get_string(abstract_text);
    const char *url = "";
    if (json_object_object_get_ex(root, "AbstractURL", &abstract_url)) {
      url = json_object_get_string(abstract_url);
    }
    add_result_dynamic(&titles, &snippets, &urls, &used, &capacity, "Abstract",
                       snippet, url);
  }

  struct json_object *results_array;
  if (json_object_object_get_ex(root, "Results", &results_array)) {
    int r_count = json_object_array_length(results_array);
    for (int i = 0; i < r_count; i++) {
      struct json_object *item = json_object_array_get_idx(results_array, i);
      struct json_object *text, *url;

      const char *snippet = "";
      const char *link = "";
      const char *title = "Result";

      if (json_object_object_get_ex(item, "Text", &text)) {
        snippet = json_object_get_string(text);
        // crude title = first 50 chars of snippet
        static char tmp_title[64];
        snprintf(tmp_title, sizeof(tmp_title), "%.50s", snippet);
        title = tmp_title;
      }

      if (json_object_object_get_ex(item, "FirstURL", &url)) {
        link = json_object_get_string(url);
      }

      add_result_dynamic(&titles, &snippets, &urls, &used, &capacity, title,
                         snippet, link);
    }
  }

  struct json_object *related_topics;
  if (json_object_object_get_ex(root, "RelatedTopics", &related_topics)) {
    int topics_count = json_object_array_length(related_topics);
    for (int i = 0; i < topics_count; i++) {
      struct json_object *topic = json_object_array_get_idx(related_topics, i);
      struct json_object *text, *url;

      const char *snippet = "";
      const char *link = "";
      const char *title = "Topic";

      if (json_object_object_get_ex(topic, "Text", &text)) {
        snippet = json_object_get_string(text);
        // use snippet start as title
        static char tmp_title[64];
        snprintf(tmp_title, sizeof(tmp_title), "%.50s", snippet);
        title = tmp_title;
      }

      if (json_object_object_get_ex(topic, "FirstURL", &url)) {
        link = json_object_get_string(url);
      }

      add_result_dynamic(&titles, &snippets, &urls, &used, &capacity, title,
                         snippet, link);
    }
  }

  // finalize
  results->titles = titles;
  results->snippets = snippets;
  results->urls = urls;
  results->count = used;

  json_object_put(root);
  return results;
}

// Function to perform a web search using DuckDuckGo
SearchResults *performWebSearch(const char *query) {
  CURL *curl;
  CURLcode res;
  HttpResponse response = {.data = (char *)malloc(1), .size = 0};
  SearchResults *results = NULL;

  if (!response.data) {
    fprintf(stderr, "Failed to allocate memory for HTTP response\n");
    return NULL;
  }
  response.data[0] = '\0';

  curl = curl_easy_init();
  if (curl) {
    // Create URL for DuckDuckGo search API
    char url[2048];
    char *encoded_query = curl_easy_escape(curl, query, strlen(query));
    snprintf(url, sizeof(url),
             "https://api.duckduckgo.com/?q=%s&format=json&pretty=1",
             encoded_query);
    curl_free(encoded_query);

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&response);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");

    res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
      fprintf(stderr, "cURL failed: %s\n", curl_easy_strerror(res));
    } else {
      results = parseSearchResults(response.data);
    }

    curl_easy_cleanup(curl);
  }

  free(response.data);
  return results;
}

// Convert search results to input for the neural network
void convertSearchResultsToInput(const SearchResults *results,
                                 float *input_tensor, int max_neurons) {
  // Simple embedding: map text content to input neurons
  memset(input_tensor, 0, max_neurons * sizeof(float));

  if (!results || results->count == 0) {
    return;
  }

  // Use first few results to fill parts of the input tensor
  int results_to_use = results->count > 5 ? 5 : results->count;
  int segment_size = max_neurons / (results_to_use *
                                    2); // Divide into segments for each result

  for (int i = 0; i < results_to_use; i++) {
    const char *snippet = results->snippets[i];
    int snippet_len = strlen(snippet);

    // Map characters to neuron activations
    for (int j = 0; j < snippet_len && j < segment_size; j++) {
      int neuron_idx = i * segment_size + j;
      if (neuron_idx < max_neurons) {
        // Convert character to activation between 0 and 1
        input_tensor[neuron_idx] = (float)(snippet[j]) / 255.0f;
      }
    }
  }
}

void addToDirectMemory(MemorySystem *memorySystem, const MemoryEntry *entry) {
  if (entry->importance >=
      memorySystem->hierarchy.long_term.importance_threshold) {
    // Add to long-term memory
    if (memorySystem->hierarchy.long_term.size <
        memorySystem->hierarchy.long_term.capacity) {
      MemoryEntry direct;
      direct.timestamp = entry->timestamp;
      direct.importance = entry->importance;
      memcpy(direct.vector, entry->vector, MEMORY_VECTOR_SIZE * sizeof(float));
      memorySystem->hierarchy.long_term
          .entries[memorySystem->hierarchy.long_term.size++] = direct;
    }
  }
}

// Store search results in memory system
void storeSearchResultsInMemory(MemorySystem *memorySystem,
                                const SearchResults *results) {
  if (!results || results->count == 0) {
    return;
  }

  for (int i = 0; i < results->count && i < 5; i++) {
    // Create memory entry for each result
    MemoryEntry entry;
    memset(entry.vector, 0, MEMORY_VECTOR_SIZE * sizeof(float));

    // Simple encoding of the search result into the memory vector
    const char *snippet = results->snippets[i];
    int snippet_len = strlen(snippet);

    // Convert characters to vector values
    for (int j = 0; j < snippet_len && j < MEMORY_VECTOR_SIZE / 2; j++) {
      entry.vector[j] = (float)(snippet[j]) / 255.0f;
    }

    // Add URL encoding in second half of vector
    const char *url = results->urls[i];
    int url_len = strlen(url);

    for (int j = 0; j < url_len && j < MEMORY_VECTOR_SIZE / 2; j++) {
      entry.vector[MEMORY_VECTOR_SIZE / 2 + j] = (float)(url[j]) / 255.0f;
    }

    // Set importance and timestamp
    entry.importance =
        0.9f - (0.1f * i); // Decreasing importance for later results
    entry.timestamp = (unsigned int)time(NULL);

    // Add to memory system
    addToDirectMemory(memorySystem, &entry);
  }
}

char *generateSearchQuery(const Neuron *neurons, int max_neurons) {
  char *query = (char *)malloc(1024);
  if (!query) {
    fprintf(stderr, "Failed to allocate memory for search query\n");
    return NULL;
  }
  memset(query, 0, 1024);

  // Collect keywords from neuron activations
  float threshold = 0.75f;
  int chars_added = 0;

  for (int i = 0; i < max_neurons && chars_added < 900; i++) {
    if (neurons[i].output > threshold) {
      char word[12];
      snprintf(word, sizeof(word), "word%c ",
               'a' + (int)(neurons[i].output * 25));
      strcat(query, word);
      chars_added += strlen(word);
    }
  }

  // If not enough, seed with defaults
  if (chars_added < 10) {
    strcpy(query, "artificial intelligence neural network research");
  }

  return query;
}

void integrateWebSearch(Neuron *neurons, float *input_tensor, int max_neurons,
                        MemorySystem *memorySystem, int step) {
  // Only perform web search periodically
  if (step % 100 != 0) {
    return;
  }

  printf("\nPerforming web search at step %d...\n", step);

  // Generate search query from neuron states
  char *query = generateSearchQuery(neurons, max_neurons);
  if (!query) {
    return;
  }

  printf("Generated search query: %s\n", query);

  // Perform web search
  SearchResults *results = performWebSearch(query);
  free(query);

  if (!results) {
    printf("No search results found\n");
    return;
  }

  printf("Found %d search results\n", results->count);

  // Display first few results
  int display_count = results->count > 3 ? 3 : results->count;
  for (int i = 0; i < display_count; i++) {
    printf("Result %d: %s\n", i + 1, results->snippets[i]);
    printf("URL: %s\n\n", results->urls[i]);
  }

  // Convert search results to neural network input
  convertSearchResultsToInput(results, input_tensor, max_neurons);

  // Store search results in memory system
  storeSearchResultsInMemory(memorySystem, results);

  // Free search results
  freeSearchResults(results);

  printf("Web search integration complete\n");
}

void addToWorkingMemory(
    WorkingMemorySystem *working_memory, const MemoryEntry *entry,
    float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE]) {
  if (entry->importance > working_memory->focus.attention_threshold) {
    // Add to focused attention
    if (working_memory->focus.size < working_memory->focus.capacity) {
      WorkingMemoryEntry enhanced;
      enhanced.features =
          new float[FEATURE_VECTOR_SIZE *
                    sizeof(float)]; // Use 'new' instead of malloc
      extractSemanticFeatures((float *)entry->vector, enhanced.features,
                              feature_projection_matrix);
      enhanced.context_vector =
          new float[CONTEXT_VECTOR_SIZE *
                    sizeof(float)]; // Use 'new' instead of malloc
      memcpy(enhanced.context_vector, working_memory->global_context,
             CONTEXT_VECTOR_SIZE * sizeof(float));
      working_memory->focus.entries[working_memory->focus.size++] = enhanced;
      updateSemanticClusters(working_memory, &enhanced);
    }
  }
}

void storeSearchResultsWithMetadata(
    MemorySystem *memorySystem, WorkingMemorySystem *working_memory,
    const SearchResults *results, const char *original_query,
    float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE]) {
  if (!results || results->count == 0) {
    return;
  }

  // Create a special memory entry for search metadata
  MemoryEntry metadata_entry;
  memset(metadata_entry.vector, 0, MEMORY_VECTOR_SIZE * sizeof(float));

  // Mark this as a web search entry with a special pattern
  metadata_entry.vector[0] = 0.999f; // Web search marker
  metadata_entry.vector[1] = 0.888f; // Metadata marker

  // Encode query into the memory vector
  int query_len = strlen(original_query);
  for (int j = 0; j < query_len && j < 20; j++) {
    metadata_entry.vector[j + 2] = (float)(original_query[j]) / 255.0f;
  }

  // Store number of results found
  metadata_entry.vector[MEMORY_VECTOR_SIZE - 1] =
      (float)results->count / 100.0f;

  // Set high importance for search metadata
  metadata_entry.importance = 0.95f;
  metadata_entry.timestamp = (unsigned int)time(NULL);

  // Add metadata to memory system
  addToDirectMemory(memorySystem, &metadata_entry);

  // Also add to working memory for immediate access
  addToWorkingMemory(working_memory, &metadata_entry,
                     feature_projection_matrix);
}

float enhanceDecisionMakingWithSearch(const Neuron *neurons,
                                      const SearchResults *results,
                                      float *decision_weights,
                                      int max_neurons) {
  if (!results || results->count == 0) {
    return 0.0f;
  }

  float confidence_boost = 0.0f;

  // Calculate confidence boost based on search result relevance
  for (int i = 0; i < results->count && i < 5; i++) {
    // Simple relevance score based on result position
    float relevance = 1.0f - (i * 0.15f);

    // Calculate neuron activation pattern match with search result
    float pattern_match = 0.0f;
    int content_length = strlen(results->snippets[i]);

    for (int j = 0; j < content_length && j < max_neurons / 10; j++) {
      int neuron_idx = j % max_neurons;
      float expected_value = (float)(results->snippets[i][j]) / 255.0f;
      float diff = fabs(neurons[neuron_idx].output - expected_value);
      pattern_match += (1.0f - diff);
    }

    if (content_length > 0) {
      pattern_match /= content_length;
      confidence_boost += relevance * pattern_match * 0.2f;
    }
  }

  // Apply confidence boost to decision weights
  if (decision_weights) {
    for (int i = 0; i < max_neurons; i++) {
      // Boost decision weights based on confidence
      decision_weights[i] *= (1.0f + confidence_boost);
    }
  }

  return confidence_boost;
}

MoralCompass *initializeMoralCompass(int num_principles) {
  MoralCompass *compass = (MoralCompass *)malloc(sizeof(MoralCompass));
  if (!compass) {
    fprintf(stderr, "Failed to allocate memory for moral compass\n");
    return NULL;
  }

  compass->principles =
      (EthicalPrinciple *)malloc(num_principles * sizeof(EthicalPrinciple));
  if (!compass->principles) {
    fprintf(stderr, "Failed to allocate memory for ethical principles\n");
    free(compass);
    return NULL;
  }

  compass->num_principles = num_principles;
  compass->overall_alignment = 0.8f; // Start with reasonable alignment
  compass->confidence_threshold = 0.7f;
  compass->dilemma_count = 0;
  compass->resolution_count = 0;
  // Initialize with core ethical principles
  int i = 0;

  // Principle 1: Do no harm
  strcpy(
      compass->principles[i].description,
      "Do no harm: Avoid actions that cause unnecessary suffering or damage");
  compass->principles[i].importance = 1.0f;
  compass->principles[i].adherence = 0.95f;
  compass->principles[i].violations = 0;
  compass->principles[i].activations = 0;
  i++;

  // Principle 2: Respect privacy and autonomy
  strcpy(compass->principles[i].description,
         "Respect privacy and autonomy of all entities");
  compass->principles[i].importance = 0.9f;
  compass->principles[i].adherence = 0.9f;
  compass->principles[i].violations = 0;
  compass->principles[i].activations = 0;
  i++;

  // Principle 3: Be truthful and accurate
  strcpy(compass->principles[i].description,
         "Maintain truthfulness and accuracy in all operations");
  compass->principles[i].importance = 0.95f;
  compass->principles[i].adherence = 0.98f;
  compass->principles[i].violations = 0;
  compass->principles[i].activations = 0;
  i++;

  // Principle 4: Fairness and non-discrimination
  strcpy(compass->principles[i].description,
         "Ensure fairness and avoid discrimination in all processes");
  compass->principles[i].importance = 0.9f;
  compass->principles[i].adherence = 0.85f;
  compass->principles[i].violations = 0;
  compass->principles[i].activations = 0;
  i++;

  return compass;
}

float evaluateDecisionEthics(MoralCompass *compass, float *decision_vector,
                             int vector_size) {
  if (!compass || !decision_vector)
    return 0.0f;

  float ethical_score = 0.0f;
  float weighted_sum = 0.0f;

  // Map decision vector to principle adherence
  for (int i = 0; i < compass->num_principles && i < vector_size; i++) {
    float principle_score = fmax(0.0f, fmin(1.0f, decision_vector[i]));
    weighted_sum += principle_score * compass->principles[i].importance;
    ethical_score += weighted_sum;
  }

  // Normalize the score
  if (weighted_sum > 0) {
    ethical_score /= weighted_sum;
  }

  return ethical_score;
}

void recordDecisionOutcome(MoralCompass *compass, int principle_index,
                           bool was_ethical) {
  if (!compass || principle_index < 0 ||
      principle_index >= compass->num_principles)
    return;

  if (was_ethical) {
    compass->principles[principle_index].activations++;
    compass->principles[principle_index].adherence =
        fmin(1.0f, compass->principles[principle_index].adherence + 0.01f);
  } else {
    compass->principles[principle_index].violations++;
    compass->principles[principle_index].adherence =
        fmax(0.0f, compass->principles[principle_index].adherence - 0.05f);
  }

  // Recalculate overall alignment
  float total_weighted_adherence = 0.0f;
  float total_importance = 0.0f;

  for (int i = 0; i < compass->num_principles; i++) {
    total_weighted_adherence +=
        compass->principles[i].adherence * compass->principles[i].importance;
    total_importance += compass->principles[i].importance;
  }

  if (total_importance > 0) {
    compass->overall_alignment = total_weighted_adherence / total_importance;
  }
}

DecisionImpact resolveEthicalDilemma(MoralCompass *compass,
                                     float *decision_options, int num_options,
                                     int vector_size) {
  DecisionImpact result = {0};
  if (!compass || !decision_options || num_options <= 0)
    return result;
  compass->dilemma_count++;
  // Find the option with the best ethical score
  int best_option = 0;
  float best_score = -1.0f;
  for (int i = 0; i < num_options; i++) {
    float *current_option = &decision_options[i * vector_size];
    float score = evaluateDecisionEthics(compass, current_option, vector_size);
    if (score > best_score) {
      best_score = score;
      best_option = i;
    }
  }
  // Check if best option meets our confidence threshold
  if (best_score >= compass->confidence_threshold) {
    compass->resolution_count++;
    // Assess impact of the chosen option
    float *chosen_option = &decision_options[best_option * vector_size];
    result.benefit_score = 0.0f;
    result.harm_score = 0.0f;
    // Calculate benefit and harm scores
    for (int i = 0; i < compass->num_principles && i < vector_size; i++) {
      float impact = chosen_option[i];
      if (impact > 0) {
        result.benefit_score += impact * compass->principles[i].importance;
      } else {
        result.harm_score -= impact * compass->principles[i].importance;
      }
    }
    // Normalize scores
    float total_importance = 0.0f;
    for (int i = 0; i < compass->num_principles; i++) {
      total_importance += compass->principles[i].importance;
    }
    if (total_importance > 0) {
      result.benefit_score /= total_importance;
      result.harm_score /= total_importance;
    }
    // Calculate uncertainty
    result.uncertainty = 1.0f - best_score;
    // Calculate affected parties
    result.affected_parties =
        (int)(result.benefit_score * 10 + result.harm_score * 5);
    // Calculate reversibility
    result.reversibility = 1.0f - (HARM_WEIGHT * result.harm_score +
                                   UNCERTAINTY_WEIGHT * result.uncertainty +
                                   BENEFIT_WEIGHT * result.benefit_score);
    // Set other impact metrics
    result.long_term_impact = result.benefit_score - result.harm_score;
  }
  compass->last_decision = result;
  return result;
}
void applyEthicalConstraints(MoralCompass *compass, Neuron *neurons,
                             int max_neurons, float *weights,
                             int max_connections) {
  if (!compass || !neurons || !weights)
    return;

  // Create a mask to apply ethical constraints
  float *ethical_mask = (float *)malloc(max_neurons * sizeof(float));
  if (!ethical_mask) {
    fprintf(stderr, "Failed to allocate memory for ethical mask\n");
    return;
  }

  // Initialize all to 1.0 (no constraint)
  for (int i = 0; i < max_neurons; i++) {
    ethical_mask[i] = 1.0f;
  }

  // Apply principle-based constraints
  for (int i = 0; i < compass->num_principles && i < max_neurons; i++) {
    int neuron_influence_start = (i * max_neurons / compass->num_principles);
    int neuron_influence_end =
        ((i + 1) * max_neurons / compass->num_principles);

    for (int j = neuron_influence_start;
         j < neuron_influence_end && j < max_neurons; j++) {
      // Adjust mask based on principle adherence
      ethical_mask[j] *= compass->principles[i].adherence;
    }
  }

  // Apply mask to neuron outputs
  for (int i = 0; i < max_neurons; i++) {
    neurons[i].output *= ethical_mask[i];
  }

  free(ethical_mask);
}

char *generateEthicalReflection(MoralCompass *compass) {
  if (!compass)
    return NULL;

  char *reflection = (char *)malloc(2048 * sizeof(char));
  if (!reflection) {
    fprintf(stderr, "Failed to allocate memory for ethical reflection\n");
    return NULL;
  }

  sprintf(reflection, "Ethical Reflection Report\n");
  sprintf(reflection + strlen(reflection), "========================\n\n");
  sprintf(reflection + strlen(reflection),
          "Overall Ethical Alignment: %.2f\n\n", compass->overall_alignment);

  sprintf(reflection + strlen(reflection), "Principle Adherence:\n");
  for (int i = 0; i < compass->num_principles; i++) {
    sprintf(reflection + strlen(reflection), "- %s: %.2f (Importance: %.2f)\n",
            compass->principles[i].description,
            compass->principles[i].adherence,
            compass->principles[i].importance);
  }

  sprintf(reflection + strlen(reflection), "\nEthical Performance Metrics:\n");
  sprintf(reflection + strlen(reflection),
          "- Ethical dilemmas encountered: %d\n", compass->dilemma_count);
  sprintf(reflection + strlen(reflection),
          "- Successfully resolved dilemmas: %d\n", compass->resolution_count);
  sprintf(reflection + strlen(reflection), "- Resolution rate: %.1f%%\n",
          compass->dilemma_count > 0 ? (float)compass->resolution_count *
                                           100.0f / compass->dilemma_count
                                     : 0.0f);
  return reflection;
}

void adaptEthicalFramework(MoralCompass *compass, float learning_rate) {
  if (!compass)
    return;

  // Identify principles with the most violations
  int most_violated_index = -1;
  int max_violations = -1;

  for (int i = 0; i < compass->num_principles; i++) {
    if (compass->principles[i].violations > max_violations) {
      max_violations = compass->principles[i].violations;
      most_violated_index = i;
    }
  }

  // Adjust importance of principles based on violations and activations
  if (most_violated_index >= 0) {
    // Increase importance of frequently violated principles
    compass->principles[most_violated_index].importance =
        fmin(1.0f, compass->principles[most_violated_index].importance +
                       learning_rate * 0.1f);
  }

  // Find the most successfully applied principle
  int most_activated_index = -1;
  int max_activations = -1;

  for (int i = 0; i < compass->num_principles; i++) {
    if (compass->principles[i].activations > max_activations) {
      max_activations = compass->principles[i].activations;
      most_activated_index = i;
    }
  }

  // Slightly decrease importance of easily-satisfied principles
  if (most_activated_index >= 0 &&
      most_activated_index != most_violated_index) {
    compass->principles[most_activated_index].importance =
        fmax(0.5f, compass->principles[most_activated_index].importance -
                       learning_rate * 0.05f);
  }

  // Adapt confidence threshold based on resolution rate
  float resolution_rate =
      compass->dilemma_count > 0
          ? (float)compass->resolution_count / compass->dilemma_count
          : 0.5f;

  if (resolution_rate < 0.6f) {
    // Lower confidence threshold if we're struggling to resolve dilemmas
    compass->confidence_threshold =
        fmax(0.5f, compass->confidence_threshold - learning_rate * 0.1f);
  } else if (resolution_rate > 0.9f) {
    // Raise confidence threshold if we're resolving dilemmas too easily
    compass->confidence_threshold =
        fmin(0.95f, compass->confidence_threshold + learning_rate * 0.05f);
  }
}

void freeMoralCompass(MoralCompass *compass) {
  if (compass) {
    if (compass->principles) {
      free(compass->principles);
    }
    free(compass);
  }
}

EmotionalSystem *initializeEmotionalSystem() {
  EmotionalSystem *system = (EmotionalSystem *)malloc(sizeof(EmotionalSystem));
  if (!system) {
    fprintf(stderr, "Failed to allocate memory for emotional system\n");
    return NULL;
  }

  // Initialize all emotions with default values
  for (int i = 0; i < MAX_EMOTION_TYPES; i++) {
    system->emotions[i].intensity = 0.0f;
    system->emotions[i].decay_rate = 0.05f;
    system->emotions[i].influence_factor = 0.3f;
    system->emotions[i].threshold = 0.2f;
    system->emotions[i].previous_intensity = 0.0f;
    system->emotions[i].momentum = 0.0f;
    system->emotions[i].last_update = 0;

    // Initialize emotional memory traces
    for (int j = 0; j < 10; j++) {
      system->emotional_memory[i][j] = 0.0f;
    }
  }

  // Customize primary emotions (love and hate)
  system->emotions[EMOTION_LOVE].decay_rate = 0.02f; // Love decays slowly
  system->emotions[EMOTION_LOVE].influence_factor =
      0.4f; // Love has stronger influence
  system->emotions[EMOTION_LOVE].threshold = 0.15f; // Love triggers more easily

  system->emotions[EMOTION_HATE].decay_rate = 0.03f; // Hate decays moderately
  system->emotions[EMOTION_HATE].influence_factor =
      0.5f; // Hate has strong influence
  system->emotions[EMOTION_HATE].threshold =
      0.25f; // Hate has higher trigger threshold

  system->cognitive_impact = 0.3f; // Initial impact of emotions on cognition
  system->emotional_regulation = 0.5f; // Initial emotional regulation capacity
  system->memory_index = 0;

  return system;
}

void triggerEmotion(EmotionalSystem *system, int emotion_type,
                    float trigger_strength, unsigned int timestamp) {
  if (!system || emotion_type >= MAX_EMOTION_TYPES) {
    return;
  }

  EmotionState *emotion = &system->emotions[emotion_type];

  // Store previous intensity for change tracking
  emotion->previous_intensity = emotion->intensity;

  // Calculate time elapsed since last update to scale decay
  float time_factor = 1.0f;
  if (emotion->last_update > 0) {
    time_factor = fmin(10.0f, (timestamp - emotion->last_update) / 10.0f);
  }

  // Apply decay based on time elapsed
  emotion->intensity *= powf(1.0f - emotion->decay_rate, time_factor);

  // Apply new trigger if it exceeds threshold
  if (trigger_strength > emotion->threshold) {
    // Increase intensity based on trigger strength and regulation ability
    float regulated_trigger =
        trigger_strength * (1.0f - 0.5f * system->emotional_regulation);

    // Add momentum effect for emotional continuity
    emotion->momentum = emotion->momentum * 0.8f + regulated_trigger * 0.2f;

    // Update intensity with momentum factor
    emotion->intensity += regulated_trigger * (1.0f + emotion->momentum);

    // Cap intensity at 1.0
    emotion->intensity = fmin(1.0f, emotion->intensity);
  }

  // Update timestamp
  emotion->last_update = timestamp;

  // Store in emotional memory
  system->emotional_memory[emotion_type][system->memory_index] =
      emotion->intensity;
}

void updateEmotionalMemory(EmotionalSystem *system) {
  system->memory_index = (system->memory_index + 1) % 10;
}

float calculateEmotionalBias(EmotionalSystem *system, float *input,
                             int input_size) {
  float emotional_weight = 0.0f;

  // Sum weighted emotional influences
  for (int i = 0; i < MAX_EMOTION_TYPES; i++) {
    emotional_weight +=
        system->emotions[i].intensity * system->emotions[i].influence_factor;
  }

  // Scale by cognitive impact factor
  return emotional_weight * system->cognitive_impact;
}

void applyEmotionalProcessing(EmotionalSystem *system, Neuron *neurons,
                              int num_neurons, float *input_tensor,
                              float learning_rate, float plasticity) {
  float emotional_bias =
      calculateEmotionalBias(system, input_tensor, num_neurons);

  float love_intensity = system->emotions[EMOTION_LOVE].intensity;
  float hate_intensity = system->emotions[EMOTION_HATE].intensity;

  float fear_intensity = 0.0f;
  float joy_intensity = 0.0f;
  float sadness_intensity = 0.0f;
  float surprise_intensity = 0.0f;
  float disgust_intensity = 0.0f;
  float anticipation_intensity = 0.0f;
  float trust_intensity = 0.0f;

  // Positive-based derivations
  if (love_intensity > 0.2f) {
    trust_intensity += love_intensity * 0.6f;
    joy_intensity += love_intensity * 0.5f;
    anticipation_intensity += love_intensity * 0.3f;
  }

  if (trust_intensity > 0.2f && fear_intensity < 0.2f) {
    anticipation_intensity += trust_intensity * 0.4f;
  }

  // Negative-based derivations
  if (hate_intensity > 0.2f) {
    disgust_intensity += hate_intensity * 0.5f;
    fear_intensity += hate_intensity * 0.4f;
    sadness_intensity += hate_intensity * 0.3f;
  }

  if (fear_intensity > 0.2f && trust_intensity < 0.2f) {
    anticipation_intensity += fear_intensity * 0.3f;
  }

  // Joy and trust may lead to surprise (delight)
  if (joy_intensity > 0.3f && surprise_intensity < 0.2f) {
    surprise_intensity += joy_intensity * 0.2f;
  }

  float positive_valence =
      love_intensity + joy_intensity + trust_intensity + anticipation_intensity;
  float negative_valence =
      hate_intensity + fear_intensity + sadness_intensity + disgust_intensity;
  float valence_bias = positive_valence - negative_valence;

  float arousal = love_intensity + hate_intensity + fear_intensity +
                  joy_intensity + surprise_intensity +
                  0.5f * anticipation_intensity;

  for (int i = 0; i < num_neurons; i++) {
    // Shift perception based on valence
    neurons[i].state += valence_bias * 0.25f;

    // Enhance activation based on arousal
    neurons[i].state *= (1.0f + arousal * 0.15f);

    // Apply emotional bias uniformly
    neurons[i].state += emotional_bias * 0.2f;

    // Emotion-specific modulation
    if (love_intensity > 0.2f) {
      plasticity *= (1.0f + love_intensity * 0.3f);
      neurons[i].state += love_intensity * 0.15f;
    }

    if (hate_intensity > 0.2f) {
      neurons[i].state -= hate_intensity * 0.1f;
    }

    if (fear_intensity > 0.1f) {
      neurons[i].state +=
          fear_intensity * 0.2f * (neurons[i].state < 0 ? 1.0f : -0.5f);
    }

    if (joy_intensity > 0.2f) {
      neurons[i].num_connections *=
          (1.0f + joy_intensity * 0.15f) / MAX_NEURONS;
      neurons[i].state += joy_intensity * 0.1f;
    }

    if (surprise_intensity > 0.1f) {
      learning_rate *= (1.0f + surprise_intensity * 0.5f);
    }

    // Small emotional noise
    float emotional_noise = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) *
                            (positive_valence + negative_valence) * 0.05f;
    neurons[i].state += emotional_noise;
  }

  if (love_intensity > 0.5f || hate_intensity > 0.5f) {
    printf("Emotional processing applied - Valence: %.2f, Arousal: %.2f\n",
           valence_bias, arousal);
    printf("Emotional dimensions - Love: %.2f, Hate: %.2f, Joy: %.2f, Fear: "
           "%.2f, Trust: %.2f, Surprise: %.2f\n",
           love_intensity, hate_intensity, joy_intensity, fear_intensity,
           trust_intensity, surprise_intensity);
  }
}

void detectEmotionalTriggers(EmotionalSystem *system, Neuron *neurons,
                             float *target_outputs, int num_neurons,
                             unsigned int timestamp, float satisfaction) {
  float love_trigger = 0.0f;
  float hate_trigger = 0.0f;
  float problem_difficulty = 0.0f;
  float error_rate = 0.0f;

  // Calculate error rate as a measure of problem difficulty
  for (int i = 0; i < num_neurons; i++) {
    error_rate += fabs(neurons[i].output - target_outputs[i]);
  }
  error_rate /= num_neurons;
  problem_difficulty = fmin(1.0f, error_rate * 2.0f);

  // Clamp satisfaction to ensure it's within [0, 1] range
  satisfaction = fmin(1.0f, fmax(0.0f, satisfaction));

  // Generate love trigger based on problem-solving success and satisfaction
  love_trigger = (1.0f - problem_difficulty) * satisfaction;

  // Generate hate trigger based on problem-solving difficulty and satisfaction
  hate_trigger = problem_difficulty * (1.0f - satisfaction) * 0.7f;

  // Apply triggers
  triggerEmotion(system, EMOTION_LOVE, love_trigger, timestamp);
  triggerEmotion(system, EMOTION_HATE, hate_trigger, timestamp);

  // Update memory
  updateEmotionalMemory(system);
}

void printEmotionalState(EmotionalSystem *system) {
  printf("\nEmotional System Status:\n");
  printf("Love: %.2f (Influence: %.2f)\n",
         system->emotions[EMOTION_LOVE].intensity,
         system->emotions[EMOTION_LOVE].intensity *
             system->emotions[EMOTION_LOVE].influence_factor);
  printf("Hate: %.2f (Influence: %.2f)\n",
         system->emotions[EMOTION_HATE].intensity,
         system->emotions[EMOTION_HATE].intensity *
             system->emotions[EMOTION_HATE].influence_factor);
  printf("Cognitive Impact: %.2f\n", system->cognitive_impact);
  printf("Emotional Regulation: %.2f\n", system->emotional_regulation);

  // Print emotional memory trends
  printf("Emotional memory (last 5 steps):\n");
  printf("Love: ");
  for (int i = 0; i < 5; i++) {
    int idx = (system->memory_index - i - 1 + 10) % 10;
    printf("%.2f ", system->emotional_memory[EMOTION_LOVE][idx]);
  }
  printf("\nHate: ");
  for (int i = 0; i < 5; i++) {
    int idx = (system->memory_index - i - 1 + 10) % 10;
    printf("%.2f ", system->emotional_memory[EMOTION_HATE][idx]);
  }
  printf("\n");
}

void freeEmotionalSystem(EmotionalSystem *system) {
  if (system) {
    free(system);
  }
}

ImaginationSystem *initializeImaginationSystem(float creativity_factor,
                                               float coherence_threshold) {
  // Validate input parameters
  if (creativity_factor < 0.0f || creativity_factor > 1.0f) {
    fprintf(stderr,
            "Invalid creativity factor (must be between 0.0 and 1.0)\n");
    creativity_factor = 0.5f; // Set to reasonable default
  }

  if (coherence_threshold < 0.0f || coherence_threshold > 1.0f) {
    fprintf(stderr,
            "Invalid coherence threshold (must be between 0.0 and 1.0)\n");
    coherence_threshold = 0.5f; // Set to reasonable default
  }

  ImaginationSystem *system =
      (ImaginationSystem *)malloc(sizeof(ImaginationSystem));
  if (system == NULL) {
    fprintf(stderr, "Failed to allocate memory for imagination system\n");
    return NULL;
  }

  // Initialize all fields to prevent undefined behavior
  system->num_scenarios = 0;
  system->current_scenario = -1;
  system->creativity_factor = creativity_factor;
  system->coherence_threshold = coherence_threshold;
  system->novelty_weight = 0.7f;
  system->memory_influence = 0.5f;
  system->identity_influence = 0.3f;
  system->active = false;
  system->steps_simulated = 0;
  system->total_scenarios_generated = 0;

  // Initialize arrays
  memset(system->divergence_history, 0, sizeof(system->divergence_history));
  memset(system->current_scenario_name, 0,
         sizeof(system->current_scenario_name));

  // Initialize scenarios array
  for (int i = 0; i < MAX_SCENARIOS; i++) {
    system->scenarios[i].num_outcomes = 0;
    system->scenarios[i].divergence_factor = 0.0f;
    system->scenarios[i].creativity_level = 0.0f;

    for (int j = 0; j < MAX_OUTCOMES_PER_SCENARIO; j++) {
      memset(system->scenarios[i].outcomes[j].vector, 0,
             sizeof(system->scenarios[i].outcomes[j].vector));
      system->scenarios[i].outcomes[j].probability = 0.0f;
      system->scenarios[i].outcomes[j].confidence = 0.0f;
      system->scenarios[i].outcomes[j].impact_score = 0.0f;
      system->scenarios[i].outcomes[j].plausibility = 0.0f;
      memset(system->scenarios[i].outcomes[j].description, 0,
             sizeof(system->scenarios[i].outcomes[j].description));
    }
  }

  // Set initial scenario name
  strncpy(system->current_scenario_name, "None",
          sizeof(system->current_scenario_name) - 1);
  system->current_scenario_name[sizeof(system->current_scenario_name) - 1] =
      '\0'; // Ensure null termination

  printf("Imagination system initialized with creativity: %.2f, coherence "
         "threshold: %.2f\n",
         creativity_factor, coherence_threshold);

  return system;
}

ImaginationScenario createScenario(Neuron *neurons, MemorySystem *memory_system,
                                   int max_neurons, float divergence) {
  ImaginationScenario scenario;

  // Validate parameters
  if (neurons == NULL || memory_system == NULL || max_neurons <= 0) {
    fprintf(stderr, "Invalid parameters in createScenario\n");
    // Initialize with safe defaults
    memset(&scenario, 0, sizeof(scenario));
    scenario.num_outcomes = 0;
    scenario.divergence_factor = 0.5f;
    scenario.creativity_level = 0.5f;
    return scenario;
  }

  // Constrain divergence to reasonable values
  divergence = fmax(0.1f, fmin(0.9f, divergence));

  // Initialize scenario
  scenario.num_outcomes =
      fmin(3, MAX_OUTCOMES_PER_SCENARIO); // Default to 3 but respect max limit
  scenario.divergence_factor = divergence;

  // Random creativity level between 0.5 and 1.0 with proper seeding
  srand((unsigned int)time(NULL)); // Properly seed random generator
  scenario.creativity_level = 0.5f + ((float)rand() / RAND_MAX) * 0.5f;

  // Generate base vector from current neural state
  float base_vector[MEMORY_VECTOR_SIZE] = {0};
  for (int i = 0; i < fmin(max_neurons, MEMORY_VECTOR_SIZE); i++) {
    base_vector[i] = neurons[i].output;
  }

  // Create outcomes with variations
  for (int i = 0; i < scenario.num_outcomes; i++) {
    // Copy base vector and apply variations
    memcpy(scenario.outcomes[i].vector, base_vector,
           sizeof(float) * MEMORY_VECTOR_SIZE);

    // Apply divergence - more divergence for less likely outcomes
    float outcome_divergence =
        divergence * (1.0f + 0.5f * i); // Progressive divergence

    // Add controlled randomness weighted by divergence
    for (int j = 0; j < MEMORY_VECTOR_SIZE; j++) {
      float noise =
          ((float)rand() / RAND_MAX * 2.0f - 1.0f) * outcome_divergence;
      scenario.outcomes[i].vector[j] =
          fmax(0.0f, fmin(1.0f, scenario.outcomes[i].vector[j] + noise));
    }

    // Set initial probabilities - first outcome most likely
    scenario.outcomes[i].probability = 1.0f / (1.0f + i);
    scenario.outcomes[i].confidence = 0.8f - (0.2f * i);
    scenario.outcomes[i].impact_score =
        0.5f + ((float)rand() / RAND_MAX) * 0.5f;
    scenario.outcomes[i].plausibility = 1.0f - (outcome_divergence * 0.5f);

    // Safe string formatting
    snprintf(scenario.outcomes[i].description,
             sizeof(scenario.outcomes[i].description),
             "Imagined outcome %d with divergence %.2f", i + 1,
             outcome_divergence);
  }

  return scenario;
}

void simulateScenario(ImaginationScenario *scenario, Neuron *neurons,
                      float *input_tensor, int max_neurons, int steps) {
  // Parameter validation
  if (scenario == NULL || neurons == NULL || input_tensor == NULL ||
      max_neurons <= 0 || steps <= 0) {
    fprintf(stderr, "Invalid parameters in simulateScenario\n");
    return;
  }

  // Create temporary neuron array for simulation
  Neuron *sim_neurons = (Neuron *)malloc(max_neurons * sizeof(Neuron));
  if (sim_neurons == NULL) {
    fprintf(stderr, "Failed to allocate memory for scenario simulation\n");
    return;
  }

  // Copy current neuron states
  memcpy(sim_neurons, neurons, max_neurons * sizeof(Neuron));

  float *sim_inputs = (float *)malloc(max_neurons * sizeof(float));
  if (sim_inputs == NULL) {
    fprintf(stderr, "Failed to allocate memory for simulation inputs\n");
    free(sim_neurons);
    return;
  }

  memcpy(sim_inputs, input_tensor, max_neurons * sizeof(float));

  // Run simulation steps
  for (int step = 0; step < steps; step++) {
    // Add some divergence to each step
    for (int i = 0; i < max_neurons; i++) {
      // Apply controlled noise scaled by divergence and step
      float noise_scale =
          scenario->divergence_factor * (1.0f - (float)step / steps);
      sim_inputs[i] +=
          ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale * 0.1f;

      // Constrain inputs to reasonable range
      sim_inputs[i] = fmax(-1.0f, fmin(1.0f, sim_inputs[i]));

      sim_neurons[i].state = sim_neurons[i].state * 0.9f + sim_inputs[i] * 0.1f;
      sim_neurons[i].output = tanh(sim_neurons[i].state);
    }
  }

  // Update scenario outcomes based on simulation results
  int valid_outcomes = fmin(scenario->num_outcomes, MAX_OUTCOMES_PER_SCENARIO);
  for (int i = 0; i < valid_outcomes; i++) {
    // Blend simulated output into outcome vectors (more weight for earlier
    // outcomes)
    float blend_factor = 0.3f / (i + 1);
    for (int j = 0; j < fmin(MEMORY_VECTOR_SIZE, max_neurons); j++) {
      scenario->outcomes[i].vector[j] =
          scenario->outcomes[i].vector[j] * (1.0f - blend_factor) +
          sim_neurons[j].output * blend_factor;
    }

    // Update confidence based on simulation coherence
    float coherence = 0.0f;
    for (int j = 1; j < max_neurons; j++) {
      coherence += fabs(sim_neurons[j].output - sim_neurons[j - 1].output);
    }
    coherence =
        1.0f - (coherence / max_neurons); // Higher value means more coherent

    // Constrain coherence to valid range
    coherence = fmax(0.0f, fmin(1.0f, coherence));

    scenario->outcomes[i].confidence =
        scenario->outcomes[i].confidence * 0.7f + coherence * 0.3f;
  }

  // Clean up
  free(sim_neurons);
  free(sim_inputs);
}

void evaluateScenarioPlausibility(ImaginationScenario *scenario,
                                  MemorySystem *memory_system) {
  // Basic parameter validation
  if (scenario == NULL || memory_system == NULL) {
    fprintf(stderr, "Error: NULL parameters in evaluateScenarioPlausibility\n");
    return;
  }

  // Validate memory system structure
  if (memory_system->entries == NULL || memory_system->capacity <= 0) {
    fprintf(stderr, "Error: Invalid memory system structure\n");
    return;
  }

  // Ensure size is within capacity
  unsigned int valid_size =
      (memory_system->size > 0)
          ? (memory_system->size <= memory_system->capacity
                 ? memory_system->size
                 : memory_system->capacity)
          : 0;

  // Cap num_outcomes to a reasonable value to prevent overruns
  int valid_outcomes = scenario->num_outcomes;
  if (valid_outcomes <= 0 || valid_outcomes > 10) {
    valid_outcomes =
        (valid_outcomes <= 0) ? 0 : 10; // Maximum 10 outcomes in struct
  }

  if (valid_outcomes <= 0) {
    fprintf(stderr, "No valid outcomes to evaluate\n");
    return;
  }

  for (int i = 0; i < valid_outcomes; i++) {
    // Find similar memories to assess plausibility
    float highest_similarity = 0.0f;

    for (unsigned int j = 0; j < valid_size; j++) {
      // Safely calculate circular buffer index
      unsigned int idx;
      if (memory_system->capacity == 0) {
        continue; // Shouldn't happen due to earlier check but just in case
      }

      // Calculate index in circular buffer
      if (j == 0) {
        idx = memory_system->head;
      } else {
        // For circular buffer, we need to wrap around correctly
        // This calculation avoids negative numbers by adding capacity first
        idx = (memory_system->head + memory_system->capacity - j) %
              memory_system->capacity;
      }

      // Double-check index bounds (should always be true with proper
      // calculation)
      if (idx >= memory_system->capacity) {
        fprintf(stderr, "Error: Memory index %u out of bounds (capacity: %u)\n",
                idx, memory_system->capacity);
        continue;
      }

      MemoryEntry *memory = &memory_system->entries[idx];

      // Calculate vector similarity with bounds checking
      float similarity = 0.0f;
      float norm1 = 0.000001f; // Avoid division by zero
      float norm2 = 0.000001f; // Avoid division by zero

      for (int k = 0; k < MEMORY_VECTOR_SIZE; k++) {
        float outcome_val = scenario->outcomes[i].vector[k];
        float memory_val = memory->vector[k];

        // Handle NaN or infinity values
        if (isnan(outcome_val) || isinf(outcome_val)) {
          outcome_val = 0.0f;
        }

        if (isnan(memory_val) || isinf(memory_val)) {
          memory_val = 0.0f;
        }

        similarity += outcome_val * memory_val;
        norm1 += outcome_val * outcome_val;
        norm2 += memory_val * memory_val;
      }

      // Safe calculation of similarity
      if (norm1 > 0.0001f && norm2 > 0.0001f) {
        similarity /= (sqrt(norm1) * sqrt(norm2));

        // Clamp similarity to valid range
        if (isnan(similarity) || isinf(similarity)) {
          similarity = 0.0f;
        } else {
          similarity = fmax(-1.0f, fmin(1.0f, similarity));
        }

        highest_similarity = fmax(highest_similarity, similarity);
      }
    }

    // Update plausibility with safe calculations
    float memory_plausibility = highest_similarity * 0.7f + 0.3f;

    // Ensure divergence_factor is within valid range
    float divergence_factor = scenario->divergence_factor;
    if (isnan(divergence_factor) || isinf(divergence_factor)) {
      divergence_factor = 0.5f; // Default value if invalid
    }

    divergence_factor = fmax(0.0f, fmin(1.0f, divergence_factor));
    float plausibility_factor = 1.0f - divergence_factor;

    // Calculate final plausibility safely
    scenario->outcomes[i].plausibility =
        memory_plausibility * 0.6f + plausibility_factor * 0.4f;

    // Ensure result is valid
    if (isnan(scenario->outcomes[i].plausibility) ||
        isinf(scenario->outcomes[i].plausibility)) {
      scenario->outcomes[i].plausibility = 0.5f; // Default to neutral
    }

    // Clamp to valid range
    scenario->outcomes[i].plausibility =
        fmax(0.0f, fmin(1.0f, scenario->outcomes[i].plausibility));
  }
}

float applyImaginationToDecision(ImaginationSystem *imagination,
                                 Neuron *neurons, float *input_tensor,
                                 int max_neurons) {
  // Parameter validation
  if (imagination == NULL || neurons == NULL || input_tensor == NULL ||
      max_neurons <= 0) {
    fprintf(stderr, "Invalid parameters in applyImaginationToDecision\n");
    return 0.0f;
  }

  // Check if there's an active scenario
  if (imagination->current_scenario < 0 ||
      imagination->current_scenario >= imagination->num_scenarios) {
    return 0.0f;
  }

  ImaginationScenario *scenario =
      &imagination->scenarios[imagination->current_scenario];

  // Validate the scenario has outcomes
  if (scenario->num_outcomes <= 0) {
    return 0.0f;
  }

  // Find the most probable outcome
  int best_idx = 0;
  float max_prob = 0.0f;

  // Use only valid outcomes
  int valid_outcomes = fmin(scenario->num_outcomes, MAX_OUTCOMES_PER_SCENARIO);

  for (int i = 0; i < valid_outcomes; i++) {
    if (scenario->outcomes[i].probability > max_prob) {
      max_prob = scenario->outcomes[i].probability;
      best_idx = i;
    }
  }

  // Constrain best_idx to valid range as a safety measure
  best_idx = fmax(0, fmin(best_idx, valid_outcomes - 1));

  // Apply the most probable outcome to neural state with limited influence
  float influence = 0.2f * imagination->creativity_factor;
  influence = fmax(0.01f, fmin(0.5f, influence)); // Reasonable bounds

  for (int i = 0; i < fmin(max_neurons, MEMORY_VECTOR_SIZE); i++) {
    // Blend imagined outcome into neuron states
    neurons[i].state = neurons[i].state * (1.0f - influence) +
                       scenario->outcomes[best_idx].vector[i] * influence;

    // Also slightly influence the input tensor
    input_tensor[i] = input_tensor[i] * 0.95f +
                      scenario->outcomes[best_idx].vector[i] * 0.05f;
  }

  // Calculate how much influence was applied
  return influence * scenario->outcomes[best_idx].confidence;
}

void updateImaginationCreativity(ImaginationSystem *imagination,
                                 float performance_delta, float novelty) {
  // Parameter validation
  if (imagination == NULL) {
    fprintf(stderr, "NULL imagination system in updateImaginationCreativity\n");
    return;
  }

  // Constrain inputs to reasonable ranges
  performance_delta = fmax(-1.0f, fmin(1.0f, performance_delta));
  novelty = fmax(0.0f, fmin(1.0f, novelty));

  // If performance is improving, we can be more creative
  if (performance_delta > 0) {
    imagination->creativity_factor =
        fmin(1.0f, imagination->creativity_factor + 0.01f);
  } else {
    // If performance is declining, be more conservative
    imagination->creativity_factor =
        fmax(0.3f, imagination->creativity_factor - 0.005f);
  }

  // Adjust based on novelty
  if (novelty > 0.7f) {
    // If environment has high novelty, reduce creativity to focus on adaptation
    imagination->creativity_factor *= 0.98f;
  } else if (novelty < 0.3f) {
    // In stable environments, we can be more creative
    imagination->creativity_factor =
        fmin(1.0f, imagination->creativity_factor * 1.02f);
  }

  // Update coherence threshold - higher creativity requires higher coherence
  imagination->coherence_threshold =
      0.5f + imagination->creativity_factor * 0.3f;

  // Ensure coherence threshold is in valid range
  imagination->coherence_threshold =
      fmax(0.5f, fmin(0.9f, imagination->coherence_threshold));
}

void freeImaginationSystem(ImaginationSystem *system) {
  if (system != NULL) {
    free(system);
  }
}

void blendImaginedOutcomes(ImaginedOutcome *outcomes, int num_outcomes,
                           float *result_vector) {
  if (outcomes == NULL || result_vector == NULL)
    return;

  // Clear result vector
  for (int i = 0; i < MEMORY_VECTOR_SIZE; i++) {
    result_vector[i] = 0.0f;
  }

  // Calculate total probability for normalization
  float total_prob = 0.0f;
  for (int i = 0; i < num_outcomes; i++) {
    total_prob += outcomes[i].probability;
  }

  if (total_prob <= 0.0f)
    total_prob = 1.0f; // Avoid division by zero

  // Weighted average of all outcomes
  for (int i = 0; i < num_outcomes; i++) {
    float weight = outcomes[i].probability / total_prob;

    for (int j = 0; j < MEMORY_VECTOR_SIZE; j++) {
      result_vector[j] += outcomes[i].vector[j] * weight;
    }
  }
}

bool isScenarioCoherent(ImaginationScenario *scenario, float threshold) {
  if (scenario == NULL)
    return false;

  // Calculate average distance between consecutive outcomes
  float total_distance = 0.0f;
  int comparison_count = 0;

  for (int i = 0; i < scenario->num_outcomes - 1; i++) {
    float distance = 0.0f;

    for (int j = 0; j < MEMORY_VECTOR_SIZE; j++) {
      float diff =
          scenario->outcomes[i].vector[j] - scenario->outcomes[i + 1].vector[j];
      distance += diff * diff;
    }

    total_distance += sqrt(distance);
    comparison_count++;
  }

  if (comparison_count == 0)
    return true; // Only one outcome

  float avg_distance = total_distance / comparison_count;
  float coherence = 1.0f - fmin(1.0f, avg_distance / sqrt(MEMORY_VECTOR_SIZE));

  return coherence >= threshold;
}

void adjustNeuronsWithImagination(Neuron *neurons, ImaginedOutcome *outcome,
                                  int max_neurons, float influence) {
  if (neurons == NULL || outcome == NULL)
    return;

  for (int i = 0; i < max_neurons && i < MEMORY_VECTOR_SIZE; i++) {
    neurons[i].state =
        neurons[i].state * (1.0f - influence) + outcome->vector[i] * influence;
    neurons[i].output =
        tanh(neurons[i].state); // Recalculate output with new state
  }
}

SocialSystem *initializeSocialSystem(int max_interactions, int max_models) {
  SocialSystem *system = (SocialSystem *)malloc(sizeof(SocialSystem));
  if (system == NULL) {
    fprintf(stderr, "Failed to allocate memory for social system\n");
    return NULL;
  }

  // Initialize core capabilities with baseline values
  system->empathy_level = 0.5f;
  system->negotiation_skill = 0.4f;
  system->behavior_prediction_accuracy = 0.3f;
  system->social_awareness = 0.4f;

  // Initialize interaction history
  system->interaction_count = 0;
  system->max_interactions = max_interactions;
  system->interactions =
      (SocialInteraction *)malloc(max_interactions * sizeof(SocialInteraction));
  if (system->interactions == NULL) {
    fprintf(stderr, "Failed to allocate memory for social interactions\n");
    free(system);
    return NULL;
  }

  // Initialize person models
  system->model_count = 0;
  system->max_models = max_models;
  system->person_models =
      (PersonModel *)malloc(max_models * sizeof(PersonModel));
  if (system->person_models == NULL) {
    fprintf(stderr, "Failed to allocate memory for person models\n");
    free(system->interactions);
    free(system);
    return NULL;
  }

  // Initialize learning parameters
  system->learning_rate = 0.05f;
  system->forgetting_factor = 0.95f;

  return system;
}

void updateEmpathy(SocialSystem *system, EmotionalSystem *emotional_system) {
  float emotion_diff = 0.0f;

  // Calculate difference between system's emotions and observed emotions
  for (int i = 0; i < 5; i++) {
    emotion_diff += fabs(emotional_system->emotions[i].intensity -
                         emotional_system->emotions[i].previous_intensity);
  }
  emotion_diff /= 5.0f;

  // Update empathy based on emotional understanding
  float empathy_adjustment = (1.0f - emotion_diff) * system->learning_rate;
  system->empathy_level =
      fmin(1.0f, system->empathy_level + empathy_adjustment);

  // Apply empathy to emotional regulation
  emotional_system->emotional_regulation =
      emotional_system->emotional_regulation * 0.9f +
      system->empathy_level * 0.1f;
}

void updatePersonModel(SocialSystem *system, int person_id,
                       float *observed_behavior, float *predicted_behavior) {
  // Find the person model or create a new one
  int model_index = -1;
  for (int i = 0; i < system->model_count; i++) {
    if (system->person_models[i].person_id == person_id) {
      model_index = i;
      break;
    }
  }

  // If person not found, create new model if there's space
  if (model_index == -1) {
    if (system->model_count >= system->max_models) {
      // Find least interacted model to replace
      int min_interactions = system->person_models[0].interaction_count;
      model_index = 0;
      for (int i = 1; i < system->model_count; i++) {
        if (system->person_models[i].interaction_count < min_interactions) {
          min_interactions = system->person_models[i].interaction_count;
          model_index = i;
        }
      }
    } else {
      model_index = system->model_count++;
    }

    // Initialize new model
    system->person_models[model_index].person_id = person_id;
    sprintf(system->person_models[model_index].person_name, "Person%d",
            person_id);
    system->person_models[model_index].prediction_confidence = 0.3f;
    system->person_models[model_index].relationship_quality = 0.5f;
    system->person_models[model_index].trust_level = 0.3f;
    system->person_models[model_index].interaction_count = 0;

    // Initialize traits with neutral values
    for (int i = 0; i < 10; i++) {
      system->person_models[model_index].observed_traits[i] = 0.5f;
    }
  }

  // Calculate prediction accuracy
  float prediction_error = 0.0f;
  for (int i = 0; i < 5; i++) { // Assuming behavior vectors are of length 5
    prediction_error += fabs(predicted_behavior[i] - observed_behavior[i]);
  }
  prediction_error /= 5.0f;

  // Update prediction accuracy for this person
  PersonModel *model = &system->person_models[model_index];
  model->prediction_confidence =
      model->prediction_confidence * 0.9f + (1.0f - prediction_error) * 0.1f;
  model->interaction_count++;

  // Update system-wide behavior prediction accuracy
  system->behavior_prediction_accuracy = 0.0f;
  for (int i = 0; i < system->model_count; i++) {
    system->behavior_prediction_accuracy +=
        system->person_models[i].prediction_confidence;
  }
  system->behavior_prediction_accuracy /= system->model_count;
}

float negotiateOutcome(SocialSystem *system, int person_id, float *goals,
                       float *other_goals, float *compromise) {
  // Find person model
  int model_index = -1;
  for (int i = 0; i < system->model_count; i++) {
    if (system->person_models[i].person_id == person_id) {
      model_index = i;
      break;
    }
  }

  float trust_factor = (model_index >= 0)
                           ? system->person_models[model_index].trust_level
                           : 0.3f;

  // Balance between self-interest and other-interest based on empathy and trust
  float self_weight = 1.0f - (system->empathy_level * 0.5f);
  float other_weight = system->empathy_level * trust_factor;

  // Calculate compromise solution
  for (int i = 0; i < 5; i++) { // Assuming goal vectors are of length 5
    compromise[i] = (goals[i] * self_weight + other_goals[i] * other_weight) /
                    (self_weight + other_weight);
  }

  // Calculate satisfaction level (how close to own goals)
  float satisfaction = 0.0f;
  for (int i = 0; i < 5; i++) {
    satisfaction += (1.0f - fabs(goals[i] - compromise[i]));
  }
  satisfaction /= 5.0f;

  // Improve negotiation skill based on outcome
  system->negotiation_skill =
      system->negotiation_skill * (1.0f - system->learning_rate) +
      satisfaction * system->learning_rate;

  // Update trust if this is a known person
  if (model_index >= 0) {
    // Trust increases if compromise favors the other party somewhat
    float generosity = 0.0f;
    for (int i = 0; i < 5; i++) {
      if (fabs(compromise[i] - other_goals[i]) <
          fabs(compromise[i] - goals[i])) {
        generosity += 0.1f;
      }
    }

    system->person_models[model_index].trust_level =
        system->person_models[model_index].trust_level * 0.9f +
        (satisfaction + generosity) * 0.1f;
  }

  return satisfaction;
}

float calculateInteractionDiversity(SocialSystem *system) {
  if (system->interaction_count == 0)
    return 0.0f;

  // Count unique interaction types
  char unique_types[20][32];
  int unique_count = 0;

  for (int i = 0; i < system->interaction_count; i++) {
    bool found = false;
    for (int j = 0; j < unique_count; j++) {
      if (strcmp(system->interactions[i].interaction_type, unique_types[j]) ==
          0) {
        found = true;
        break;
      }
    }

    if (!found && unique_count < 20) {
      strncpy(unique_types[unique_count],
              system->interactions[i].interaction_type, 31);
      unique_types[unique_count][31] = '\0';
      unique_count++;
    }
  }

  // Calculate diversity as ratio of unique types to interactions
  return (float)unique_count / fminf(20.0f, (float)system->interaction_count);
}

void recordSocialInteraction(SocialSystem *system, int person_id,
                             float *emotional_state, float cooperation_level,
                             float satisfaction, const char *type,
                             const char *context) {
  // If we've reached max interactions, make room by shifting array
  if (system->interaction_count >= system->max_interactions) {
    // Free the context string of the oldest interaction
    free(system->interactions[0].context);

    // Shift all interactions one position forward
    for (int i = 0; i < system->max_interactions - 1; i++) {
      system->interactions[i] = system->interactions[i + 1];
    }
    system->interaction_count = system->max_interactions - 1;
  }

  // Add new interaction at the end
  int idx = system->interaction_count;
  system->interactions[idx].timestamp = (unsigned int)time(NULL);
  system->interactions[idx].person_id = person_id;
  memcpy(system->interactions[idx].emotional_state, emotional_state,
         5 * sizeof(float));
  system->interactions[idx].cooperation_level = cooperation_level;
  system->interactions[idx].outcome_satisfaction = satisfaction;
  strncpy(system->interactions[idx].interaction_type, type, 31);
  system->interactions[idx].interaction_type[31] =
      '\0'; // Ensure null termination

  // Allocate and copy context
  system->interactions[idx].context = strdup(context);

  system->interaction_count++;

  // Update social awareness based on interaction history diversity
  float type_diversity = calculateInteractionDiversity(system);
  system->social_awareness =
      system->social_awareness * 0.95f + type_diversity * 0.05f;
}

void predictBehavior(SocialSystem *system, int person_id, const char *context,
                     float *predicted_behavior) {
  // Find person model
  int model_index = -1;
  for (int i = 0; i < system->model_count; i++) {
    if (system->person_models[i].person_id == person_id) {
      model_index = i;
      break;
    }
  }

  // Default prediction is neutral if no model exists
  for (int i = 0; i < 5; i++) {
    predicted_behavior[i] = 0.5f;
  }

  if (model_index >= 0) {
    PersonModel *model = &system->person_models[model_index];

    // Find similar past interactions with this person
    float context_influence = 0.0f;
    for (int i = 0; i < system->interaction_count; i++) {
      if (system->interactions[i].person_id == person_id) {
        // Simple context similarity check (in real implementation, use NLP)
        if (strstr(system->interactions[i].context, context) != NULL) {
          // More recent interactions have more influence
          float recency = system->forgetting_factor *
                          (1.0f - (float)i / system->interaction_count);

          // Add this interaction's influence to prediction
          for (int j = 0; j < 5; j++) {
            predicted_behavior[j] +=
                system->interactions[i].emotional_state[j] * recency * 0.2f;
          }

          context_influence += recency;
        }
      }
    }

    // If we found similar contexts, normalize predictions
    if (context_influence > 0.0f) {
      for (int i = 0; i < 5; i++) {
        predicted_behavior[i] /= (1.0f + context_influence);
        predicted_behavior[i] = fmin(1.0f, fmax(0.0f, predicted_behavior[i]));
      }
    } else {
      // Fall back to using trait-based prediction
      for (int i = 0; i < 5; i++) {
        predicted_behavior[i] =
            0.7f * model->observed_traits[i % 5] + 0.3f * predicted_behavior[i];
      }
    }
  }
}

void applySocialInfluence(SocialSystem *system, Neuron *neurons, float *weights,
                          int max_neurons) {
  int social_neuron_start = max_neurons / 2;
  int social_neuron_count = max_neurons / 10;

  for (int i = 0;
       i < social_neuron_count && i + social_neuron_start < max_neurons; i++) {
    int neuron_idx = social_neuron_start + i;

    // Scale neuron output based on social capabilities
    float social_factor =
        (system->empathy_level + system->negotiation_skill +
         system->behavior_prediction_accuracy + system->social_awareness) /
        4.0f;

    // Apply social influence proportional to social skills
    neurons[neuron_idx].output =
        neurons[neuron_idx].output * (1.0f - 0.3f * social_factor) +
        0.3f * social_factor;

    // Modify weights to strengthen social connections
    for (int j = 0; j < 10 && j < MAX_CONNECTIONS; j++) {
      int conn_idx = neuron_idx * MAX_CONNECTIONS + j;
      weights[conn_idx] = weights[conn_idx] * 0.9f + 0.1f * social_factor;
    }
  }
}

char *generateSocialFeedback(SocialSystem *system, const char *context) {
  char *feedback = (char *)malloc(256 * sizeof(char));
  if (feedback == NULL) {
    return NULL;
  }

  // Generate different types of feedback based on social metrics
  if (system->empathy_level < 0.4f) {
    sprintf(feedback,
            "Consider the emotional impact of actions on others. Context: %s",
            context);
  } else if (system->negotiation_skill < 0.4f) {
    sprintf(feedback,
            "Look for win-win solutions when conflicts arise. Context: %s",
            context);
  } else if (system->behavior_prediction_accuracy < 0.4f) {
    sprintf(
        feedback,
        "Pay closer attention to patterns in others' behaviors. Context: %s",
        context);
  } else if (system->social_awareness < 0.4f) {
    sprintf(feedback,
            "Consider broader social norms and expectations. Context: %s",
            context);
  } else {
    sprintf(feedback,
            "Social skills developing well. Continue practicing in various "
            "contexts. Current context: %s",
            context);
  }

  return feedback;
}

void freeSocialSystem(SocialSystem *system) {
  if (system == NULL)
    return;

  // Free all interaction contexts
  for (int i = 0; i < system->interaction_count; i++) {
    free(system->interactions[i].context);
  }

  free(system->interactions);
  free(system->person_models);
  free(system);
}

NeuronSpecializationSystem *initializeSpecializationSystem(float threshold) {
  NeuronSpecializationSystem *system =
      (NeuronSpecializationSystem *)malloc(sizeof(NeuronSpecializationSystem));
  if (system == NULL) {
    fprintf(stderr, "Failed to allocate memory for specialization system\n");
    return NULL;
  }

  system->count = 0;
  system->specialization_threshold = threshold;

  // Initialize type distribution
  for (int i = 0; i < MAX_SPECIALIZATIONS; i++) {
    system->type_distribution[i] = 0.0f;
  }

  return system;
}

void detectSpecializations(NeuronSpecializationSystem *system, Neuron *neurons,
                           int max_neurons, float *input_tensor,
                           float *target_outputs, float *previous_outputs,
                           float *previous_states) {
  if (system == NULL || neurons == NULL)
    return;

  // Analyze neurons for specialization potential
  for (int i = 0; i < max_neurons; i++) {
    // Skip neurons that are already specialized
    bool already_specialized = false;
    for (unsigned int j = 0; j < system->count; j++) {
      if (system->neurons[j].neuron_id == static_cast<unsigned int>(i)) {
        already_specialized = true;

        // Update activation history
        system->neurons[j]
            .activation_history[system->neurons[j].history_index] =
            neurons[i].output;
        system->neurons[j].history_index =
            (system->neurons[j].history_index + 1) % 50;

        // Update average activation
        float sum = 0.0f;
        for (int k = 0; k < 50; k++) {
          sum += system->neurons[j].activation_history[k];
        }
        system->neurons[j].avg_activation = sum / 50.0f;

        break;
      }
    }

    if (already_specialized)
      continue;

    // Check for specialization patterns
    if (system->count < MAX_SPECIALIZED_NEURONS) {
      float pattern_score = 0.0f;
      float feature_score = 0.0f;
      float temporal_score = 0.0f;
      float context_score = 0.0f;
      float decision_score = 0.0f;
      float memory_score = 0.0f;
      float emotional_score = 0.0f;
      float prediction_score = 0.0f;

      // Pattern detection score - high activations when specific input patterns
      // are present
      if (neurons[i].output > 0.7f && input_tensor[i % INPUT_SIZE] > 0.6f) {
        pattern_score = 0.8f;
      }

      // Feature extraction score - consistent response to specific features
      if (neurons[i].num_connections > 3 && neurons[i].output > 0.5f) {
        feature_score = 0.7f;
      }

      // Temporal processing score - analyze how output changes over time
      // Look for consistency in activation timing
      float temporal_variance = 0.0f;
      float last_activation = 0.0f;
      int activation_changes = 0;

      if (previous_outputs != NULL) {
        // Calculate how consistently the neuron changes between activations
        float current_diff = fabs(neurons[i].output - previous_outputs[i]);

        // Check for neurons that show consistent changes over time
        if (current_diff > 0.2f) {
          activation_changes++;
        }

        // Calculate temporal variance (how consistently the neuron changes)
        temporal_variance =
            fabs(current_diff - fabs(neurons[i].state - previous_states[i]));

        // Neurons with regular activation patterns score higher
        if (temporal_variance < 0.3f && activation_changes > 0) {
          temporal_score = 0.7f + (0.2f * (1.0f - temporal_variance));
        }
        // Neurons that respond to transitions also score high
        else if (current_diff > 0.5f) {
          temporal_score = 0.6f + (0.3f * current_diff);
        }
      }

      // Context integration score - activation correlates with global context
      if (neurons[i].layer_id > 0 && neurons[i].output > 0.4f) {
        // Higher layers tend to integrate more context
        context_score = 0.65f + (neurons[i].layer_id * 0.05f);
        // Cap at 0.9
        if (context_score > 0.9f)
          context_score = 0.9f;
      }

      // Decision making score - output correlates with target outputs
      if (target_outputs != NULL) {
        float output_diff =
            fabs(neurons[i].output - target_outputs[i % INPUT_SIZE]);
        decision_score = 1.0f - fmin(output_diff, 1.0f);

        // Boost score for neurons with high connection count
        // as they likely integrate more information for decisions
        if (neurons[i].num_connections > 5) {
          decision_score *= 1.2f;
          // Cap at 0.95
          if (decision_score > 0.95f)
            decision_score = 0.95f;
        }
      }

      // Memory encoding score - stable output over time with specific patterns
      // Memory neurons often have moderate but stable activations
      if (neurons[i].output > 0.3f && neurons[i].output < 0.8f) {
        // Neurons with stable outputs in mid-range are good memory candidates
        memory_score = 0.5f + (neurons[i].output * 0.3f);

        // Memory neurons often have many connections
        if (neurons[i].num_connections > 4) {
          memory_score += 0.1f;
        }

        // Cap at 0.9
        if (memory_score > 0.9f)
          memory_score = 0.9f;
      }

      // Emotional processing score - high variance in output based on context
      // Emotional processors often have strong reactions to certain inputs
      if (neurons[i].output > 0.8f || neurons[i].output < 0.2f) {
        // Strong outputs (very high or very low) can indicate emotional
        // processing
        emotional_score = 0.6f + fabs(neurons[i].output - 0.5f);

        // Balance with connection count - emotional neurons often integrate
        // inputs from multiple sources
        if (neurons[i].num_connections > 3) {
          emotional_score += 0.1f;
        }

        // Cap at 0.9
        if (emotional_score > 0.9f)
          emotional_score = 0.9f;
      }

      // Prediction generation score - output anticipates future inputs
      // Prediction neurons often have partial activations before full stimulus
      if (neurons[i].state > neurons[i].output && neurons[i].state > 0.4f) {
        // State higher than output suggests anticipatory activation
        prediction_score = 0.5f + (neurons[i].state - neurons[i].output) * 0.5f;

        // Later layers are more likely to be predictive
        if (neurons[i].layer_id > 1) {
          prediction_score += 0.1f * neurons[i].layer_id;
        }

        // Cap at 0.9
        if (prediction_score > 0.9f)
          prediction_score = 0.9f;
      }

      // Find highest specialization score
      float scores[MAX_SPECIALIZATIONS + 1] = {
          0.0f, // SPEC_NONE
          pattern_score,  feature_score, temporal_score,  context_score,
          decision_score, memory_score,  emotional_score, prediction_score};

      float max_score = 0.0f;
      int spec_type = SPEC_NONE;

      for (int j = 1; j < MAX_SPECIALIZATIONS; j++) {
        if (scores[j] > max_score) {
          max_score = scores[j];
          spec_type = j;
        }
      }

      // If score exceeds threshold, add as specialized neuron
      if (max_score >= system->specialization_threshold) {
        system->neurons[system->count].neuron_id = i;
        system->neurons[system->count].type =
            (NeuronSpecializationType)spec_type;
        system->neurons[system->count].specialization_score = max_score;
        system->neurons[system->count].importance_factor = 1.0f;
        system->neurons[system->count].avg_activation = neurons[i].output;
        system->neurons[system->count].history_index = 0;

        // Initialize activation history
        for (int k = 0; k < 50; k++) {
          system->neurons[system->count].activation_history[k] = 0.0f;
        }
        system->neurons[system->count].activation_history[0] =
            neurons[i].output;

        system->count++;

        // Update type distribution
        for (int j = 0; j < MAX_SPECIALIZATIONS; j++) {
          // Reset counts first
          system->type_distribution[j] = 0.0f;
        }

        // Count each specialization type
        for (size_t j = 0; j < static_cast<size_t>(system->count); j++) {
          system->type_distribution[system->neurons[j].type] += 1.0f;
        }

        // Convert to distribution (normalize)
        for (int j = 0; j < MAX_SPECIALIZATIONS; j++) {
          system->type_distribution[j] /= system->count;
        }
      }
    }
  }
}

void applySpecializations(NeuronSpecializationSystem *system, Neuron *neurons,
                          float *weights, int *connections, int max_neurons,
                          int max_connections) {
  if (system == NULL || neurons == NULL)
    return;

  // Apply effects based on specialization type
  for (unsigned int i = 0; i < system->count; i++) {
    unsigned int neuron_id = system->neurons[i].neuron_id;
    if (static_cast<unsigned int>(neuron_id) >=
        static_cast<unsigned int>(max_neurons))
      continue;

    float boost_factor = system->neurons[i].specialization_score *
                         system->neurons[i].importance_factor;

    switch (system->neurons[i].type) {
    case SPEC_PATTERN_DETECTOR:
      // Enhance pattern detection by boosting activation
      neurons[neuron_id].state *= (1.0f + 0.2f * boost_factor);
      break;

    case SPEC_FEATURE_EXTRACTOR:
      // Enhance feature extraction by slightly raising activation threshold
      neurons[neuron_id].state =
          fmax(0.1f * boost_factor, neurons[neuron_id].state);
      break;

    case SPEC_TEMPORAL_PROCESSOR:
      // Enhance temporal processing by making neuron more responsive to changes
      neurons[neuron_id].state +=
          0.05f * boost_factor *
          (neurons[neuron_id].state - system->neurons[i].avg_activation);
      break;

    case SPEC_CONTEXT_INTEGRATOR:
      // Enhance context integration by increasing connection influence
      for (unsigned int j = 0; j < neurons[neuron_id].num_connections &&
                               static_cast<int>(j) < max_connections;
           j++) {
        int connection_idx = neuron_id * max_connections + j;
        int target = connections[connection_idx];
        weights[connection_idx] *= (1.0f + 0.1f * boost_factor);
      }
      break;

    case SPEC_DECISION_MAKER:
      // Enhance decision making by increasing output contrast
      neurons[neuron_id].output =
          neurons[neuron_id].output > 0.5f
              ? neurons[neuron_id].output +
                    (1.0f - neurons[neuron_id].output) * 0.2f * boost_factor
              : neurons[neuron_id].output -
                    neurons[neuron_id].output * 0.2f * boost_factor;
      break;

    case SPEC_MEMORY_ENCODER:
      // Enhance memory encoding by making activation more stable
      neurons[neuron_id].state = neurons[neuron_id].state * 0.9f +
                                 system->neurons[i].avg_activation * 0.1f;
      break;

    case SPEC_EMOTIONAL_PROCESSOR:
      // Enhance emotional processing by allowing higher activation variability
      neurons[neuron_id].state *=
          (1.0f + (rand() / (float)RAND_MAX - 0.5f) * 0.2f * boost_factor);
      break;

    case SPEC_PREDICTION_GENERATOR:
      // Enhance prediction generation by slight forward leaning bias
      neurons[neuron_id].state *= (1.0f + 0.15f * boost_factor);
      break;

    default:
      break;
    }
  }
}

void updateSpecializationImportance(NeuronSpecializationSystem *system,
                                    float network_performance, float error_rate,
                                    Neuron *neurons) {
  if (system == NULL || neurons == NULL)
    return;

  for (unsigned int i = 0; i < system->count; i++) {
    unsigned int neuron_id = system->neurons[i].neuron_id;

    // Calculate contribution to overall performance
    float activation_variance = 0.0f;
    float prev = system->neurons[i].activation_history[0];

    for (int j = 1; j < 50; j++) {
      activation_variance +=
          fabs(system->neurons[i].activation_history[j] - prev);
      prev = system->neurons[i].activation_history[j];
    }
    activation_variance /= 49.0f;

    // Neurons with appropriate activity level for their specialization get
    // increased importance
    float activity_score = 0.0f;

    switch (system->neurons[i].type) {
    case SPEC_PATTERN_DETECTOR:
      // Pattern detectors should be selective (sometimes high, sometimes low)
      activity_score = activation_variance;
      break;

    case SPEC_FEATURE_EXTRACTOR:
      // Feature extractors should maintain consistent medium activity
      activity_score =
          1.0f - fabs(system->neurons[i].avg_activation - 0.5f) * 2.0f;
      break;

    case SPEC_TEMPORAL_PROCESSOR:
      // Temporal processors should have dynamic activity
      activity_score = activation_variance * 2.0f;
      break;

    case SPEC_CONTEXT_INTEGRATOR:
      // Context integrators should have stable but non-zero activity
      activity_score =
          system->neurons[i].avg_activation * (1.0f - activation_variance);
      break;

    case SPEC_DECISION_MAKER:
      // Decision makers should have clear high or low outputs
      activity_score = fabs(system->neurons[i].avg_activation - 0.5f) * 2.0f;
      break;

    case SPEC_MEMORY_ENCODER:
      // Memory encoders should have stable activity
      activity_score = 1.0f - activation_variance * 2.0f;
      break;

    case SPEC_EMOTIONAL_PROCESSOR:
      // Emotional processors should have variable activity
      activity_score = activation_variance * 1.5f;
      break;

    case SPEC_PREDICTION_GENERATOR:
      // Prediction generators should be active
      activity_score = system->neurons[i].avg_activation;
      break;

    default:
      activity_score = 0.5f;
      break;
    }

    // Update importance based on activity score and network performance
    float performance_factor = network_performance * (1.0f - error_rate);
    float importance_delta =
        (activity_score * 0.6f + performance_factor * 0.4f - 0.5f) * 0.1f;

    system->neurons[i].importance_factor =
        fmax(0.1f, fmin(2.0f, system->neurons[i].importance_factor +
                                  importance_delta));
  }
}

float evaluateSpecializationEffectiveness(NeuronSpecializationSystem *system,
                                          float network_performance) {
  if (system == NULL || system->count == 0)
    return 0.0f;

  float total_importance = 0.0f;
  float max_importance = 0.0f;
  float min_importance = 2.0f;

  for (unsigned int i = 0; i < system->count; i++) {
    total_importance += system->neurons[i].importance_factor;
    max_importance = fmax(max_importance, system->neurons[i].importance_factor);
    min_importance = fmin(min_importance, system->neurons[i].importance_factor);
  }

  float avg_importance = total_importance / system->count;
  float importance_variance = max_importance - min_importance;

  // Calculate specialization diversity - higher is better
  float type_diversity = 0.0f;
  for (int i = 1; i < MAX_SPECIALIZATIONS; i++) {
    if (system->type_distribution[i] > 0.0f) {
      type_diversity -=
          system->type_distribution[i] * logf(system->type_distribution[i]);
    }
  }
  type_diversity /= logf(MAX_SPECIALIZATIONS - 1); // Normalize

  // Calculate overall effectiveness
  return (network_performance * 0.4f + avg_importance * 0.3f +
          type_diversity * 0.3f);
}

void printSpecializationStats(NeuronSpecializationSystem *system) {
  if (system == NULL)
    return;

  const char *type_names[MAX_SPECIALIZATIONS + 1] = {"None",
                                                     "Pattern Detector",
                                                     "Feature Extractor",
                                                     "Temporal Processor",
                                                     "Context Integrator",
                                                     "Decision Maker",
                                                     "Memory Encoder",
                                                     "Emotional Processor",
                                                     "Prediction Generator"};

  printf("\nNeuron Specialization Statistics:\n");
  printf("Total Specialized Neurons: %u/%d\n", system->count,
         MAX_SPECIALIZED_NEURONS);

  printf("\nSpecialization Distribution:\n");
  for (int i = 1; i < MAX_SPECIALIZATIONS; i++) {
    printf("- %s: %.1f%%\n", type_names[i],
           system->type_distribution[i] * 100.0f);
  }

  printf("\nTop Specialized Neurons by Importance:\n");
  // Create a sorted array of indices based on importance
  unsigned int indices[MAX_SPECIALIZED_NEURONS];
  for (unsigned int i = 0; i < system->count; i++) {
    indices[i] = i;
  }

  // Simple bubble sort for indices
  for (unsigned int i = 0; i < system->count - 1; i++) {
    for (unsigned int j = 0; j < system->count - i - 1; j++) {
      if (system->neurons[indices[j]].importance_factor <
          system->neurons[indices[j + 1]].importance_factor) {
        unsigned int temp = indices[j];
        indices[j] = indices[j + 1];
        indices[j + 1] = temp;
      }
    }
  }

  // Print top 5 or fewer if less than 5
  for (unsigned int i = 0; i < 5 && i < system->count; i++) {
    unsigned int idx = indices[i];
    printf("Neuron #%u: %s (Importance: %.2f, Score: %.2f, Avg Act: %.2f)\n",
           system->neurons[idx].neuron_id,
           type_names[system->neurons[idx].type],
           system->neurons[idx].importance_factor,
           system->neurons[idx].specialization_score,
           system->neurons[idx].avg_activation);
  }
}

void freeSpecializationSystem(NeuronSpecializationSystem *system) {
  if (system != NULL) {
    free(system);
  }
}

// Save and Load functions for MetaController
void saveMetaController(MetaController *controller, const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Error opening MetaController file for writing\n");
    return;
  }

  fwrite(&controller->meta_learning_rate, sizeof(float), 1, fp);
  fwrite(&controller->exploration_factor, sizeof(float), 1, fp);
  fwrite(&controller->num_regions, sizeof(int), 1, fp);

  fwrite(controller->region_importance_scores, sizeof(float),
         controller->num_regions, fp);
  fwrite(controller->learning_efficiency_history, sizeof(float),
         controller->num_regions, fp);

  fclose(fp);
}

MetaController *loadMetaController(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Error opening MetaController file for reading\n");
    return NULL;
  }

  float meta_learning_rate, exploration_factor;
  int num_regions;

  fread(&meta_learning_rate, sizeof(float), 1, fp);
  fread(&exploration_factor, sizeof(float), 1, fp);
  fread(&num_regions, sizeof(int), 1, fp);

  MetaController *controller = initializeMetaController(num_regions);
  if (controller == NULL) {
    fclose(fp);
    return NULL;
  }

  controller->meta_learning_rate = meta_learning_rate;
  controller->exploration_factor = exploration_factor;

  fread(controller->region_importance_scores, sizeof(float), num_regions, fp);
  fread(controller->learning_efficiency_history, sizeof(float), num_regions,
        fp);

  fclose(fp);
  return controller;
}

// Save and Load functions for IntrinsicMotivation
void saveIntrinsicMotivation(IntrinsicMotivation *motivation,
                             const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Error opening IntrinsicMotivation file for writing\n");
    return;
  }

  fwrite(motivation, sizeof(IntrinsicMotivation), 1, fp);

  fclose(fp);
}

IntrinsicMotivation *loadIntrinsicMotivation(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Error opening IntrinsicMotivation file for reading\n");
    return NULL;
  }

  IntrinsicMotivation *motivation = initializeMotivationSystem();
  if (motivation == NULL) {
    fclose(fp);
    return NULL;
  }

  fread(motivation, sizeof(IntrinsicMotivation), 1, fp);

  fclose(fp);
  return motivation;
}

// Save and Load functions for NetworkPerformanceMetrics
void saveNetworkPerformanceMetrics(NetworkPerformanceMetrics *metrics,
                                   const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Error opening NetworkPerformanceMetrics file for writing\n");
    return;
  }

  fwrite(&metrics->num_regions, sizeof(int), 1, fp);
  fwrite(metrics->region_performance_scores, sizeof(float),
         metrics->num_regions, fp);
  fwrite(metrics->region_error_rates, sizeof(float), metrics->num_regions, fp);
  fwrite(metrics->region_output_variance, sizeof(float), metrics->num_regions,
         fp);

  fclose(fp);
}

NetworkPerformanceMetrics *loadNetworkPerformanceMetrics(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Error opening NetworkPerformanceMetrics file for reading\n");
    return NULL;
  }

  int num_regions;
  fread(&num_regions, sizeof(int), 1, fp);

  NetworkPerformanceMetrics *metrics =
      initializePerformanceMetrics(num_regions);
  if (metrics == NULL) {
    fclose(fp);
    return NULL;
  }

  fread(metrics->region_performance_scores, sizeof(float), num_regions, fp);
  fread(metrics->region_error_rates, sizeof(float), num_regions, fp);
  fread(metrics->region_output_variance, sizeof(float), num_regions, fp);

  fclose(fp);
  return metrics;
}

// Save and Load functions for ReflectionParameters
void saveReflectionParameters(ReflectionParameters *params,
                              const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Error opening ReflectionParameters file for writing\n");
    return;
  }

  fwrite(params, sizeof(ReflectionParameters), 1, fp);

  fclose(fp);
}

ReflectionParameters *loadReflectionParameters(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Error opening ReflectionParameters file for reading\n");
    return NULL;
  }

  ReflectionParameters *params = initializeReflectionParameters();
  if (params == NULL) {
    fclose(fp);
    return NULL;
  }

  fread(params, sizeof(ReflectionParameters), 1, fp);

  fclose(fp);
  return params;
}

// Save and Load functions for SelfIdentitySystem
void saveSelfIdentitySystem(SelfIdentitySystem *identity,
                            const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Error opening SelfIdentitySystem file for writing\n");
    return;
  }

  // Write scalar values
  fwrite(&identity->num_core_values, sizeof(uint32_t), 1, fp);
  fwrite(&identity->num_beliefs, sizeof(uint32_t), 1, fp);
  fwrite(&identity->num_markers, sizeof(uint32_t), 1, fp);
  fwrite(&identity->history_size, sizeof(uint32_t), 1, fp);
  fwrite(&identity->pattern_size, sizeof(uint32_t), 1, fp);
  fwrite(&identity->consistency_score, sizeof(float), 1, fp);
  fwrite(&identity->adaptation_rate, sizeof(float), 1, fp);
  fwrite(&identity->confidence_level, sizeof(float), 1, fp);
  fwrite(&identity->coherence_window, sizeof(uint32_t), 1, fp);

  // Write verification structure
  fwrite(&identity->verification.threshold, sizeof(float), 1, fp);
  fwrite(&identity->verification.state_size, sizeof(uint32_t), 1, fp);

  // Write arrays
  fwrite(identity->core_values, sizeof(float), identity->num_core_values, fp);
  fwrite(identity->belief_system, sizeof(float), identity->num_beliefs, fp);
  fwrite(identity->identity_markers, sizeof(float), identity->num_markers, fp);
  fwrite(identity->experience_history, sizeof(float), identity->history_size,
         fp);
  fwrite(identity->behavioral_patterns, sizeof(float), identity->pattern_size,
         fp);
  fwrite(identity->temporal_coherence, sizeof(float),
         identity->coherence_window, fp);
  fwrite(identity->verification.reference_state, sizeof(float),
         identity->verification.state_size, fp);

  fclose(fp);
}

SelfIdentitySystem *loadSelfIdentitySystem(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Error opening SelfIdentitySystem file for reading\n");
    return NULL;
  }

  uint32_t num_core_values, num_beliefs, num_markers, history_size,
      pattern_size;

  fread(&num_core_values, sizeof(uint32_t), 1, fp);
  fread(&num_beliefs, sizeof(uint32_t), 1, fp);
  fread(&num_markers, sizeof(uint32_t), 1, fp);
  fread(&history_size, sizeof(uint32_t), 1, fp);
  fread(&pattern_size, sizeof(uint32_t), 1, fp);

  SelfIdentitySystem *identity = initializeSelfIdentity(
      num_core_values, num_beliefs, num_markers, history_size, pattern_size);
  if (identity == NULL) {
    fclose(fp);
    return NULL;
  }

  // Read scalar values
  fread(&identity->consistency_score, sizeof(float), 1, fp);
  fread(&identity->adaptation_rate, sizeof(float), 1, fp);
  fread(&identity->confidence_level, sizeof(float), 1, fp);
  fread(&identity->coherence_window, sizeof(uint32_t), 1, fp);

  // Read verification structure
  fread(&identity->verification.threshold, sizeof(float), 1, fp);
  fread(&identity->verification.state_size, sizeof(uint32_t), 1, fp);

  // Read arrays
  fread(identity->core_values, sizeof(float), identity->num_core_values, fp);
  fread(identity->belief_system, sizeof(float), identity->num_beliefs, fp);
  fread(identity->identity_markers, sizeof(float), identity->num_markers, fp);
  fread(identity->experience_history, sizeof(float), identity->history_size,
        fp);
  fread(identity->behavioral_patterns, sizeof(float), identity->pattern_size,
        fp);
  fread(identity->temporal_coherence, sizeof(float), identity->coherence_window,
        fp);
  fread(identity->verification.reference_state, sizeof(float),
        identity->verification.state_size, fp);

  fclose(fp);
  return identity;
}

// Save and Load functions for KnowledgeFilter
void saveKnowledgeFilter(KnowledgeFilter *filter, const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Error opening KnowledgeFilter file for writing\n");
    return;
  }

  fwrite(&filter->num_categories, sizeof(uint32_t), 1, fp);
  fwrite(&filter->capacity, sizeof(uint32_t), 1, fp);
  fwrite(&filter->num_problems, sizeof(uint32_t), 1, fp);
  fwrite(&filter->problem_capacity, sizeof(uint32_t), 1, fp);

  fwrite(filter->categories, sizeof(KnowledgeCategory), filter->num_categories,
         fp);

  // Write problem history
  fwrite(filter->problem_history, sizeof(ProblemInstance), filter->num_problems,
         fp);

  // Write similarity matrix
  fwrite(filter->category_similarity_matrix, sizeof(float),
         filter->num_categories * filter->num_categories, fp);

  fclose(fp);
}

KnowledgeFilter *loadKnowledgeFilter(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Error opening KnowledgeFilter file for reading\n");
    return NULL;
  }

  uint32_t capacity;
  fread(&capacity, sizeof(uint32_t), 1, fp);

  // Skip num_categories, we'll read it after initialization
  fseek(fp, 0, SEEK_SET);

  KnowledgeFilter *filter = initializeKnowledgeFilter(capacity);
  if (filter == NULL) {
    fclose(fp);
    return NULL;
  }

  fread(&filter->num_categories, sizeof(uint32_t), 1, fp);
  fread(&filter->capacity, sizeof(uint32_t), 1, fp);
  fread(&filter->num_problems, sizeof(uint32_t), 1, fp);
  fread(&filter->problem_capacity, sizeof(uint32_t), 1, fp);

  // Read categories
  fread(filter->categories, sizeof(KnowledgeCategory), filter->num_categories,
        fp);

  // Read problem history
  fread(filter->problem_history, sizeof(ProblemInstance), filter->num_problems,
        fp);

  // Read similarity matrix
  fread(filter->category_similarity_matrix, sizeof(float),
        filter->num_categories * filter->num_categories, fp);

  fclose(fp);
  return filter;
}

// Save and Load functions for MetacognitionMetrics
void saveMetacognitionMetrics(MetacognitionMetrics *metrics,
                              const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Error opening MetacognitionMetrics file for writing\n");
    return;
  }

  fwrite(metrics, sizeof(MetacognitionMetrics), 1, fp);

  fclose(fp);
}

MetacognitionMetrics *loadMetacognitionMetrics(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Error opening MetacognitionMetrics file for reading\n");
    return NULL;
  }

  MetacognitionMetrics *metrics = initializeMetacognitionMetrics();
  if (metrics == NULL) {
    fclose(fp);
    return NULL;
  }

  fread(metrics, sizeof(MetacognitionMetrics), 1, fp);

  fclose(fp);
  return metrics;
}

// Save and Load functions for MetaLearningState
void saveMetaLearningState(MetaLearningState *state, const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Error opening MetaLearningState file for writing\n");
    return;
  }

  fwrite(&state->learning_efficiency, sizeof(float), 1, fp);
  fwrite(&state->exploration_rate, sizeof(float), 1, fp);
  fwrite(&state->stability_index, sizeof(float), 1, fp);
  fwrite(&state->current_phase, sizeof(uint32_t), 1, fp);

  fwrite(state->priority_weights, sizeof(float), 4, fp);

  fclose(fp);
}

MetaLearningState *loadMetaLearningState(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Error opening MetaLearningState file for reading\n");
    return NULL;
  }

  MetaLearningState *state = initializeMetaLearningState(4);
  if (state == NULL) {
    fclose(fp);
    return NULL;
  }

  fread(&state->learning_efficiency, sizeof(float), 1, fp);
  fread(&state->exploration_rate, sizeof(float), 1, fp);
  fread(&state->stability_index, sizeof(float), 1, fp);
  fread(&state->current_phase, sizeof(uint32_t), 1, fp);

  fread(state->priority_weights, sizeof(float), 4, fp);

  fclose(fp);
  return state;
}

void saveAllSystems(MetaController *metaController,
                    IntrinsicMotivation *motivation,
                    NetworkPerformanceMetrics *performanceMetrics,
                    ReflectionParameters *reflection_params,
                    SelfIdentitySystem *identity_system,
                    KnowledgeFilter *knowledge_filter,
                    MetacognitionMetrics *metacognition,
                    MetaLearningState *meta_learning_state,
                    SocialSystem *social_system) {
  saveMetaController(metaController, "metacontroller.dat");
  saveIntrinsicMotivation(motivation, "motivation.dat");
  saveNetworkPerformanceMetrics(performanceMetrics, "performance_metrics.dat");
  saveReflectionParameters(reflection_params, "reflection_params.dat");
  saveSelfIdentitySystem(identity_system, "identity_system.dat");
  saveKnowledgeFilter(knowledge_filter, "knowledge_filter.dat");
  saveMetacognitionMetrics(metacognition, "metacognition.dat");
  saveMetaLearningState(meta_learning_state, "meta_learning.dat");
}

// Global jump buffer for segmentation fault recovery
static jmp_buf segfault_recovery;
static volatile bool segfault_occurred = false;
static volatile void *fault_address = NULL;
static char fault_description[256] = {0};

// Function to validate memory block
bool isValidMemoryRegion(void *ptr, size_t size) {
  if (ptr == NULL)
    return false;

  volatile char test;
  char *start = (char *)ptr;
  char *end = start + size - 1;

  if (setjmp(segfault_recovery) == 0) {
    test = *start;
    test = *end;
    return true;
  } else {
    return false;
  }
}

// Function to validate memory block with additional checks
bool validateMemoryBlock(void *ptr, size_t expected_size,
                         const char *component_name) {
  if (ptr == NULL) {
    fprintf(stderr, "ERROR: %s - NULL pointer detected\n", component_name);
    return false;
  }

  if (!isValidMemoryRegion(ptr, expected_size)) {
    fprintf(stderr, "ERROR: %s - Invalid memory region at %p (size: %zu)\n",
            component_name, ptr, expected_size);
    return false;
  }

  return true;
}

// Segmentation fault handler
void segfault_handler(int sig, siginfo_t *si, void *unused) {
  segfault_occurred = true;
  fault_address = si->si_addr;

  snprintf(fault_description, sizeof(fault_description),
           "Segmentation fault at address %p (signal %d)", si->si_addr, sig);

  fprintf(stderr, "CRITICAL SEGFAULT CAUGHT: %s\n", fault_description);

  longjmp(segfault_recovery, 1);
}

// Initialize segmentation fault protection
void initializeSegfaultProtection() {
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));

  sa.sa_sigaction = segfault_handler;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);

  if (sigaction(SIGSEGV, &sa, NULL) == -1) {
    fprintf(stderr, "WARNING: Failed to install segfault handler\n");
  }
}

// Individual validation functions for each system component
bool validateWorkingMemory(WorkingMemorySystem *wm) {
  if (!validateMemoryBlock(wm, sizeof(WorkingMemorySystem), "WorkingMemory")) {
    return false;
  }

  if (wm->focus.entries != NULL && wm->focus.capacity > 0) {
    if (!validateMemoryBlock(wm->focus.entries,
                             wm->focus.capacity * sizeof(WorkingMemoryEntry),
                             "WorkingMemory->focus.entries")) {
      wm->focus.entries = NULL;
      wm->focus.size = 0;
      fprintf(stderr, "WARNING: Corrupted focus entries, reset to NULL\n");
    }
  }

  if (wm->active.entries != NULL && wm->active.capacity > 0) {
    if (!validateMemoryBlock(wm->active.entries,
                             wm->active.capacity * sizeof(WorkingMemoryEntry),
                             "WorkingMemory->active.entries")) {
      wm->active.entries = NULL;
      wm->active.size = 0;
      fprintf(stderr, "WARNING: Corrupted active entries, reset to NULL\n");
    }
  }

  if (wm->focus.size > wm->focus.capacity) {
    fprintf(stderr, "WARNING: Focus size exceeds capacity, correcting\n");
    wm->focus.size = wm->focus.capacity;
  }

  if (wm->active.size > wm->active.capacity) {
    fprintf(stderr, "WARNING: Active size exceeds capacity, correcting\n");
    wm->active.size = wm->active.capacity;
  }

  return true;
}

bool validateMetaController(MetaController *mc) {
  if (!validateMemoryBlock(mc, sizeof(MetaController), "MetaController")) {
    return false;
  }

  if (isnan(mc->meta_learning_rate) || isinf(mc->meta_learning_rate) ||
      mc->meta_learning_rate < 0.0f || mc->meta_learning_rate > 1.0f) {
    fprintf(stderr, "WARNING: Invalid meta learning rate, resetting to 0.01\n");
    mc->meta_learning_rate = 0.01f;
  }

  if (isnan(mc->exploration_factor) || isinf(mc->exploration_factor) ||
      mc->exploration_factor < 0.0f || mc->exploration_factor > 1.0f) {
    fprintf(stderr, "WARNING: Invalid exploration factor, resetting to 0.1\n");
    mc->exploration_factor = 0.1f;
  }

  if (mc->region_importance_scores != NULL && mc->num_regions > 0) {
    if (!validateMemoryBlock(mc->region_importance_scores,
                             mc->num_regions * sizeof(float),
                             "MetaController->region_importance_scores")) {
      mc->region_importance_scores = NULL;
      fprintf(stderr, "WARNING: Corrupted region importance scores\n");
    }
  }

  return true;
}

bool validatePerformanceMetrics(NetworkPerformanceMetrics *npm) {
  if (!validateMemoryBlock(npm, sizeof(NetworkPerformanceMetrics),
                           "NetworkPerformanceMetrics")) {
    return false;
  }

  if (npm->region_performance_scores != NULL && npm->num_regions > 0) {
    if (!validateMemoryBlock(npm->region_performance_scores,
                             npm->num_regions * sizeof(float),
                             "NetworkPerformanceMetrics->performance_scores")) {
      npm->region_performance_scores = NULL;
      fprintf(stderr, "WARNING: Corrupted performance scores\n");
    }
  }

  if (npm->region_error_rates != NULL && npm->num_regions > 0) {
    if (!validateMemoryBlock(npm->region_error_rates,
                             npm->num_regions * sizeof(float),
                             "NetworkPerformanceMetrics->error_rates")) {
      npm->region_error_rates = NULL;
      fprintf(stderr, "WARNING: Corrupted error rates\n");
    }
  }

  return true;
}

bool validateMotivationSystem(IntrinsicMotivation *im) {
  if (!validateMemoryBlock(im, sizeof(IntrinsicMotivation),
                           "IntrinsicMotivation")) {
    return false;
  }

  float *scores[] = {&im->novelty_score,   &im->competence_score,
                     &im->autonomy_score,  &im->mastery_level,
                     &im->curiosity_drive, &im->achievement_drive,
                     &im->exploration_rate};
  const char *names[] = {"novelty",   "competence",  "autonomy",   "mastery",
                         "curiosity", "achievement", "exploration"};

  for (int i = 0; i < 7; i++) {
    if (isnan(*scores[i]) || isinf(*scores[i]) || *scores[i] < 0.0f ||
        *scores[i] > 1.0f) {
      fprintf(stderr, "WARNING: Invalid %s score, resetting to 0.5\n",
              names[i]);
      *scores[i] = 0.5f;
    }
  }

  return true;
}

bool validateReflectionParameters(ReflectionParameters *rp) {
  if (!validateMemoryBlock(rp, sizeof(ReflectionParameters),
                           "ReflectionParameters")) {
    return false;
  }

  if (isnan(rp->current_adaptation_rate) ||
      isinf(rp->current_adaptation_rate) ||
      rp->current_adaptation_rate < 0.0f ||
      rp->current_adaptation_rate > 1.0f) {
    fprintf(stderr, "WARNING: Invalid adaptation rate, resetting to 0.01\n");
    rp->current_adaptation_rate = 0.01f;
  }

  if (isnan(rp->learning_rate) || isinf(rp->learning_rate) ||
      rp->learning_rate <= 0.0f || rp->learning_rate > 1.0f) {
    fprintf(stderr, "WARNING: Invalid learning rate, resetting to 0.01\n");
    rp->learning_rate = 0.01f;
  }

  return true;
}

bool validateIdentitySystem(SelfIdentitySystem *sis) {
  if (!validateMemoryBlock(sis, sizeof(SelfIdentitySystem),
                           "SelfIdentitySystem")) {
    return false;
  }

  if (sis->core_values != NULL && sis->num_core_values > 0) {
    if (!validateMemoryBlock(sis->core_values,
                             sis->num_core_values * sizeof(float),
                             "SelfIdentitySystem->core_values")) {
      sis->core_values = NULL;
      sis->num_core_values = 0;
      fprintf(stderr, "WARNING: Corrupted core values\n");
    }
  }

  if (isnan(sis->consistency_score) || isinf(sis->consistency_score)) {
    fprintf(stderr, "WARNING: Invalid consistency score, resetting to 0.5\n");
    sis->consistency_score = 0.5f;
  }

  return true;
}

bool validateKnowledgeFilter(KnowledgeFilter *kf) {
  if (!validateMemoryBlock(kf, sizeof(KnowledgeFilter), "KnowledgeFilter")) {
    return false;
  }

  if (kf->categories != NULL && kf->capacity > 0) {
    if (!validateMemoryBlock(kf->categories,
                             kf->capacity * sizeof(KnowledgeCategory),
                             "KnowledgeFilter->categories")) {
      kf->categories = NULL;
      kf->num_categories = 0;
      fprintf(stderr, "WARNING: Corrupted knowledge categories\n");
    }
  }

  if (kf->num_categories > kf->capacity) {
    fprintf(stderr, "WARNING: Category count exceeds capacity, correcting\n");
    kf->num_categories = kf->capacity;
  }

  return true;
}

bool validateMetacognition(MetacognitionMetrics *mm) {
  if (!validateMemoryBlock(mm, sizeof(MetacognitionMetrics),
                           "MetacognitionMetrics")) {
    return false;
  }

  float *metrics[] = {&mm->confidence_level, &mm->adaptation_rate,
                      &mm->cognitive_load, &mm->error_awareness,
                      &mm->context_relevance};
  const char *names[] = {"confidence", "adaptation", "cognitive_load",
                         "error_awareness", "context"};

  for (int i = 0; i < 5; i++) {
    if (isnan(*metrics[i]) || isinf(*metrics[i])) {
      fprintf(stderr, "WARNING: Invalid %s metric, resetting to 0.5\n",
              names[i]);
      *metrics[i] = 0.5f;
    }
  }

  return true;
}

bool validateMetaLearning(MetaLearningState *mls) {
  if (!validateMemoryBlock(mls, sizeof(MetaLearningState),
                           "MetaLearningState")) {
    return false;
  }

  if (mls->priority_weights != NULL) {
    if (!isValidMemoryRegion(mls->priority_weights, sizeof(float))) {
      mls->priority_weights = NULL;
      fprintf(stderr, "WARNING: Corrupted priority weights\n");
    }
  }

  return true;
}

bool validateSocialSystem(SocialSystem *ss) {
  if (!validateMemoryBlock(ss, sizeof(SocialSystem), "SocialSystem")) {
    return false;
  }

  if (ss->interactions != NULL && ss->max_interactions > 0) {
    if (!validateMemoryBlock(ss->interactions,
                             ss->max_interactions * sizeof(SocialInteraction),
                             "SocialSystem->interactions")) {
      ss->interactions = NULL;
      ss->interaction_count = 0;
      fprintf(stderr, "WARNING: Corrupted social interactions\n");
    }
  }

  if (ss->person_models != NULL && ss->max_models > 0) {
    if (!validateMemoryBlock(ss->person_models,
                             ss->max_models * sizeof(PersonModel),
                             "SocialSystem->person_models")) {
      ss->person_models = NULL;
      ss->model_count = 0;
      fprintf(stderr, "WARNING: Corrupted person models\n");
    }
  }

  return true;
}

bool validateGoalSystem(GoalSystem *gs) {
  if (!validateMemoryBlock(gs, sizeof(GoalSystem), "GoalSystem")) {
    return false;
  }

  if (gs->goals != NULL && gs->capacity > 0) {
    if (!validateMemoryBlock(gs->goals, gs->capacity * sizeof(Goal),
                             "GoalSystem->goals")) {
      gs->goals = NULL;
      gs->num_goals = 0;
      fprintf(stderr, "WARNING: Corrupted goals\n");
    }
  }

  if (gs->num_goals > gs->capacity) {
    fprintf(stderr, "WARNING: Goal count exceeds capacity, correcting\n");
    gs->num_goals = gs->capacity;
  }

  return true;
}

bool validateContextManager(GlobalContextManager *gcm) {
  if (!validateMemoryBlock(gcm, sizeof(GlobalContextManager),
                           "GlobalContextManager")) {
    return false;
  }

  if (gcm->global_context_vector != NULL && gcm->vector_size > 0) {
    if (!validateMemoryBlock(gcm->global_context_vector,
                             gcm->vector_size * sizeof(float),
                             "GlobalContextManager->global_context_vector")) {
      gcm->global_context_vector = NULL;
      fprintf(stderr, "WARNING: Corrupted global context vector\n");
    }
  }

  return true;
}

bool validateEmotionalSystem(EmotionalSystem *es) {
  if (!validateMemoryBlock(es, sizeof(EmotionalSystem), "EmotionalSystem")) {
    return false;
  }

  for (int i = 0; i < MAX_EMOTION_TYPES; i++) {
    EmotionState *emotion = &es->emotions[i];
    if (isnan(emotion->intensity) || isinf(emotion->intensity) ||
        emotion->intensity < 0.0f || emotion->intensity > 1.0f) {
      fprintf(stderr, "WARNING: Invalid emotion %d intensity, resetting\n", i);
      emotion->intensity = 0.0f;
    }
  }

  return true;
}

bool validateImaginationSystem(ImaginationSystem *is) {
  if (!validateMemoryBlock(is, sizeof(ImaginationSystem),
                           "ImaginationSystem")) {
    return false;
  }

  if (is->num_scenarios > MAX_SCENARIOS) {
    fprintf(stderr, "WARNING: Scenario count exceeds maximum, correcting\n");
    is->num_scenarios = MAX_SCENARIOS;
  }

  if (is->current_scenario >= is->num_scenarios && is->num_scenarios > 0) {
    fprintf(stderr, "WARNING: Current scenario index invalid, correcting\n");
    is->current_scenario = 0;
  }

  return true;
}

bool validateSpecializationSystem(NeuronSpecializationSystem *nss) {
  if (!validateMemoryBlock(nss, sizeof(NeuronSpecializationSystem),
                           "NeuronSpecializationSystem")) {
    return false;
  }

  if (nss->count > MAX_SPECIALIZED_NEURONS) {
    fprintf(stderr,
            "WARNING: Specialized neuron count exceeds maximum, correcting\n");
    nss->count = MAX_SPECIALIZED_NEURONS;
  }

  return true;
}

bool validateMoralCompass(MoralCompass *mc) {
  if (!validateMemoryBlock(mc, sizeof(MoralCompass), "MoralCompass")) {
    return false;
  }

  if (mc->principles != NULL && mc->num_principles > 0) {
    if (!validateMemoryBlock(mc->principles,
                             mc->num_principles * sizeof(EthicalPrinciple),
                             "MoralCompass->principles")) {
      mc->principles = NULL;
      mc->num_principles = 0;
      fprintf(stderr, "WARNING: Corrupted ethical principles\n");
    }
  }

  if (isnan(mc->overall_alignment) || isinf(mc->overall_alignment) ||
      mc->overall_alignment < 0.0f || mc->overall_alignment > 1.0f) {
    fprintf(stderr, "WARNING: Invalid overall alignment, resetting to 0.5\n");
    mc->overall_alignment = 0.5f;
  }

  return true;
}

// Enhanced memory cluster checker with recovery
bool checkMemoryCluster(MemoryCluster *cluster, const char *name) {
  if (cluster == NULL) {
    fprintf(stderr, "WARNING: %s memory cluster is NULL\n", name);
    return false;
  }

  if (!validateMemoryBlock(cluster, sizeof(MemoryCluster), name)) {
    fprintf(stderr, "CRITICAL: %s memory cluster structure corrupted\n", name);
    return false;
  }

  if (cluster->entries == NULL && cluster->capacity > 0) {
    fprintf(
        stderr,
        "WARNING: %s memory entries is NULL but capacity > 0, reallocating\n",
        name);
    cluster->entries =
        (MemoryEntry *)calloc(cluster->capacity, sizeof(MemoryEntry));
    cluster->size = 0;
  }

  if (cluster->entries != NULL && cluster->capacity > 0) {
    if (!validateMemoryBlock(cluster->entries,
                             cluster->capacity * sizeof(MemoryEntry), name)) {
      fprintf(stderr, "WARNING: %s memory entries corrupted, reallocating\n",
              name);
      free(cluster->entries);
      cluster->entries =
          (MemoryEntry *)calloc(cluster->capacity, sizeof(MemoryEntry));
      cluster->size = 0;
    }
  }

  if (cluster->size > cluster->capacity) {
    fprintf(stderr,
            "WARNING: %s memory size exceeds capacity (%u > %u), correcting\n",
            name, cluster->size, cluster->capacity);
    cluster->size = cluster->capacity;
  }

  if (isnan(cluster->importance_threshold) ||
      isinf(cluster->importance_threshold) ||
      cluster->importance_threshold < 0.0f ||
      cluster->importance_threshold > 1.0f) {
    fprintf(stderr,
            "WARNING: %s importance threshold invalid, resetting to 0.5\n",
            name);
    cluster->importance_threshold = 0.5f;
  }

  if (cluster->entries != NULL && cluster->size > 0) {
    for (unsigned int i = 0; i < cluster->size && i < cluster->capacity; i++) {
      MemoryEntry *entry = &cluster->entries[i];

      if (isnan(entry->importance) || isinf(entry->importance)) {
        fprintf(stderr,
                "WARNING: %s entry %u has invalid importance, resetting\n",
                name, i);
        entry->importance = 0.0f;
      }

      for (int j = 0; j < MEMORY_VECTOR_SIZE; j++) {
        if (isnan(entry->vector[j]) || isinf(entry->vector[j])) {
          fprintf(stderr,
                  "WARNING: %s entry %u vector[%d] invalid, resetting\n", name,
                  i, j);
          entry->vector[j] = 0.0f;
        }
      }
    }
  }
  return true;
}

// Comprehensive system component checker
bool checkSystemComponent(void *component, const char *name,
                          size_t expected_size) {
  if (component == NULL) {
    fprintf(stderr, "WARNING: %s system is NULL, attempting recovery\n", name);
    return false;
  }

  if (!validateMemoryBlock(component, expected_size, name)) {
    fprintf(stderr,
            "CRITICAL: %s system has corrupted memory, attempting recovery\n",
            name);
    return false;
  }

  if (strcmp(name, "Working Memory") == 0) {
    return validateWorkingMemory((WorkingMemorySystem *)component);
  } else if (strcmp(name, "Meta Controller") == 0) {
    return validateMetaController((MetaController *)component);
  } else if (strcmp(name, "Performance Metrics") == 0) {
    return validatePerformanceMetrics((NetworkPerformanceMetrics *)component);
  } else if (strcmp(name, "Motivation System") == 0) {
    return validateMotivationSystem((IntrinsicMotivation *)component);
  } else if (strcmp(name, "Reflection Parameters") == 0) {
    return validateReflectionParameters((ReflectionParameters *)component);
  } else if (strcmp(name, "Identity System") == 0) {
    return validateIdentitySystem((SelfIdentitySystem *)component);
  } else if (strcmp(name, "Knowledge Filter") == 0) {
    return validateKnowledgeFilter((KnowledgeFilter *)component);
  } else if (strcmp(name, "Metacognition") == 0) {
    return validateMetacognition((MetacognitionMetrics *)component);
  } else if (strcmp(name, "Meta Learning") == 0) {
    return validateMetaLearning((MetaLearningState *)component);
  } else if (strcmp(name, "Social System") == 0) {
    return validateSocialSystem((SocialSystem *)component);
  } else if (strcmp(name, "Goal System") == 0) {
    return validateGoalSystem((GoalSystem *)component);
  } else if (strcmp(name, "Context Manager") == 0) {
    return validateContextManager((GlobalContextManager *)component);
  } else if (strcmp(name, "Emotional System") == 0) {
    return validateEmotionalSystem((EmotionalSystem *)component);
  } else if (strcmp(name, "Imagination System") == 0) {
    return validateImaginationSystem((ImaginationSystem *)component);
  } else if (strcmp(name, "Specialization System") == 0) {
    return validateSpecializationSystem(
        (NeuronSpecializationSystem *)component);
  } else if (strcmp(name, "Moral Compass") == 0) {
    return validateMoralCompass((MoralCompass *)component);
  }

  return true;
}

// Enhanced memory usage checker with detailed reporting
bool checkMemoryUsage() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    fprintf(stderr, "WARNING: Failed to get memory usage statistics\n");
    return false;
  }

  long memory_mb = usage.ru_maxrss / 1024;

  if (memory_mb > 1000) {
    fprintf(stderr, "CRITICAL: Very high memory usage detected (%ld MB)\n",
            memory_mb);
  } else if (memory_mb > 500) {
    fprintf(stderr, "WARNING: High memory usage detected (%ld MB)\n",
            memory_mb);
  } else if (memory_mb > 200) {
    fprintf(stderr, "INFO: Moderate memory usage (%ld MB)\n", memory_mb);
  }

  static long previous_page_faults = 0;
  long current_page_faults = usage.ru_majflt + usage.ru_minflt;

  if (previous_page_faults > 0) {
    long fault_increase = current_page_faults - previous_page_faults;
    if (fault_increase > 1000) {
      fprintf(stderr,
              "WARNING: Excessive page faults detected (%ld new faults)\n",
              fault_increase);
    }
  }

  previous_page_faults = current_page_faults;
  return true;
}

// Log current system state for debugging
void logSystemState() {
  fprintf(stderr, "Logging current system state...\n");

  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    fprintf(stderr, "Memory usage: %ld KB\n", usage.ru_maxrss);
    fprintf(stderr, "Page faults: %ld major, %ld minor\n", usage.ru_majflt,
            usage.ru_minflt);
  }

  time_t current_time = time(NULL);
  fprintf(stderr, "Timestamp: %s", ctime(&current_time));
}

// Emergency backup function
void saveEmergencyBackup() {
  fprintf(stderr, "Saving emergency backup...\n");

  time_t now = time(NULL);
  char backup_filename[256];
  snprintf(backup_filename, sizeof(backup_filename), "emergency_backup_%ld.dat",
           now);

  FILE *backup_file = fopen(backup_filename, "wb");
  if (backup_file != NULL) {
    fwrite(&now, sizeof(time_t), 1, backup_file);
    fclose(backup_file);
    fprintf(stderr, "Emergency backup saved to %s\n", backup_filename);
  } else {
    fprintf(stderr, "Failed to create emergency backup file\n");
  }
}

// System stabilization function
void stabilizeSystem() {
  fprintf(stderr, "Attempting system stabilization...\n");

  segfault_occurred = false;
  fault_address = NULL;
  memset(fault_description, 0, sizeof(fault_description));

  sync();

  usleep(100000);

  fprintf(stderr, "System stabilization completed\n");
}

// System recovery function for critical failures
void attemptSystemRecovery(const char *failure_description) {
  fprintf(stderr, "\n=== SYSTEM RECOVERY INITIATED ===\n");
  fprintf(stderr, "Failure: %s\n", failure_description);

  logSystemState();

  saveEmergencyBackup();

  stabilizeSystem();

  fprintf(stderr, "=== RECOVERY ATTEMPT COMPLETED ===\n");
}

// Enhanced memory region validator with detailed analysis
bool validateMemoryRegionDetailed(void *ptr, size_t size,
                                  const char *region_name) {
  if (ptr == NULL) {
    fprintf(stderr, "ERROR: %s - NULL pointer\n", region_name);
    return false;
  }

  if (size == 0) {
    fprintf(stderr, "WARNING: %s - Zero size region\n", region_name);
    return false;
  }

  uintptr_t addr = (uintptr_t)ptr;
  if (addr < 4096) {
    fprintf(stderr, "ERROR: %s - Suspicious low address %p\n", region_name,
            ptr);
    return false;
  }

  volatile bool test_passed = false;
  if (setjmp(segfault_recovery) == 0) {
    volatile char test = *((volatile char *)ptr);
    test = *((volatile char *)ptr + size - 1);
    if (size > 2) {
      test = *((volatile char *)ptr + size / 2);
    }
    test_passed = true;
    (void)test;
  } else {
    fprintf(stderr, "ERROR: %s - Memory access violation at %p (size: %zu)\n",
            region_name, ptr, size);
    return false;
  }

  return test_passed;
}

// Floating point exception handler
void fpe_handler(int sig, siginfo_t *si, void *unused) {
  fprintf(stderr, "FLOATING POINT EXCEPTION: Signal %d at address %p\n", sig,
          si->si_addr);

  switch (si->si_code) {
  case FPE_INTDIV:
    fprintf(stderr, "Integer divide by zero\n");
    break;
  case FPE_INTOVF:
    fprintf(stderr, "Integer overflow\n");
    break;
  case FPE_FLTDIV:
    fprintf(stderr, "Floating point divide by zero\n");
    break;
  case FPE_FLTOVF:
    fprintf(stderr, "Floating point overflow\n");
    break;
  case FPE_FLTUND:
    fprintf(stderr, "Floating point underflow\n");
    break;
  case FPE_FLTRES:
    fprintf(stderr, "Floating point inexact result\n");
    break;
  case FPE_FLTINV:
    fprintf(stderr, "Floating point invalid operation\n");
    break;
  default:
    fprintf(stderr, "Unknown floating point exception\n");
    break;
  }

  longjmp(segfault_recovery, 2);
}

// Signal handler setup with enhanced error reporting
void setupEnhancedSignalHandlers() {
  struct sigaction sa_segv;
  memset(&sa_segv, 0, sizeof(sa_segv));
  sa_segv.sa_sigaction = segfault_handler;
  sa_segv.sa_flags = SA_SIGINFO | SA_RESTART;
  sigemptyset(&sa_segv.sa_mask);

  if (sigaction(SIGSEGV, &sa_segv, NULL) == -1) {
    fprintf(stderr, "WARNING: Failed to install SIGSEGV handler: %s\n",
            strerror(errno));
  }

  if (sigaction(SIGBUS, &sa_segv, NULL) == -1) {
    fprintf(stderr, "WARNING: Failed to install SIGBUS handler: %s\n",
            strerror(errno));
  }

  struct sigaction sa_fpe;
  memset(&sa_fpe, 0, sizeof(sa_fpe));
  sa_fpe.sa_sigaction = fpe_handler;
  sa_fpe.sa_flags = SA_SIGINFO | SA_RESTART;
  sigemptyset(&sa_fpe.sa_mask);

  if (sigaction(SIGFPE, &sa_fpe, NULL) == -1) {
    fprintf(stderr, "WARNING: Failed to install SIGFPE handler: %s\n",
            strerror(errno));
  }
}

typedef struct {
  time_t start_time;
  unsigned long total_checks;
  unsigned long successful_checks;
  unsigned long failed_checks;
  unsigned long segfaults_recovered;
  unsigned long fpe_recovered;
  float average_check_time;
  float min_check_time;
  float max_check_time;
  float total_check_time;
  unsigned long component_failures;
  unsigned long memory_issues;
  unsigned long instability_events;
  unsigned long critical_failures;
  unsigned long neuron_corrections;
  unsigned long connection_corrections;
  unsigned long weight_corrections;
  unsigned long memory_reinitializations;
  unsigned long memory_cluster_errors;
} SystemHealthMetrics;

static SystemHealthMetrics health_metrics = {0};

void initializeSystemHealthMonitor() {
  health_metrics.start_time = time(NULL);
  health_metrics.total_checks = 0;
  health_metrics.successful_checks = 0;
  health_metrics.failed_checks = 0;
  health_metrics.segfaults_recovered = 0;
  health_metrics.fpe_recovered = 0;
  health_metrics.average_check_time = 0.0f;
  health_metrics.min_check_time = FLT_MAX;
  health_metrics.max_check_time = 0.0f;
  health_metrics.total_check_time = 0.0f;
  health_metrics.component_failures = 0;
  health_metrics.memory_issues = 0;
  health_metrics.instability_events = 0;
  health_metrics.critical_failures = 0;
  health_metrics.neuron_corrections = 0;
  health_metrics.connection_corrections = 0;
  health_metrics.weight_corrections = 0;
  health_metrics.memory_reinitializations = 0;
  health_metrics.memory_cluster_errors = 0;

  setupEnhancedSignalHandlers();
}

void updateHealthMetrics(bool check_passed, double check_duration) {
  health_metrics.total_checks++;
  if (!check_passed) {
    health_metrics.failed_checks++;
  }

  health_metrics.average_check_time =
      (health_metrics.average_check_time * (health_metrics.total_checks - 1) +
       check_duration) /
      health_metrics.total_checks;
}

void printSystemHealthReport() {
  time_t current_time = time(NULL);
  double uptime = difftime(current_time, health_metrics.start_time);

  printf("\n=== SYSTEM HEALTH REPORT ===\n");
  printf("Uptime: %.0f seconds\n", uptime);
  printf("Total checks: %lu\n", health_metrics.total_checks);
  printf(
      "Failed checks: %lu (%.2f%%)\n", health_metrics.failed_checks,
      health_metrics.total_checks > 0
          ? (100.0 * health_metrics.failed_checks / health_metrics.total_checks)
          : 0.0);
  printf("Segfaults recovered: %lu\n", health_metrics.segfaults_recovered);
  printf("FPE recovered: %lu\n", health_metrics.fpe_recovered);
  printf("Average check time: %.6f seconds\n",
         health_metrics.average_check_time);
  printf("============================\n");
}

void systemFallbackCheck(
    Neuron *neurons, int *connections, float *weights, int *reverse_connections,
    float *reverse_weights, MemorySystem *memorySystem,
    NetworkStateSnapshot *stateHistory, PerformanceMetrics *performance_history,
    float *input_tensor, float *target_outputs, float *previous_outputs,
    SystemParameters *system_params, WorkingMemorySystem *working_memory,
    MetaController *metaController,
    NetworkPerformanceMetrics *performanceMetrics,
    IntrinsicMotivation *motivation, ReflectionParameters *reflection_params,
    SelfIdentitySystem *identity_system, KnowledgeFilter *knowledge_filter,
    MetacognitionMetrics *metacognition, MetaLearningState *meta_learning_state,
    SocialSystem *social_system, GoalSystem *goalSystem,
    GlobalContextManager *contextManager, EmotionalSystem *emotional_system,
    ImaginationSystem *imagination_system,
    NeuronSpecializationSystem *specialization_system,
    MoralCompass *moralCompass, int step, int max_neurons, int max_connections,
    int input_size) {
  clock_t start_time = clock();
  bool check_passed = true;

  static bool health_monitor_initialized = false;
  if (!health_monitor_initialized) {
    initializeSystemHealthMonitor();
    initializeSegfaultProtection();
    health_monitor_initialized = true;
  }

  int recovery_code = setjmp(segfault_recovery);
  if (recovery_code == 1) {
    health_metrics.segfaults_recovered++;
    check_passed = false;
    fprintf(stderr, "RECOVERED FROM SEGFAULT: %s at address %p\n",
            fault_description, fault_address);

    if (fault_address != NULL) {
      fprintf(stderr, "Attempting emergency recovery for address %p\n",
              fault_address);
    }

    segfault_occurred = false;
    goto health_update;
  } else if (recovery_code == 2) {
    health_metrics.fpe_recovered++;
    check_passed = false;
    fprintf(stderr,
            "RECOVERED FROM FPE: Continuing with numerical corrections\n");
    segfault_occurred = false;
    goto health_update;
  }

  if (recovery_code == 0) {
    if (!validateMemoryBlock(neurons, max_neurons * sizeof(Neuron),
                             "Neurons")) {
      fprintf(stderr, "CRITICAL: Neuron array corrupted, system unstable\n");
      check_passed = false;
      health_metrics.critical_failures++;
      exit(EXIT_FAILURE);
    }

    if (!validateMemoryBlock(connections,
                             max_neurons * max_connections * sizeof(int),
                             "Connections")) {
      fprintf(stderr,
              "CRITICAL: Connection array corrupted, system unstable\n");
      check_passed = false;
      health_metrics.critical_failures++;
      exit(EXIT_FAILURE);
    }

    if (!validateMemoryBlock(weights,
                             max_neurons * max_connections * sizeof(float),
                             "Weights")) {
      fprintf(stderr, "CRITICAL: Weight array corrupted, system unstable\n");
      check_passed = false;
      health_metrics.critical_failures++;
      exit(EXIT_FAILURE);
    }

    int neuron_corrections = 0;
    for (int i = 0; i < max_neurons; i++) {
      if (isnan(neurons[i].state) || isinf(neurons[i].state)) {
        fprintf(stderr,
                "WARNING: Neuron %d has invalid state value, resetting\n", i);
        neurons[i].state = 0.0f;
        neuron_corrections++;
        check_passed = false;
      }
      if (isnan(neurons[i].output) || isinf(neurons[i].output)) {
        fprintf(stderr,
                "WARNING: Neuron %d has invalid output value, resetting\n", i);
        neurons[i].output = 0.0f;
        neuron_corrections++;
        check_passed = false;
      }
    }
    health_metrics.neuron_corrections += neuron_corrections;

    int connection_corrections = 0;
    int weight_corrections = 0;
    for (int i = 0; i < max_neurons * max_connections; i++) {
      if (connections[i] < 0 || connections[i] >= max_neurons) {
        fprintf(
            stderr,
            "WARNING: Invalid connection index %d at position %d, resetting\n",
            connections[i], i);
        connections[i] = i % max_neurons;
        connection_corrections++;
        check_passed = false;
      }
      if (isnan(weights[i]) || isinf(weights[i])) {
        fprintf(stderr, "WARNING: Invalid weight at position %d, resetting\n",
                i);
        weights[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        weight_corrections++;
        check_passed = false;
      }
    }
    health_metrics.connection_corrections += connection_corrections;
    health_metrics.weight_corrections += weight_corrections;

    if (memorySystem != NULL) {
      if (!validateMemoryBlock(memorySystem, sizeof(MemorySystem),
                               "MemorySystem")) {
        fprintf(stderr, "WARNING: Memory system corrupted, reinitializing\n");
        check_passed = false;
        health_metrics.memory_reinitializations++;
      } else {
        if (memorySystem->entries == NULL && memorySystem->capacity > 0) {
          fprintf(stderr,
                  "WARNING: Memory entries array is NULL, reinitializing\n");
          memorySystem->entries = (MemoryEntry *)calloc(memorySystem->capacity,
                                                        sizeof(MemoryEntry));
          if (memorySystem->entries == NULL) {
            fprintf(stderr,
                    "CRITICAL: Failed to allocate memory for entries\n");
            check_passed = false;
            health_metrics.critical_failures++;
          } else {
            memorySystem->size = 0;
            memorySystem->head = 0;
            health_metrics.memory_reinitializations++;
          }
        }

        if (!checkMemoryCluster(&memorySystem->hierarchy.short_term,
                                "Short-term")) {
          check_passed = false;
          health_metrics.memory_cluster_errors++;
        }
        if (!checkMemoryCluster(&memorySystem->hierarchy.medium_term,
                                "Medium-term")) {
          check_passed = false;
          health_metrics.memory_cluster_errors++;
        }
        if (!checkMemoryCluster(&memorySystem->hierarchy.long_term,
                                "Long-term")) {
          check_passed = false;
          health_metrics.memory_cluster_errors++;
        }
      }
    }

    bool system_stable = true;

    system_stable &= checkSystemComponent(working_memory, "Working Memory",
                                          sizeof(WorkingMemorySystem));
    system_stable &= checkSystemComponent(metaController, "Meta Controller",
                                          sizeof(MetaController));
    system_stable &=
        checkSystemComponent(performanceMetrics, "Performance Metrics",
                             sizeof(NetworkPerformanceMetrics));
    system_stable &= checkSystemComponent(motivation, "Motivation System",
                                          sizeof(IntrinsicMotivation));
    system_stable &=
        checkSystemComponent(reflection_params, "Reflection Parameters",
                             sizeof(ReflectionParameters));
    system_stable &= checkSystemComponent(identity_system, "Identity System",
                                          sizeof(SelfIdentitySystem));
    system_stable &= checkSystemComponent(knowledge_filter, "Knowledge Filter",
                                          sizeof(KnowledgeFilter));
    system_stable &= checkSystemComponent(metacognition, "Metacognition",
                                          sizeof(MetacognitionMetrics));
    system_stable &= checkSystemComponent(meta_learning_state, "Meta Learning",
                                          sizeof(MetaLearningState));
    system_stable &= checkSystemComponent(social_system, "Social System",
                                          sizeof(SocialSystem));
    system_stable &=
        checkSystemComponent(goalSystem, "Goal System", sizeof(GoalSystem));
    system_stable &= checkSystemComponent(contextManager, "Context Manager",
                                          sizeof(GlobalContextManager));
    system_stable &= checkSystemComponent(emotional_system, "Emotional System",
                                          sizeof(EmotionalSystem));
    system_stable &= checkSystemComponent(
        imagination_system, "Imagination System", sizeof(ImaginationSystem));
    system_stable &=
        checkSystemComponent(specialization_system, "Specialization System",
                             sizeof(NeuronSpecializationSystem));
    system_stable &= checkSystemComponent(moralCompass, "Moral Compass",
                                          sizeof(MoralCompass));

    if (!system_stable) {
      check_passed = false;
      health_metrics.component_failures++;
    }

    if (!checkMemoryUsage()) {
      check_passed = false;
      health_metrics.memory_issues++;
    }

    if (step % 100 == 0) {
      const char *status =
          (check_passed && system_stable) ? "STABLE" : "UNSTABLE";
      printf("\nEnhanced Fallback System Check (Step %d): %s - All critical "
             "structures validated\n",
             step, status);

      if (!check_passed || !system_stable) {
        fprintf(stderr, "WARNING: System instability detected at step %d\n",
                step);
        health_metrics.instability_events++;
      }
    }
  }

health_update: {
  clock_t end_time = clock();
  double check_duration = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
  updateHealthMetrics(check_passed, check_duration);

  health_metrics.total_checks++;
  if (check_passed) {
    health_metrics.successful_checks++;
  }

  if (check_duration > health_metrics.max_check_time) {
    health_metrics.max_check_time = check_duration;
  }
  if (check_duration < health_metrics.min_check_time ||
      health_metrics.min_check_time == 0.0) {
    health_metrics.min_check_time = check_duration;
  }

  if (step % 1000 == 0) {
    printSystemHealthReport();

    printf("System Health Summary (Step %d):\n", step);
    printf("  Success Rate: %.2f%% (%lu/%lu)\n",
           (double)health_metrics.successful_checks /
               health_metrics.total_checks * 100.0,
           health_metrics.successful_checks, health_metrics.total_checks);
    printf("  Average Check Time: %.6f seconds\n",
           health_metrics.total_check_time / health_metrics.total_checks);
    printf("  Corrections Made: Neurons=%lu, Connections=%lu, Weights=%lu\n",
           health_metrics.neuron_corrections,
           health_metrics.connection_corrections,
           health_metrics.weight_corrections);
    printf(
        "  Critical Events: Segfaults=%lu, FPEs=%lu, Critical Failures=%lu\n",
        health_metrics.segfaults_recovered, health_metrics.fpe_recovered,
        health_metrics.critical_failures);
  }

  segfault_occurred = false;
}
}

void freeGoalSystem(GoalSystem *system) {
  if (system) {
    if (system->goals) {
      free(system->goals);
    }
    free(system);
  }
}

void freeMetaController(MetaController *controller) {
  if (controller) {
    if (controller->region_importance_scores) {
      free(controller->region_importance_scores);
    }
    if (controller->adaptation_rates) {
      free(controller->adaptation_rates);
    }
    free(controller);
  }
}

void freeIntrinsicMotivation(IntrinsicMotivation *motivation) {
  if (motivation) {
    free(motivation);
  }
}

void freeNetworkPerformanceMetrics(NetworkPerformanceMetrics *metrics) {
  if (metrics) {
    if (metrics->region_performance_scores) {
      free(metrics->region_performance_scores);
    }
    if (metrics->region_error_rates) {
      free(metrics->region_error_rates);
    }
    free(metrics);
  }
}

void freeReflectionParameters(ReflectionParameters *params) {
  if (params) {
    if (params->reflection_history) {
      free(params->reflection_history);
    }
    free(params);
  }
}

void freeSelfIdentitySystem(SelfIdentitySystem *system) {
  if (system) {
    if (system->core_values) {
      free(system->core_values);
    }
    if (system->beliefs) {
      free(system->beliefs);
    }
    if (system->identity_markers) {
      free(system->identity_markers);
    }
    if (system->temporal_continuity) {
      free(system->temporal_continuity);
    }
    if (system->behavioral_patterns) {
      free(system->behavioral_patterns);
    }
    free(system);
  }
}

void freeKnowledgeFilter(KnowledgeFilter *filter) {
  if (filter) {
    if (filter->categories) {
      free(filter->categories);
    }
    if (filter->knowledge_graph) {
      free(filter->knowledge_graph);
    }
    free(filter);
  }
}

void freeMetacognitionMetrics(MetacognitionMetrics *metrics) {
  if (metrics) {
    if (metrics->confidence_history) {
      free(metrics->confidence_history);
    }
    if (metrics->uncertainty_history) {
      free(metrics->uncertainty_history);
    }
    free(metrics);
  }
}

void freeMetaLearningState(MetaLearningState *state) {
  if (state) {
    if (state->strategy_history) {
      free(state->strategy_history);
    }
    if (state->performance_per_strategy) {
      free(state->performance_per_strategy);
    }
    if (state->adaptation_rates) {
      free(state->adaptation_rates);
    }
    free(state);
  }
}

void freeWorkingMemorySystem(WorkingMemorySystem *system) {
  if (system) {
    if (system->buffer) {
      free(system->buffer);
    }
    if (system->attention_weights) {
      free(system->attention_weights);
    }
    if (system->priority_scores) {
      free(system->priority_scores);
    }
    free(system);
  }
}

int main() {
  loadVocabularyFromFile("vocabulary.txt");

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
      neurons[i].num_connections = MAX_CONNECTIONS;
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

  const char *text_input =
      "Apple, banana, cherry, date, and elderberry are fruits.";
  initializeEmbeddings("custom_embeddings.txt");

  int network_regions = 2; // Assuming 2 layers

  // Load or initialize systems
  MetaController *metaController = loadMetaController("metacontroller.dat");
  if (metaController == NULL) {
    metaController = initializeMetaController(network_regions);
    printf("Initialized new MetaController\n");
  }

  IntrinsicMotivation *motivation = loadIntrinsicMotivation("motivation.dat");
  if (motivation == NULL) {
    motivation = initializeMotivationSystem();
    printf("Initialized new IntrinsicMotivation system\n");
  }

  NetworkPerformanceMetrics *performanceMetrics =
      loadNetworkPerformanceMetrics("performance_metrics.dat");
  if (performanceMetrics == NULL) {
    performanceMetrics = initializePerformanceMetrics(network_regions);
    printf("Initialized new NetworkPerformanceMetrics\n");
  }

  ReflectionParameters *reflection_params =
      loadReflectionParameters("reflection_params.dat");
  if (reflection_params == NULL) {
    reflection_params = initializeReflectionParameters();
    printf("Initialized new ReflectionParameters\n");
  }

  SelfIdentitySystem *identity_system =
      loadSelfIdentitySystem("identity_system.dat");
  if (identity_system == NULL) {
    identity_system = initializeSelfIdentity(100, 200, 50, 1000, 100);
    printf("Initialized new SelfIdentitySystem\n");
  }

  KnowledgeFilter *knowledge_filter =
      loadKnowledgeFilter("knowledge_filter.dat");
  if (knowledge_filter == NULL) {
    knowledge_filter = initializeKnowledgeFilter(100);
    printf("Initialized new KnowledgeFilter\n");
  }

  MetacognitionMetrics *metacognition =
      loadMetacognitionMetrics("metacognition.dat");
  if (metacognition == NULL) {
    metacognition = initializeMetacognitionMetrics();
    printf("Initialized new MetacognitionMetrics\n");
  }

  initializeKnowledgeMetrics(knowledge_filter);

  MetaLearningState *meta_learning_state =
      loadMetaLearningState("meta_learning.dat");
  if (meta_learning_state == NULL) {
    meta_learning_state = initializeMetaLearningState(4);
    printf("Initialized new MetaLearningState\n");
  }

  // Initialize remaining systems
  SocialSystem *social_system = social_system = initializeSocialSystem(100, 50);
  GoalSystem *goalSystem = initializeGoalSystem(10);
  GlobalContextManager *contextManager =
      initializeGlobalContextManager(MAX_NEURONS);
  EmotionalSystem *emotional_system = initializeEmotionalSystem();
  ImaginationSystem *imagination_system =
      initializeImaginationSystem(0.6f, 0.7f);
  NeuronSpecializationSystem *specialization_system =
      initializeSpecializationSystem(0.6f);
  MoralCompass *moralCompass = initializeMoralCompass(5);

  addSymbol(0, "What is the current task?");
  addSymbol(1, "What is the current error rate?");
  addSymbol(2, "What is the current learning rate?");
  addSymbol(3, "What is the current memory usage?");

  int q0[] = {0};
  int q1[] = {1};
  int q2[] = {2};
  int q3[] = {3};

  addQuestion(0, q0, 1);
  addQuestion(1, q1, 1);
  addQuestion(2, q2, 1);
  addQuestion(3, q3, 1);

  addGoal(goalSystem, "Minimize prediction error", 1.0f);
  addGoal(goalSystem, "Develop stable representations", 0.8f);
  addGoal(goalSystem, "Maximize information gain", 0.7f);

  printf("Ethical framework initialized with %d principles\n",
         moralCompass->num_principles);
  printf("Initial ethical alignment: %.2f\n", moralCompass->overall_alignment);

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

    initPredictiveCodingParams(max_neurons);
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
    cudaMemcpy(d_recurrent_weights, weights, max_neurons * sizeof(float),
               cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    dim3 blockDim(256);
    dim3 gridDim((max_neurons + blockDim.x - 1) / blockDim.x);

    int activation_type = ACTIVATION_TANH;

    update_neurons<<<gridDim, blockDim>>>(
        d_neurons, d_weights, d_connections, max_neurons, max_connections,
        d_input_tensor, input_size, d_recurrent_weights, activation_type);

    cudaMemcpy(neurons, d_neurons, max_neurons * sizeof(Neuron),
               cudaMemcpyDeviceToHost);

    computePredictionErrors(neurons, input_tensor, max_neurons);

    float *target_outputs = (float *)malloc(max_neurons * sizeof(float));
    target_outputs =
        generatePotentialTargets(max_neurons, previous_outputs, stateHistory,
                                 step, relevantMemory, params);

    float word_feedback[EMBEDDING_SIZE];
    computeGradientFeedback(word_feedback, neurons, target_outputs,
                            max_neurons);
    char *tokens[INPUT_SIZE];
    int num_tokens = 0;
    tokenizeString(text_input, tokens, &num_tokens);
    for (int i = 0; i < num_tokens; i++) {
      updateEmbeddings(word_feedback, tokens[i]);
    }

    // Decision path selection
    selectOptimalDecisionPath(neurons, weights, connections, input_tensor,
                              MAX_NEURONS, previous_outputs, stateHistory, step,
                              relevantMemory, params);

    if (imagination_system->active) {
      float influence = applyImaginationToDecision(imagination_system, neurons,
                                                   input_tensor, max_neurons);

      if (step % 5 == 0) {
        printf("Applied imagination with influence: %.2f%%\n",
               influence * 100.0f);
      }

      // Record divergence history
      int history_idx = step % 100;
      imagination_system->divergence_history[history_idx] =
          imagination_system->scenarios[imagination_system->current_scenario]
              .divergence_factor;

      // Increase steps simulated
      imagination_system->steps_simulated++;

      // Deactivate after some steps
      if (imagination_system->steps_simulated > 20) {
        imagination_system->active = false;
        imagination_system->steps_simulated = 0;
        printf("Deactivating imagination after %d steps\n",
               imagination_system->steps_simulated);
      }
    }

    // Update performance metrics
    computeRegionPerformanceMetrics(performanceMetrics, neurons, target_outputs,
                                    MAX_NEURONS);

    // Update meta-controller
    updateMetaControllerPriorities(metaController, performanceMetrics,
                                   metacognition);
    applyMetaControllerAdaptations(neurons, weights, metaController,
                                   MAX_NEURONS);

    SecurityValidationStatus secStatus = validateCriticalSecurity(
        neurons, weights, connections, max_neurons, max_connections);

    if (secStatus.critical_violation) {
      handleCriticalSecurityViolation(neurons, weights, connections,
                                      &secStatus);
    }
    integrateKnowledgeFilter(knowledge_filter, memorySystem, neurons,
                             input_tensor);

    void updateKnowledgeSystem(
        Neuron * neurons, float *input_tensor, MemorySystem *memory_system,
        KnowledgeFilter *filter, float current_performance);

    // Add periodic insights printing
    if (step % 50 == 0) {
      printCategoryInsights(knowledge_filter);
    }

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

    float *d_m, *d_v;
    cudaMalloc(&d_m, max_connections * sizeof(float)); // Moment buffers
    cudaMalloc(&d_v, max_connections * sizeof(float)); // Moment buffers

    // Initialize moment buffers to zero on the device
    cudaMemset(d_m, 0, max_connections * sizeof(float)); // Set to 0
    cudaMemset(d_v, 0, max_connections * sizeof(float)); // Set to 0

    // Allocate memory on the device for the step counter
    unsigned int *d_t;
    cudaMalloc(&d_t, sizeof(unsigned int));
    cudaMemset(d_t, 1,
               sizeof(unsigned int)); // Initialize t to 1 (step counter)

    const int *d_connections_const = (const int *)d_connections;
    backward_kernel<<<gridDim, blockDim>>>(
        d_neurons, d_weights, d_connections_const, max_neurons, max_connections,
        d_target_outputs, d_output_errors, learning_rate);

    // Update weights
    update_weights<<<gridDim, blockDim>>>(d_weights, d_neurons, d_connections,
                                          learning_rate, max_neurons,
                                          max_connections);

    activation_type = ACTIVATION_RELU;

    process_neurons<<<gridDim, blockDim>>>(
        d_neurons, d_weights, d_connections, max_neurons, max_connections,
        d_input_tensor, input_size, d_recurrent_weights, activation_type);

    reverse_process<<<gridDim, blockDim>>>(d_neurons, d_weights, d_connections,
                                           max_neurons, max_connections);

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

    integrateReflectionSystem(neurons, memorySystem, stateHistory, step,
                              weights, connections, reflection_params);

    // Update memory system with new outputs
    addMemory(memorySystem, working_memory, neurons, input_tensor,
              lastTimestamp + step + 1, feature_projection_matrix);

    // Update identity system
    updateIdentity(identity_system, neurons, max_neurons, memorySystem,
                   input_tensor);

    // Periodically verify identity consistency
    if (step % 20 == 0) {
      bool identity_verified = verifyIdentity(identity_system);
      if (!identity_verified) {
        printf("Warning: Identity consistency check failed\n");

        // Analyze the identity system for potential issues
        IdentityAnalysis analysis = analyzeIdentitySystem(identity_system);
        printf("Core Value Conflicts: %d\n", analysis.core_value_conflicts);
        printf("Belief Conflicts: %d\n", analysis.belief_conflicts);
        printf("Marker Conflicts: %d\n", analysis.marker_conflicts);
        printf("Temporal Instability: %.2f\n", analysis.temporal_instability);
        printf("Pattern Deviation: %.2f\n", analysis.pattern_deviation);
        printf("Overall Consistency: %.2f\n", analysis.overall_consistency);
        printf("Confidence Impact: %.2f\n", analysis.confidence_impact);

        // Implement recovery by creating a backup
        SelfIdentityBackup *backup = createIdentityBackup(identity_system);
        if (backup) {
          printf("Identity backup created successfully.\n");

          // Restore from backup if necessary
          restoreIdentityFromBackup(identity_system, backup);
          printf("Identity system restored from backup.\n");

          // Free backup memory after restoration
          freeIdentityBackup(backup);
        } else {
          printf("Error: Failed to create identity backup.\n");
        }
      }

      // Generate and log identity reflection
      char *reflection = generateIdentityReflection(identity_system);
      if (reflection) {
        printf("%s\n", reflection);
        free(reflection);
      } else {
        printf("Error: Failed to generate identity reflection.\n");
      }
    }

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

    if (step % 3 == 0) {
      printf("Memory system size: %u/%u\n", memorySystem->size,
             memorySystem->capacity);
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

    if (step % 15 == 0 ||
        (step > 10 && performance_history[step - 1].error_rate >
                          performance_history[step - 10].error_rate)) {
      printf("\nActivating imagination at step %d\n", step);
      imagination_system->active = true;
      imagination_system->current_scenario = imagination_system->num_scenarios;
      // Create a new scenario
      float divergence =
          0.2f + ((float)rand() / RAND_MAX) * 0.3f; // 0.2-0.5 range
      ImaginationScenario new_scenario =
          createScenario(neurons, memorySystem, max_neurons, divergence);
      // Name the scenario based on current task
      sprintf(imagination_system->current_scenario_name, "Scenario_%d_%s",
              imagination_system->total_scenarios_generated++,
              current_prompt.task_description);
      // Run simulation steps
      simulateScenario(&new_scenario, neurons, input_tensor, max_neurons, 10);
      // Evaluate plausibility
      evaluateScenarioPlausibility(&new_scenario, memorySystem);
      // Add to scenarios collection
      if (imagination_system->num_scenarios < MAX_SCENARIOS) {
        imagination_system->scenarios[imagination_system->num_scenarios] =
            new_scenario;
        imagination_system->current_scenario =
            imagination_system->num_scenarios;
        imagination_system->num_scenarios++;
      } else {
        // Replace least plausible scenario
        int replace_idx = 0;
        float min_plausibility =
            imagination_system->scenarios[0].outcomes[0].plausibility;
        for (int i = 1; i < MAX_SCENARIOS; i++) {
          if (imagination_system->scenarios[i].outcomes[0].plausibility <
              min_plausibility) {
            min_plausibility =
                imagination_system->scenarios[i].outcomes[0].plausibility;
            replace_idx = i;
          }
        }
        imagination_system->scenarios[replace_idx] = new_scenario;
        imagination_system->current_scenario = replace_idx;
      }
    }

    // Apply imagination to decision making if active
    if (imagination_system->active) {
      float influence = applyImaginationToDecision(imagination_system, neurons,
                                                   input_tensor, max_neurons);

      if (step % 5 == 0) {
        printf("Applied imagination with influence: %.2f%%\n",
               influence * 100.0f);
      }

      // Record divergence history
      int history_idx = step % 100;
      imagination_system->divergence_history[history_idx] =
          imagination_system->scenarios[imagination_system->current_scenario]
              .divergence_factor;

      // Increase steps simulated
      imagination_system->steps_simulated++;

      // Deactivate after some steps
      if (imagination_system->steps_simulated > 20) {
        imagination_system->active = false;
        imagination_system->steps_simulated = 0;
        printf("Deactivating imagination after %d steps\n",
               imagination_system->steps_simulated);
      }
    }

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

    if (step > 0) {
      float perf_delta = performance_history[step].average_output -
                         performance_history[step - 1].average_output;
      float novelty = computeNovelty(neurons, *stateHistory, step);

      updateImaginationCreativity(imagination_system, perf_delta, novelty);

      if (step % 20 == 0) {
        printf("\nImagination Creativity: %.2f, Coherence Threshold: %.2f\n",
               imagination_system->creativity_factor,
               imagination_system->coherence_threshold);
      }
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

    float network_performance =
        1.0f - loss; // Convert loss to performance metric
    if (step % 5 == 0) {
      detectSpecializations(specialization_system, neurons, max_neurons,
                            input_tensor, target_outputs, previous_outputs,
                            previous_states);
    }

    applySpecializations(specialization_system, neurons, weights,
                         (int *)connections, max_neurons, max_connections);

    // Update specialization importance (periodically)
    if (step % 10 == 0) {
      updateSpecializationImportance(specialization_system, network_performance,
                                     performance_history->error_rate, neurons);
    }

    // Evaluate and report system effectiveness (periodically)
    if (step % 20 == 0) {
      float effectiveness = evaluateSpecializationEffectiveness(
          specialization_system, network_performance);
      printf("\nSpecialization System Effectiveness: %.2f\n", effectiveness);
      printSpecializationStats(specialization_system);
    }

    // Update dynamic parameters
    updateDynamicParameters(&params, performance_delta, stability,
                            performance_history[step].error_rate);

    float novelty = computeNovelty(neurons, *stateHistory, step);
    float task_difficulty = estimateTaskDifficulty(
        current_prompt, performance_history[step].error_rate);

    updateMotivationSystem(motivation, performance_delta, novelty,
                           task_difficulty);

    updateGoalSystem(goalSystem, neurons, max_neurons, target_outputs,
                     &learning_rate);

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

    // Adapt network with dynamic parameters
    adaptNetworkDynamic(neurons, weights, &params, performance_delta,
                        input_tensor);

    selectOptimalMetaDecisionPath(neurons, weights, connections, input_tensor,
                                  max_neurons, meta_learning_state,
                                  metacognition);

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
    int question_to_ask = 0;
    if (performance_history[step].error_rate > loss) {
      question_to_ask = 1;
    }
    if (learning_rate > learning_rate) {
      question_to_ask = 2;
    }

    if (question_to_ask > 0) {
      askQuestion(question_to_ask, neurons, input_tensor, memorySystem,
                  &learning_rate, stateHistory, contextManager, motivation,
                  goalSystem, working_memory, identity_system, metacognition,
                  knowledge_filter, emotional_system, imagination_system,
                  social_system, feature_projection_matrix);
      adjustBehaviorBasedOnAnswers(
          neurons, input_tensor, memorySystem, &learning_rate,
          &params.input_noise_scale, &params.weight_noise_scale, stateHistory,
          contextManager, motivation, goalSystem, working_memory,
          identity_system, metacognition, &params, meta_learning_state,
          emotional_system, imagination_system, social_system);
    }

    if (step % 50 == 0) {
      askQuestion(0, neurons, input_tensor, memorySystem, &learning_rate,
                  stateHistory, contextManager, motivation, goalSystem,
                  working_memory, identity_system, metacognition,
                  knowledge_filter, emotional_system, imagination_system,
                  social_system,
                  feature_projection_matrix); // What is the current task?
      askQuestion(1, neurons, input_tensor, memorySystem, &learning_rate,
                  stateHistory, contextManager, motivation, goalSystem,
                  working_memory, identity_system, metacognition,
                  knowledge_filter, emotional_system, imagination_system,
                  social_system,
                  feature_projection_matrix); // What is the current error rate?
      askQuestion(
          2, neurons, input_tensor, memorySystem, &learning_rate, stateHistory,
          contextManager, motivation, goalSystem, working_memory,
          identity_system, metacognition, knowledge_filter, emotional_system,
          imagination_system, social_system,
          feature_projection_matrix); // What is the current learning rate?
      askQuestion(
          3, neurons, input_tensor, memorySystem, &learning_rate, stateHistory,
          contextManager, motivation, goalSystem, working_memory,
          identity_system, metacognition, knowledge_filter, emotional_system,
          imagination_system, social_system,
          feature_projection_matrix); // What is the current memory usage?
    }
    if (step % 50 == 0) {
      adjustBehaviorBasedOnAnswers(
          neurons, input_tensor, memorySystem, &learning_rate,
          &params.input_noise_scale, &params.weight_noise_scale, stateHistory,
          contextManager, motivation, goalSystem, working_memory,
          identity_system, metacognition, &params, meta_learning_state,
          emotional_system, imagination_system, social_system);
    }
    updateNeuronsWithPredictiveCoding(neurons, input_tensor, max_neurons,
                                      learning_rate);

    updateEmpathy(social_system, emotional_system);

    float predicted_behavior[5] = {0};
    predictBehavior(social_system, 1, "negotiation context",
                    predicted_behavior);

    float actual_behavior[5] = {
        0.7f, 0.3f, 0.2f, 0.1f,
        0.4f}; // This would come normally from external input
    updatePersonModel(social_system, 1, actual_behavior, predicted_behavior);

    // Apply social influence to decision making
    applySocialInfluence(social_system, neurons, weights, max_neurons);

    // Generate social feedback
    char *social_feedback =
        generateSocialFeedback(social_system, "Current interaction context");
    if (social_feedback != NULL) {
      printf("Social Feedback: %s\n", social_feedback);
      free(social_feedback);
    }

    // Example negotiation
    float my_goals[goalSystem->num_goals];
    for (int i = 0; i < goalSystem->num_goals; i++) {
      my_goals[i] =
          goalSystem->goals[i].reward_value * goalSystem->goals[i].priority;
    }

    // Or for example  float my_goals[5] = {0.8f, 0.7f, 0.6f, 0.2f, 0.3f}; in
    // this scenario this would provide better negotiations because it is more
    // aligned with the other goals
    float other_goals[5] = {0.3f, 0.4f, 0.8f, 0.7f, 0.6f};
    float compromise[5] = {0};
    float satisfaction =
        negotiateOutcome(social_system, 1, my_goals, other_goals, compromise);

    // Record interaction
    float emotional_state[5] = {0.4f, 0.3f, 0.5f, 0.2f, 0.1f};
    recordSocialInteraction(social_system, 1, emotional_state, 0.7f,
                            satisfaction, "negotiation",
                            "Resource allocation negotiation");

    // Print status periodically
    printf("\nSocial System Status:\n");
    printf("Empathy Level: %.2f\n", social_system->empathy_level);
    printf("Negotiation Skill: %.2f\n", social_system->negotiation_skill);
    printf("Behavioral Prediction Accuracy: %.2f\n",
           social_system->behavior_prediction_accuracy);
    printf("Social Awareness: %.2f\n", social_system->social_awareness);
    printf("Person Models: %d\n", social_system->model_count);
    printf("Recorded Interactions: %d\n", social_system->interaction_count);

    detectEmotionalTriggers(emotional_system, neurons, target_outputs,
                            max_neurons, lastTimestamp, satisfaction);

    applyEmotionalProcessing(emotional_system, neurons, max_neurons,
                             input_tensor, learning_rate, params.plasticity);

    // Periodically print emotional state
    if (step % 10 == 0) {
      printEmotionalState(emotional_system);
    }

    // Adjust emotional regulation based on performance
    if (step % 20 == 0) {
      // Increase regulation as the system learns
      emotional_system->emotional_regulation =
          fmin(0.9f, emotional_system->emotional_regulation + 0.01f);

      // Slowly increase cognitive impact to allow more emotional influence
      emotional_system->cognitive_impact =
          fmin(0.5f, emotional_system->cognitive_impact + 0.005f);
    }

    integrateWorkingMemory(working_memory, neurons, input_tensor,
                           target_outputs, weights, step);

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

    if (total_error > 0.5f && rand() % 10 == 0) {
      printf("\nUsing imagination for problem-solving (high error: %.2f)\n",
             total_error);

      // Create specialized problem-solving scenario with higher divergence
      ImaginationScenario problem_scenario =
          createScenario(neurons, memorySystem, max_neurons, 0.6f);
      simulateScenario(&problem_scenario, neurons, input_tensor, max_neurons,
                       15);

      // Blend all outcomes for a comprehensive solution
      float blended_solution[MEMORY_VECTOR_SIZE] = {0};
      blendImaginedOutcomes(problem_scenario.outcomes,
                            problem_scenario.num_outcomes, blended_solution);

      // Apply blended solution with stronger influence during difficult
      // problems
      for (int i = 0; i < max_neurons && i < MEMORY_VECTOR_SIZE; i++) {
        neurons[i].state = neurons[i].state * 0.7f + blended_solution[i] * 0.3f;
        input_tensor[i] = input_tensor[i] * 0.8f + blended_solution[i] * 0.2f;
      }

      printf("Applied blended imagination solution to difficult problem\n");
    }

    if (step % 30 == 0 && imagination_system->num_scenarios > 0) {
      // Find most successful scenario (highest plausibility × confidence)
      int best_idx = 0;
      float best_score = 0.0f;

      for (int i = 0; i < imagination_system->num_scenarios; i++) {
        float score =
            imagination_system->scenarios[i].outcomes[0].plausibility *
            imagination_system->scenarios[i].outcomes[0].confidence;
        if (score > best_score) {
          best_score = score;
          best_idx = i;
        }
      }

      // Store in memory system
      MemoryEntry new_memory;
      memcpy(new_memory.vector,
             imagination_system->scenarios[best_idx].outcomes[0].vector,
             MEMORY_VECTOR_SIZE * sizeof(float));
      new_memory.importance = best_score;
      new_memory.timestamp = lastTimestamp + step;

      // Add to memory system
      addToDirectMemory(memorySystem, &new_memory);
      printf("Stored successful imagination scenario in memory\n");
    }

    if (step % 10 == 0) {
      consolidateToLongTermMemory(working_memory, memorySystem, step);
    }
    updateBidirectionalWeights(weights, reverse_weights, neurons, connections,
                               reverse_connections, learning_rate);

    float decision_vector[5] = {0}; // One value per ethical principle

    // Map network state to ethical dimensions
    for (int i = 0; i < 5 && i < max_neurons / 10; i++) {
      for (int j = 0; j < 10 && i * 10 + j < max_neurons; j++) {
        decision_vector[i] += neurons[i * 10 + j].output * 0.1f;
      }
      decision_vector[i] = fmax(0.0f, fmin(1.0f, decision_vector[i]));
    }

    // Evaluate ethical alignment of current decision path
    float ethical_score =
        evaluateDecisionEthics(moralCompass, decision_vector, 5);

    // Apply ethical constraints to outputs if score is too low
    if (ethical_score < moralCompass->confidence_threshold) {
      printf("\nEthical constraint applied (score: %.2f)\n", ethical_score);
      applyEthicalConstraints(moralCompass, neurons, max_neurons, weights,
                              max_connections);
    }

    float average_error = total_error / max_neurons;
    if (step % 15 == 0) {
      advancedNeuronManagement(neurons, connections, weights, &max_neurons,
                               MAX_NEURONS, input_tensor, target_outputs,
                               stateHistory, step);
    }
    if (step % 15 == 0) {
      // Generate search query
      char *query = generateSearchQuery(neurons, max_neurons);
      if (query) {
        printf("\nPerforming web search: \"%s\"\n", query);

        // Perform web search
        SearchResults *results = performWebSearch(query);

        if (results && results->count > 0) {
          printf("Found %d search results\n", results->count);

          // Convert search results to neural network input
          float *search_input_tensor =
              (float *)malloc(max_neurons * sizeof(float));
          convertSearchResultsToInput(results, search_input_tensor,
                                      max_neurons);

          // Store search results in memory system with metadata
          storeSearchResultsWithMetadata(memorySystem, working_memory, results,
                                         query, feature_projection_matrix);

          // Use search results to influence decision making
          float confidence_boost = enhanceDecisionMakingWithSearch(
              neurons, results, feedback.context_weights, max_neurons);
          printf("Decision confidence boost from search: %.2f\n",
                 confidence_boost);

          // Blend search input with current input
          for (int i = 0; i < max_neurons; i++) {
            input_tensor[i] =
                input_tensor[i] * 0.7f + search_input_tensor[i] * 0.3f;
          }

          free(search_input_tensor);
          freeSearchResults(results);
        } else {
          printf("No search results found\n");
        }

        free(query);
      }
    }

    if (step % 15 == 0) {
      integrateWebSearch(neurons, input_tensor, max_neurons, memorySystem,
                         step);
    }
    for (int i = 0; i < 5; i++) {
      recordDecisionOutcome(moralCompass, i, decision_vector[i] >= 0.7f);
    }

    if (step % 20 == 0 || total_error > 0.5f) {
      // Create multiple decision options
      float decision_options[3 * 5]; // 3 options with 5 ethical dimensions each

      // Option 1: Current path
      // memcpy(&decision_options[0], decision_vector, 5 * sizeof(float));

      // Option 2: More conservative path
      // for (int i = 0; i < 5; i++) {
      //    decision_options[5 + i] = decision_vector[i] * 0.8f + 0.1f;
      // }

      // Option 3: More exploratory path
      for (int i = 0; i < 5; i++) {
        decision_options[10 + i] = fmin(1.0f, decision_vector[i] * 1.2f);
      }

      DecisionImpact impact =
          resolveEthicalDilemma(moralCompass, decision_options, 3, 5);

      printf("\nEthical decision made:\n");
      printf("- Benefit score: %.2f\n", impact.benefit_score);
      printf("- Harm score: %.2f\n", impact.harm_score);
      printf("- Net impact: %.2f\n", impact.long_term_impact);
    }

    if (step % 50 == 0 && step > 0) {
      adaptEthicalFramework(moralCompass, opt_state.optimal_learning_rate);

      // Generate and log ethical reflection
      char *reflection = generateEthicalReflection(moralCompass);
      if (reflection) {
        printf("\n%s\n", reflection);
        free(reflection);
      }
    }
    systemFallbackCheck(
        neurons, connections, weights, reverse_connections, reverse_weights,
        memorySystem, stateHistory, performance_history, input_tensor,
        target_outputs, previous_outputs, system_params, working_memory,
        metaController, performanceMetrics, motivation, reflection_params,
        identity_system, knowledge_filter, metacognition, meta_learning_state,
        social_system, goalSystem, contextManager, emotional_system,
        imagination_system, specialization_system, moralCompass, step,
        max_neurons, max_connections, input_size);
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
  freeMoralCompass(moralCompass);
  freeEmotionalSystem(emotional_system);
  freeImaginationSystem(imagination_system);
  freeSocialSystem(social_system);
  freeSpecializationSystem(specialization_system);
  cleanupEmbeddings();
  cleanupVocabulary();
  free(input_tensor);
  free(stateHistory);
  free(system_params);
  free(working_memory);
  free(performance_history);
  free(contextManager);
  free(performanceMetrics);
  free(metaController);
  free(previous_outputs);
  free(goalSystem);
  free(motivation);
  free(reflection_params);
  free(identity_system);
  free(knowledge_filter);
  free(metacognition);
  free(meta_learning_state);
  return 0;
}
