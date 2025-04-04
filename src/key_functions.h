#ifndef KEY_FUNCTIONS_H
#define KEY_FUNCTIONS_H

#include <ctype.h>
#include <curl/curl.h>
#include <float.h>
#include <json-c/json.h>
#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

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
#define PATTERN_SIZE 3
#define EXPERIENCE_VECTOR_SIZE 256
#define MAX_USAGE_COUNT 1000 // Maximum usage count for normalization
#define HISTORY_LENGTH 10
#define NUM_PATHS 5
#define MAX_DECISION_STEPS 20
#define MAX_SYMBOLS 100
#define MAX_QUESTIONS 10
#define VOCAB_SIZE 100
#define ACTIVATION_TANH 0
#define ACTIVATION_RELU 1
#define ACTIVATION_SIGMOID 2
#define ACTIVATION_LEAKY_RELU 3
#define ACTIVATION_SWISH 4
#define MAX_EMOTION_TYPES 8
#define EMOTION_LOVE 0
#define EMOTION_HATE 1

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
  char word[50];
  char category[50];
  char *connects_to;
  float semantic_weight;
  const char *description;
  float letter_weight;
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


MemorySystem* createMemorySystem(int capacity);
void loadMemorySystem(const char* filename, MemorySystem* memorySystem);
void saveMemorySystem(MemorySystem memorySystem, const char* filename);
void freeMemorySystem(MemorySystem memorySystem);
void loadHierarchicalMemory(MemorySystem memorySystem, const char* filename);
void saveHierarchicalMemory(MemorySystem memorySystem, const char* filename);
void decayMemorySystem(MemorySystem memorySystem);
void mergeSimilarMemories(MemorySystem memorySystem);
void addMemory(MemorySystem memorySystem, WorkingMemorySystem working_memory, Neuron* neurons, float* input_tensor, int timestamp, float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE]);
void retrieveMemory(MemorySystem memorySystem);
void consolidateMemory(MemorySystem memorySystem);
void consolidateToLongTermMemory(WorkingMemorySystem working_memory, MemorySystem memorySystem, int step);

void initializeNeurons(Neuron* neurons, int connections, float* weights, float* input_tensor);
void initializeWeights(float* weights, int max_neurons, int max_connections, float* input_tensor);
void updateNeuronsWithPredictiveCoding(Neuron* neurons, float* input_tensor, int max_neurons, float learning_rate);
void updateWeights(float* weights, Neuron* neurons, int* connections, float learning_rate);
void updateBidirectionalWeights(float* weights, float* reverse_weights, Neuron* neurons, int* connections, int* reverse_connections, float learning_rate);
void computePredictionErrors(Neuron* neurons, float* input_tensor, int max_neurons);
void generatePredictiveInputs(float* predictive_inputs, NetworkStateSnapshot* previous_state, int max_neurons);
void selectOptimalDecisionPath(Neuron* neurons, float* weights, int* connections, float* input_tensor, int max_neurons, float* previous_outputs, NetworkStateSnapshot* stateHistory, int step, MemoryEntry* relevantMemory, DynamicParameters* params);
void computeRegionPerformanceMetrics(NetworkPerformanceMetrics* performanceMetrics, Neuron* neurons, float* target_outputs, int max_neurons);
void updateMetaControllerPriorities(MetaController* metaController, NetworkPerformanceMetrics* performanceMetrics, MetacognitionMetrics* metacognition);
void applyMetaControllerAdaptations(Neuron* neurons, float* weights, MetaController* metaController, int max_neurons);
void selectOptimalMetaDecisionPath(Neuron* neurons, float* weights, int* connections, float* input_tensor, int max_neurons, MetaLearningState* meta_learning_state, MetacognitionMetrics* metacognition);
void adaptNetworkDynamic(Neuron* neurons, float* weights, DynamicParameters* params, float performance_delta, float* input_tensor);

void initDynamicParameters(DynamicParameters* params);
void updateDynamicParameters(DynamicParameters* params, float performance_delta, float stability, float error_rate);
void optimizeParameters(OptimizationState* opt_state, PerformanceMetrics* performance_history, int step);
void analyzeNetworkPerformance(PerformanceMetrics* performance_history, int step);
void generatePerformanceGraph(PerformanceMetrics* performance_history, int step);

void updateGlobalContext(GlobalContextManager* contextManager, Neuron* neurons, int max_neurons, float* input_tensor);
void integrateGlobalContext(GlobalContextManager* contextManager, Neuron* neurons, int max_neurons, float* weights, int max_connections);
void integrateReflectionSystem(Neuron* neurons, MemorySystem* memorySystem, NetworkStateSnapshot* stateHistory, int step, float* weights, int* connections, ReflectionParameters* reflection_params);
void updateIdentity(SelfIdentitySystem* identity_system, Neuron* neurons, int max_neurons, MemorySystem* memorySystem, float* input_tensor);
void verifyIdentity(SelfIdentitySystem* identity_system);
void analyzeIdentitySystem(SelfIdentitySystem* identity_system);
SelfIdentityBackup* createIdentityBackup(SelfIdentitySystem* identity_system);
void restoreIdentityFromBackup(SelfIdentitySystem* identity_system, SelfIdentityBackup* backup);
void freeIdentityBackup(SelfIdentityBackup* backup);
void generateIdentityReflection(SelfIdentitySystem* identity_system);

void updateMotivationSystem(IntrinsicMotivation* motivation, float performance_delta, float novelty, float task_difficulty);
void addGoal(GoalSystem* goalSystem, const char* description, float priority);
void evaluateGoalProgress(Goal* goal, Neuron* neurons, float* target_outputs);

void validateCriticalSecurity(Neuron* neurons, float* weights, int* connections, int max_neurons, int max_connections, MemorySystem* memorySystem);
void criticalSecurityShutdown(Neuron* neurons, float* weights, int* connections, MemorySystem* memorySystem, SecurityValidationStatus* secStatus);

void integrateKnowledgeFilter(KnowledgeFilter* knowledge_filter, MemorySystem* memorySystem, Neuron* neurons, float* input_tensor);
void updateKnowledgeSystem(Neuron* neurons, float* input_tensor, MemorySystem* memory_system, KnowledgeFilter* filter, float current_performance);
void printCategoryInsights(KnowledgeFilter* knowledge_filter);

void addSymbol(int symbol_id, const char* description);
void addQuestion(int question_id, int symbol_ids[], int num_symbols);
void askQuestion(int question_id, Neuron* neurons, float* input_tensor, MemorySystem* memorySystem, float* learning_rate);
void expandMemoryCapacity(MemorySystem *memorySystem);
void adjustBehaviorBasedOnAnswers(Neuron* neurons, float* input_tensor, MemorySystem* memorySystem, float *learning_rate, float *input_noise_scale, float *weight_noise_scale);
void enhanceDecisionMakingWithSearch(const Neuron *neurons, const SearchResults *results, float *decision_weights, int max_neurons);
void storeSearchResultsWithMetadata(MemorySystem *memorySystem, WorkingMemorySystem *working_memory, const SearchResults *results, const char *original_query, float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE]);
void addToWorkingMemory(WorkingMemorySystem *working_memory, const MemoryEntry *entry, float feature_projection_matrix[FEATURE_VECTOR_SIZE][MEMORY_VECTOR_SIZE]);
void integrateWebSearch(Neuron *neurons, float *input_tensor, int max_neurons, MemorySystem *memorySystem, int step);
void generateSearchQuery(const Neuron *neurons, int max_neurons);
void storeSearchResultsInMemory(MemorySystem *memorySystem, const SearchResults *results);
void addToDirectMemory(MemorySystem *memorySystem, const MemoryEntry *entry);
void convertSearchResultsToInput(const SearchResults *results, float *input_tensor, int max_neurons);
void performWebSearch(const char *query);
void parseSearchResults(const char *json_data);
void recordDecisionOutcome(MoralCompass *compass, int principle_index, bool was_ethical);
void resolveEthicalDilemma(MoralCompass *compass, float *decision_options, int num_options, int vector_size);
void applyEthicalConstraints(MoralCompass *compass, Neuron *neurons, int max_neurons, float *weights, int max_connections);
void generateEthicalReflection(MoralCompass *compass);
void adaptEthicalFramework(MoralCompass *compass, float learning_rate);
void freeMoralCompass(MoralCompass *compass);
void freeEmotionalSystem(EmotionalSystem *system);
void printEmotionalState(EmotionalSystem *system);
void detectEmotionalTriggers(EmotionalSystem *system, Neuron *neurons, float *target_outputs, int num_neurons, unsigned int timestamp);
void applyEmotionalProcessing(EmotionalSystem *system, Neuron *neurons, int num_neurons, float *input_tensor, float learning_rate, float plasticity);
float calculateEmotionalBias(EmotionalSystem *system, float *input, int input_size);
void updateEmotionalMemory(EmotionalSystem *system);
void triggerEmotion(EmotionalSystem *system, int emotion_type, float trigger_strength, unsigned int timestamp);
time_t getCurrentTime();
float computeMSELoss(Neuron* neurons, float* target_outputs, int max_neurons);
void verifyNetworkState(Neuron* neurons, TaskPrompt* current_prompt);
void transformOutputsToText(float* previous_outputs, int max_neurons, char* outputText, size_t size);
void findSimilarMemoriesInCluster(MemorySystem* memorySystem, float* vector, float similarity_threshold, int* num_matches);
void captureNetworkState(Neuron* neurons, float* input_tensor, NetworkStateSnapshot* stateHistory, float* weights, int step);
void printNetworkStates(Neuron* neurons, float* input_tensor, int step);
void saveNetworkStates(NetworkStateSnapshot* stateHistory, int steps);
void printReplayStatistics(MemorySystem* memorySystem);
void addEmbedding(const char* text, float* embedding);
void initializeEmbeddings();
void updateEmbeddings(float* embeddings, float* input_tensor, int max_embeddings, int max_neurons);
bool isWordMeaningful(const char* word);
void importPretrainedEmbeddings(const char* embedding_file);

size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp);
void initializeMetaController(int network_regions);
void initializeMotivationSystem();
void initializeGoalSystem(int num_goals);
void initializeGlobalContextManager(int max_neurons);
void initializePerformanceMetrics(int network_regions);
void initializeReflectionParameters();
void initializeSelfIdentity(int num_values, int num_beliefs, int num_markers, int history_size, int pattern_size);
void initializeKnowledgeFilter(int size);
void initializeMetacognitionMetrics();
void initializeMetaLearningState(int size);
void createWorkingMemorySystem(int capacity);
void initializeMoralCompass(int num_principles);
EmotionalSystem* initializeEmotionalSystem();

#endif // KEY_FUNCTIONS_H
