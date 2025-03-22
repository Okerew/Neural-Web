 #ifndef INPUT_LOADER_H
 #define INPUT_LOADER_H
 
 #include <stdbool.h>
 #include <stddef.h>
 
 typedef struct {
     char source_type[32];         // "file", "string", "memory"
     char file_path[256];          // Path to input file if source_type is "file"
     const char* text_input;       // Direct text input if source_type is "string"
     void* memory_pointer;         // Pointer to memory if source_type is "memory"
     size_t memory_size;           // Size of memory block if source_type is "memory"
     int max_input_length;         // Maximum input length to process
     float normalization_factor;   // Factor to normalize input values (default: 1.0)
     bool pad_input;               // Whether to pad input to max_neurons
     char padding_value;           // Value to use for padding (default: 0)
 } InputDataConfig;

 InputDataConfig initDefaultInputConfig();
 char* loadTextFromFile(const char* file_path, int max_length);
 int textToInputTensor(const char* text_input, float* input_tensor, int max_neurons, float normalization_factor);
 bool saveInputConfig(const InputDataConfig* config, const char* file_path);
 InputDataConfig loadInputConfig(const char* file_path);
 int binaryToInputTensor(const void* data, size_t data_size, float* input_tensor, int max_neurons, float normalization_factor);
 int loadInputData(InputDataConfig* config, float* input_tensor, int max_neurons, char** text_input_out);
 int loadInputDataFromJSON(const char* json_path, float* input_tensor, int max_neurons, char** text_input_out);
 int loadInputBatch(const char** file_paths, int num_files, float** input_tensors, 
                   int max_neurons, int batch_size);
 int processTextBatch(const char** text_inputs, int num_inputs, float** input_tensors, 
                     int max_neurons, float normalization_factor);
 
 #endif /* INPUT_LOADER_H */
