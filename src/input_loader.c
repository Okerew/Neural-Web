#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <json-c/json.h>

typedef struct {
  char source_type[32];       // "file", "string", "memory"
  char file_path[256];        // Path to input file if source_type is "file"
  const char *text_input;     // Direct text input if source_type is "string"
  void *memory_pointer;       // Pointer to memory if source_type is "memory"
  size_t memory_size;         // Size of memory block if source_type is "memory"
  int max_input_length;       // Maximum input length to process
  float normalization_factor; // Factor to normalize input values (default: 1.0)
  bool pad_input;             // Whether to pad input to max_neurons
  char padding_value;         // Value to use for padding (default: 0)
} InputDataConfig;

InputDataConfig initDefaultInputConfig() {
  InputDataConfig config;
  strcpy(config.source_type, "string");
  strcpy(config.file_path, "");
  config.text_input = NULL;
  config.memory_pointer = NULL;
  config.memory_size = 0;
  config.max_input_length = 4096;
  config.normalization_factor = 1.0f;
  config.pad_input = true;
  config.padding_value = 0;
  return config;
}

char *loadTextFromFile(const char *file_path, int max_length) {
  FILE *file = fopen(file_path, "r");
  if (file == NULL) {
    fprintf(stderr, "Error: Could not open file %s\n", file_path);
    return NULL;
  }

  // Determine file size
  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  fseek(file, 0, SEEK_SET);

  // Allocate buffer (limited by max_length if specified)
  int buffer_size = (max_length > 0 && file_size > max_length) ? max_length + 1
                                                               : file_size + 1;
  char *buffer = (char *)malloc(buffer_size);

  if (buffer == NULL) {
    fprintf(stderr, "Error: Memory allocation failed for file buffer\n");
    fclose(file);
    return NULL;
  }

  // Read file contents
  size_t bytes_read = fread(buffer, 1, buffer_size - 1, file);
  buffer[bytes_read] = '\0';

  fclose(file);
  return buffer;
}

int textToInputTensor(const char *text_input, float *input_tensor,
                      int max_neurons, float normalization_factor) {
  if (text_input == NULL || input_tensor == NULL || max_neurons <= 0) {
    return 0;
  }

  int input_length = strlen(text_input);
  int values_loaded = 0;

  // Clear input tensor
  memset(input_tensor, 0, max_neurons * sizeof(float));

  // Convert text to input tensor values
  for (int i = 0; i < input_length && i < max_neurons; i++) {
    // Convert character to normalized float value
    input_tensor[i] = ((float)text_input[i]) / normalization_factor;
    values_loaded++;
  }

  return values_loaded;
}

bool saveInputConfig(const InputDataConfig *config, const char *file_path) {
  FILE *file = fopen(file_path, "w");
  if (file == NULL) {
    fprintf(stderr, "Error: Could not open file %s for writing\n", file_path);
    return false;
  }

  fprintf(file, "source_type=%s\n", config->source_type);
  fprintf(file, "file_path=%s\n", config->file_path);
  fprintf(file, "max_input_length=%d\n", config->max_input_length);
  fprintf(file, "normalization_factor=%f\n", config->normalization_factor);
  fprintf(file, "pad_input=%d\n", config->pad_input);
  fprintf(file, "padding_value=%d\n", config->padding_value);

  fclose(file);
  return true;
}

InputDataConfig loadInputConfig(const char *file_path) {
  InputDataConfig config = initDefaultInputConfig();

  FILE *file = fopen(file_path, "r");
  if (file == NULL) {
    fprintf(stderr, "Warning: Could not open config file %s, using defaults\n",
            file_path);
    return config;
  }

  char line[512];
  while (fgets(line, sizeof(line), file)) {
    char key[256], value[256];
    if (sscanf(line, "%[^=]=%[^\n]", key, value) == 2) {
      if (strcmp(key, "source_type") == 0) {
        strcpy(config.source_type, value);
      } else if (strcmp(key, "file_path") == 0) {
        strcpy(config.file_path, value);
      } else if (strcmp(key, "max_input_length") == 0) {
        config.max_input_length = atoi(value);
      } else if (strcmp(key, "normalization_factor") == 0) {
        config.normalization_factor = atof(value);
      } else if (strcmp(key, "pad_input") == 0) {
        config.pad_input = atoi(value);
      } else if (strcmp(key, "padding_value") == 0) {
        config.padding_value = atoi(value);
      }
    }
  }

  fclose(file);
  return config;
}

int binaryToInputTensor(const void *data, size_t data_size, float *input_tensor,
                        int max_neurons, float normalization_factor) {
  if (data == NULL || input_tensor == NULL || max_neurons <= 0) {
    return 0;
  }

  // Clear input tensor
  memset(input_tensor, 0, max_neurons * sizeof(float));

  // Determine how many bytes to process
  size_t bytes_to_process = (data_size < max_neurons) ? data_size : max_neurons;
  const unsigned char *bytes = (const unsigned char *)data;

  // Convert bytes to input tensor values
  for (size_t i = 0; i < bytes_to_process; i++) {
    input_tensor[i] = ((float)bytes[i]) / normalization_factor;
  }

  return bytes_to_process;
}

int loadInputData(InputDataConfig *config, float *input_tensor, int max_neurons,
                  char **text_input_out) {
  if (input_tensor == NULL || max_neurons <= 0) {
    fprintf(stderr, "Error: Invalid input tensor or max_neurons\n");
    return -1;
  }

  char *text_input = NULL;
  int values_loaded = 0;

  // Source: File
  if (strcmp(config->source_type, "file") == 0) {
    text_input = loadTextFromFile(config->file_path, config->max_input_length);
    if (text_input == NULL) {
      fprintf(stderr, "Error: Failed to load text from file %s\n",
              config->file_path);
      return -1;
    }
    values_loaded = textToInputTensor(text_input, input_tensor, max_neurons,
                                      config->normalization_factor);
  }
  // Source: String
  else if (strcmp(config->source_type, "string") == 0) {
    if (config->text_input == NULL) {
      fprintf(stderr, "Error: No text input provided in config\n");
      return -1;
    }
    text_input = strdup(config->text_input);
    values_loaded = textToInputTensor(text_input, input_tensor, max_neurons,
                                      config->normalization_factor);
  }
  // Source: Memory
  else if (strcmp(config->source_type, "memory") == 0) {
    if (config->memory_pointer == NULL || config->memory_size == 0) {
      fprintf(stderr, "Error: Invalid memory pointer or size in config\n");
      return -1;
    }
    // Create text representation of binary data for text_input_out
    if (text_input_out != NULL) {
      // Allocate enough for hex representation
      text_input = (char *)malloc(config->memory_size * 3 + 1);
      if (text_input) {
        unsigned char *data = (unsigned char *)config->memory_pointer;
        int offset = 0;
        for (size_t i = 0; i < config->memory_size; i++) {
          offset += sprintf(text_input + offset, "%02X ", data[i]);
        }
        text_input[offset] = '\0';
      }
    }
    values_loaded = binaryToInputTensor(
        config->memory_pointer, config->memory_size, input_tensor, max_neurons,
        config->normalization_factor);
  } else {
    fprintf(stderr, "Error: Unknown source type '%s'\n", config->source_type);
    return -1;
  }

  // Handle padding if needed
  if (config->pad_input && values_loaded < max_neurons) {
    for (int i = values_loaded; i < max_neurons; i++) {
      input_tensor[i] = config->padding_value;
    }
  }

  // Return text input if requested
  if (text_input_out != NULL) {
    *text_input_out = text_input;
  } else if (text_input != NULL) {
    free(text_input);
  }

  return values_loaded;
}

int loadInputDataFromJSON(const char* json_path, float* input_tensor, int max_neurons, char** text_input_out) {
    if (json_path == NULL || input_tensor == NULL || max_neurons <= 0) {
        fprintf(stderr, "Error: Invalid parameters for JSON loading\n");
        return -1;
    }
    
    // Parse JSON file
    struct json_object *json_root;
    enum json_tokener_error jerr = json_tokener_success;
    
    json_root = json_object_from_file(json_path);
    if (json_root == NULL) {
        fprintf(stderr, "Error: Failed to parse JSON file %s\n", json_path);
        return -1;
    }

    InputDataConfig config = initDefaultInputConfig();
    
    // Extract input configuration from JSON
    struct json_object *source_type_obj, *file_path_obj, *text_input_obj, *max_length_obj;
    struct json_object *norm_factor_obj, *pad_input_obj, *padding_value_obj;
    
    // Get source_type
    if (json_object_object_get_ex(json_root, "source_type", &source_type_obj)) {
        const char* source_type = json_object_get_string(source_type_obj);
        strcpy(config.source_type, source_type);
    }
    
    // Get file_path if present
    if (json_object_object_get_ex(json_root, "file_path", &file_path_obj)) {
        const char* file_path = json_object_get_string(file_path_obj);
        strcpy(config.file_path, file_path);
    }
    
    // Get text_input if present
    if (json_object_object_get_ex(json_root, "text_input", &text_input_obj)) {
        config.text_input = json_object_get_string(text_input_obj);
    }
    
    // Get max_input_length
    if (json_object_object_get_ex(json_root, "max_input_length", &max_length_obj)) {
        config.max_input_length = json_object_get_int(max_length_obj);
    }
    
    // Get normalization_factor
    if (json_object_object_get_ex(json_root, "normalization_factor", &norm_factor_obj)) {
        config.normalization_factor = json_object_get_double(norm_factor_obj);
    }
    
    // Get pad_input
    if (json_object_object_get_ex(json_root, "pad_input", &pad_input_obj)) {
        config.pad_input = json_object_get_boolean(pad_input_obj);
    }
    
    // Get padding_value
    if (json_object_object_get_ex(json_root, "padding_value", &padding_value_obj)) {
        config.padding_value = json_object_get_int(padding_value_obj);
    }
    
    // Handle direct input data if present in JSON
    struct json_object *input_data_obj;
    if (json_object_object_get_ex(json_root, "input_data", &input_data_obj)) {
        // Check if input_data is a string
        if (json_object_is_type(input_data_obj, json_type_string)) {
            strcpy(config.source_type, "string");
            config.text_input = json_object_get_string(input_data_obj);
        }
        // Check if input_data is an array
        else if (json_object_is_type(input_data_obj, json_type_array)) {
            int array_len = json_object_array_length(input_data_obj);
            if (array_len > 0) {
                // Allocate memory for input tensor directly from array
                int values_to_load = (array_len < max_neurons) ? array_len : max_neurons;
                
                for (int i = 0; i < values_to_load; i++) {
                    struct json_object *value_obj = json_object_array_get_idx(input_data_obj, i);
                    if (json_object_is_type(value_obj, json_type_double)) {
                        input_tensor[i] = json_object_get_double(value_obj);
                    } else if (json_object_is_type(value_obj, json_type_int)) {
                        input_tensor[i] = (float)json_object_get_int(value_obj);
                    }
                }
                
                // Pad remaining values if needed
                if (config.pad_input && values_to_load < max_neurons) {
                    for (int i = values_to_load; i < max_neurons; i++) {
                        input_tensor[i] = config.padding_value;
                    }
                }
                
                // Create text representation for text_input_out if requested
                if (text_input_out != NULL) {
                    *text_input_out = malloc(array_len * 20 + 1); // Allocate enough for all values
                    if (*text_input_out != NULL) {
                        char *ptr = *text_input_out;
                        *ptr = '\0';
                        for (int i = 0; i < values_to_load; i++) {
                            struct json_object *value_obj = json_object_array_get_idx(input_data_obj, i);
                            ptr += sprintf(ptr, "%f ", json_object_get_double(value_obj));
                        }
                    }
                }
                
                json_object_put(json_root);
                return values_to_load;
            }
        }
    }
    
    // If we didn't directly load from input_data array, use the standard loadInputData
    int result = loadInputData(&config, input_tensor, max_neurons, text_input_out);
    
    // Free JSON object
    json_object_put(json_root);
    
    return result;
}


int loadInputBatch(const char **file_paths, int num_files,
                   float **input_tensors, int max_neurons, int batch_size) {
  if (file_paths == NULL || input_tensors == NULL || max_neurons <= 0 ||
      batch_size <= 0) {
    return 0;
  }

  int loaded_count = 0;
  int files_to_process = (num_files < batch_size) ? num_files : batch_size;

  for (int i = 0; i < files_to_process; i++) {
    InputDataConfig config = initDefaultInputConfig();
    strcpy(config.source_type, "file");
    strcpy(config.file_path, file_paths[i]);

    if (loadInputData(&config, input_tensors[i], max_neurons, NULL) > 0) {
      loaded_count++;
    }
  }

  return loaded_count;
}

int processTextBatch(const char **text_inputs, int num_inputs,
                     float **input_tensors, int max_neurons,
                     float normalization_factor) {
  if (text_inputs == NULL || input_tensors == NULL || max_neurons <= 0) {
    return 0;
  }

  int processed_count = 0;

  for (int i = 0; i < num_inputs; i++) {
    if (text_inputs[i] != NULL) {
      InputDataConfig config = initDefaultInputConfig();
      strcpy(config.source_type, "string");
      config.text_input = text_inputs[i];
      config.normalization_factor = normalization_factor;

      if (loadInputData(&config, input_tensors[i], max_neurons, NULL) > 0) {
        processed_count++;
      }
    }
  }

  return processed_count;
}
