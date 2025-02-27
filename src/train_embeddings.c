#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define EMBEDDING_SIZE 768
#define VOCAB_SIZE 100 

typedef struct {
    char word[50];
    char category[50];
    char *connects_to;
    float semantic_weight;
    const char *description; 
    float letter_weight;
} Vocabulary;

Vocabulary vocabulary[VOCAB_SIZE];

float embeddings[VOCAB_SIZE][EMBEDDING_SIZE];

void saveEmbeddingsToFile(const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Could not open file for writing: %s\n", filename);
        return;
    }

    fprintf(file, "%d %d\n", VOCAB_SIZE, EMBEDDING_SIZE);

    for (int i = 0; i < VOCAB_SIZE; i++) {
        fprintf(file, "%s", vocabulary[i].word);
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            fprintf(file, " %.6f", embeddings[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}


void trainCustomEmbeddings(int num_epochs, float learning_rate) {
    printf("Training custom word embeddings with %d epochs...\n", num_epochs);

    // Initialize embeddings with small random values
    float **word_embeddings = (float **)malloc(VOCAB_SIZE * sizeof(float *));
    float **context_embeddings = (float **)malloc(VOCAB_SIZE * sizeof(float *));
    if (word_embeddings == NULL || context_embeddings == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }

    for (int i = 0; i < VOCAB_SIZE; i++) {
        word_embeddings[i] = (float *)malloc(EMBEDDING_SIZE * sizeof(float));
        context_embeddings[i] = (float *)malloc(EMBEDDING_SIZE * sizeof(float));
        if (word_embeddings[i] == NULL || context_embeddings[i] == NULL) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            return;
        }

        // Initialize with Xavier/Glorot initialization
        float xavier_range = sqrt(6.0f / (2 * EMBEDDING_SIZE));
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            word_embeddings[i][j] = ((float)rand() / RAND_MAX * 2 - 1) * xavier_range;
            context_embeddings[i][j] = ((float)rand() / RAND_MAX * 2 - 1) * xavier_range;
        }
    }

    // Precompute word frequencies for negative sampling
    int *word_frequencies = (int *)malloc(VOCAB_SIZE * sizeof(int));
    if (word_frequencies == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }
    int total_frequency = 0;

    for (int i = 0; i < VOCAB_SIZE; i++) {
        int frequency = 1 + (int)(vocabulary[i].semantic_weight * 10);
        if (strcmp(vocabulary[i].category, "common") == 0) {
            frequency += 10;
        }
        word_frequencies[i] = frequency;
        total_frequency += frequency;
    }

    // Create sampling table for negative sampling
    int table_size = 1e8;
    int *negative_table = (int *)malloc(table_size * sizeof(int));
    if (negative_table == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }
    int table_pos = 0;
    float d = 0.75f;

    for (int i = 0; i < VOCAB_SIZE; i++) {
        float f = pow((float)word_frequencies[i], d);
        int slots = (int)((f / total_frequency) * table_size);
        for (int j = 0; j < slots && table_pos < table_size; j++) {
            negative_table[table_pos++] = i;
        }
    }

    while (table_pos < table_size) {
        negative_table[table_pos++] = rand() % VOCAB_SIZE;
    }

    // Build word context pairs for training
    typedef struct {
        int word_idx;
        int context_idx;
    } WordContextPair;

    int max_pairs = VOCAB_SIZE * (VOCAB_SIZE - 1);
    WordContextPair *pairs = (WordContextPair *)malloc(max_pairs * sizeof(WordContextPair));
    if (pairs == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }
    int pair_count = 0;

    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < VOCAB_SIZE; j++) {
            if (i != j && strcmp(vocabulary[i].category, vocabulary[j].category) == 0) {
                pairs[pair_count].word_idx = i;
                pairs[pair_count].context_idx = j;
                pair_count++;
            }
        }
    }

    printf("Generated %d word-context pairs for training\n", pair_count);

    // Training parameters
    int negative_samples = 5;
    float current_lr = learning_rate;

    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (int i = pair_count - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            WordContextPair temp = pairs[i];
            pairs[i] = pairs[j];
            pairs[j] = temp;
        }

        float total_loss = 0.0f;

        for (int i = 0; i < pair_count; i++) {
            int word_idx = pairs[i].word_idx;
            int context_idx = pairs[i].context_idx;

            // Positive sample
            float dot_product = 0.0f;
            for (int j = 0; j < EMBEDDING_SIZE; j++) {
                dot_product += word_embeddings[word_idx][j] * context_embeddings[context_idx][j];
            }

            float sigmoid = 1.0f / (1.0f + expf(-dot_product));
            float target = 1.0f;
            float error = target - sigmoid;
            total_loss += -log(sigmoid);

            float *word_grad = (float *)malloc(EMBEDDING_SIZE * sizeof(float));
            float *context_grad = (float *)malloc(EMBEDDING_SIZE * sizeof(float));
            if (word_grad == NULL || context_grad == NULL) {
                fprintf(stderr, "Error: Memory allocation failed\n");
                return;
            }

            for (int j = 0; j < EMBEDDING_SIZE; j++) {
                word_grad[j] = error * context_embeddings[context_idx][j];
                context_grad[j] = error * word_embeddings[word_idx][j];
            }

            // Negative samples
            for (int k = 0; k < negative_samples; k++) {
                int neg_idx = negative_table[rand() % table_size];
                while (neg_idx == context_idx) {
                    neg_idx = negative_table[rand() % table_size];
                }

                float neg_dot = 0.0f;
                for (int j = 0; j < EMBEDDING_SIZE; j++) {
                    neg_dot += word_embeddings[word_idx][j] * context_embeddings[neg_idx][j];
                }

                float neg_sigmoid = 1.0f / (1.0f + expf(-neg_dot));
                float neg_target = 0.0f;
                float neg_error = neg_target - neg_sigmoid;
                total_loss += -log(1.0f - neg_sigmoid);

                for (int j = 0; j < EMBEDDING_SIZE; j++) {
                    word_grad[j] += neg_error * context_embeddings[neg_idx][j];
                    context_embeddings[neg_idx][j] += current_lr * neg_error * word_embeddings[word_idx][j];
                }
            }

            for (int j = 0; j < EMBEDDING_SIZE; j++) {
                word_embeddings[word_idx][j] += current_lr * word_grad[j];
                context_embeddings[context_idx][j] += current_lr * context_grad[j];
            }

            free(word_grad);
            free(context_grad);
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == num_epochs - 1) {
            printf("Epoch %d/%d: Avg Loss = %.6f\n", epoch + 1, num_epochs, total_loss / pair_count);
        }

        current_lr = learning_rate * (1.0f - (float)epoch / num_epochs);
    }

    printf("Training completed!\n");

    // Final embeddings
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            embeddings[i][j] = word_embeddings[i][j] + context_embeddings[i][j];
        }

        // Apply custom modifiers
        if (strcmp(vocabulary[i].category, "fruit") == 0) {
            for (int j = 0; j < 10; j++) {
                embeddings[i][j] += 0.2f;
            }
        } else if (strcmp(vocabulary[i].category, "action") == 0) {
            for (int j = 10; j < 20; j++) {
                embeddings[i][j] += 0.2f;
            }
        } else if (strcmp(vocabulary[i].category, "emotion") == 0) {
            for (int j = 20; j < 30; j++) {
                embeddings[i][j] += 0.2f;
            }
        }

        float letter_weight = vocabulary[i].letter_weight;
        for (int j = 30; j < 40; j++) {
            embeddings[i][j] += letter_weight * 0.1f;
        }

        float semantic_weight = vocabulary[i].semantic_weight;
        for (int j = 40; j < 50; j++) {
            embeddings[i][j] += semantic_weight * 0.1f;
        }
    }

    // L2 normalization
    for (int i = 0; i < VOCAB_SIZE; i++) {
        float norm = 0.0f;
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            norm += embeddings[i][j] * embeddings[i][j];
        }
        norm = sqrt(norm);

        if (norm > 1e-8) {
            for (int j = 0; j < EMBEDDING_SIZE; j++) {
                embeddings[i][j] /= norm;
            }
        }
    }

    // Free allocated memory
    for (int i = 0; i < VOCAB_SIZE; i++) {
        free(word_embeddings[i]);
        free(context_embeddings[i]);
    }
    free(word_embeddings);
    free(context_embeddings);
    free(word_frequencies);
    free(negative_table);
    free(pairs);

    // Save embeddings to file
    saveEmbeddingsToFile("custom_embeddings.txt");

    printf("Custom embeddings trained and saved to 'custom_embeddings.txt'\n");
}

void loadVocabularyFromFile(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file for reading: %s\n", filename);
        return;
    }

    char line[512];
    int i = 0;
    
    // Read each line and populate the vocabulary array
    while (fgets(line, sizeof(line), file)) {
        if (i >= VOCAB_SIZE) {
            fprintf(stderr, "Warning: Vocabulary size exceeds the limit\n");
            break;
        }

        // Use strtok to split the line by commas
        char *token = strtok(line, ",");
        if (token) strcpy(vocabulary[i].word, token);

        token = strtok(NULL, ",");
        if (token) strcpy(vocabulary[i].category, token);

        token = strtok(NULL, ",");
        if (token) {
            vocabulary[i].connects_to = strdup(token);  // Copy connects_to string dynamically
        }

        token = strtok(NULL, ",");
        if (token) vocabulary[i].semantic_weight = atof(token);

        token = strtok(NULL, ",");
        if (token) vocabulary[i].description = strdup(token);  // Copy description dynamically

        token = strtok(NULL, ",");
        if (token) vocabulary[i].letter_weight = atof(token);

        i++;
    }

    fclose(file);
    printf("Vocabulary loaded from file '%s'\n", filename);
}

int main() {
    loadVocabularyFromFile("vocabulary.txt");
    trainCustomEmbeddings(100, 0.05f);
    return 0;
}
