#ifndef DEFAULT_SPECIFIC_H
#define DEFAULT_SPECIFIC_H
#include <algorithm>

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

#endif // DEFAULT_SPECIFIC_H
