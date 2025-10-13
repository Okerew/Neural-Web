#ifndef ARM_SPECIFIC_H
#define ARM_SPECIFIC_H
typedef struct {
  char word[50];
  char category[50];
  char *connects_to;
  float semantic_weight;
  const char *description;
  float letter_weight;
} VocabularyEntry;
#endif // ARM_SPECIFIC_H
