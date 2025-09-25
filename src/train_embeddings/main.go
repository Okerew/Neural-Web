package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

const (
	EmbeddingSize = 768
	VocabSize     = 100
)

type Vocabulary struct {
	Word           string
	Category       string
	ConnectsTo     *string
	SemanticWeight float64
	Description    *string
	LetterWeight   float64
}

type WordContextPair struct {
	WordIdx    int
	ContextIdx int
}

type EmbeddingTrainer struct {
	vocabulary []Vocabulary
	embeddings [][]float64
}

func NewEmbeddingTrainer() *EmbeddingTrainer {
	return &EmbeddingTrainer{
		vocabulary: make([]Vocabulary, 0, VocabSize),
		embeddings: make([][]float64, VocabSize),
	}
}

func (et *EmbeddingTrainer) initializeEmbeddings() {
	for i := range et.embeddings {
		et.embeddings[i] = make([]float64, EmbeddingSize)
	}
}

func (et *EmbeddingTrainer) saveEmbeddingsToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("could not create file %s: %v", filename, err)
	}
	defer file.Close()

	fmt.Fprintf(file, "%d %d\n", len(et.vocabulary), EmbeddingSize)

	for i, vocab := range et.vocabulary {
		fmt.Fprintf(file, "%s", vocab.Word)
		for j := 0; j < EmbeddingSize; j++ {
			fmt.Fprintf(file, " %.6f", et.embeddings[i][j])
		}
		fmt.Fprintln(file)
	}

	return nil
}

func (et *EmbeddingTrainer) trainCustomEmbeddings(numEpochs int,
	learningRate float64) {
	fmt.Printf("Training custom word embeddings with %d epochs...\n",
		numEpochs)

	vocabLen := len(et.vocabulary)
	wordEmbeddings := make([][]float64, vocabLen)
	contextEmbeddings := make([][]float64, vocabLen)

	xavierRange := math.Sqrt(6.0 / (2 * EmbeddingSize))
	for i := 0; i < vocabLen; i++ {
		wordEmbeddings[i] = make([]float64, EmbeddingSize)
		contextEmbeddings[i] = make([]float64, EmbeddingSize)

		for j := 0; j < EmbeddingSize; j++ {
			wordEmbeddings[i][j] = (rand.Float64()*2 - 1) * xavierRange
			contextEmbeddings[i][j] = (rand.Float64()*2 - 1) * xavierRange
		}
	}

	wordFrequencies := make([]int, vocabLen)
	totalFrequency := 0

	for i, vocab := range et.vocabulary {
		frequency := 1 + int(vocab.SemanticWeight*10)
		if vocab.Category == "common" {
			frequency += 10
		}
		wordFrequencies[i] = frequency
		totalFrequency += frequency
	}

	tableSize := int(1e8)
	negativeTable := make([]int, tableSize)
	tablePos := 0
	d := 0.75

	for i := 0; i < vocabLen; i++ {
		f := math.Pow(float64(wordFrequencies[i]), d)
		slots := int((f / float64(totalFrequency)) * float64(tableSize))
		for j := 0; j < slots && tablePos < tableSize; j++ {
			negativeTable[tablePos] = i
			tablePos++
		}
	}

	for tablePos < tableSize {
		negativeTable[tablePos] = rand.Intn(vocabLen)
		tablePos++
	}

	var pairs []WordContextPair
	for i := 0; i < vocabLen; i++ {
		for j := 0; j < vocabLen; j++ {
			if i != j && et.vocabulary[i].Category ==
				et.vocabulary[j].Category {
				pairs = append(pairs, WordContextPair{
					WordIdx:    i,
					ContextIdx: j,
				})
			}
		}
	}

	fmt.Printf("Generated %d word-context pairs for training\n", len(pairs))

	negativeSamples := 5
	currentLR := learningRate

	for epoch := 0; epoch < numEpochs; epoch++ {
		rand.Shuffle(len(pairs), func(i, j int) {
			pairs[i], pairs[j] = pairs[j], pairs[i]
		})

		totalLoss := 0.0

		for _, pair := range pairs {
			wordIdx := pair.WordIdx
			contextIdx := pair.ContextIdx

			dotProduct := 0.0
			for j := 0; j < EmbeddingSize; j++ {
				dotProduct += wordEmbeddings[wordIdx][j] *
					contextEmbeddings[contextIdx][j]
			}

			sigmoid := 1.0 / (1.0 + math.Exp(-dotProduct))
			target := 1.0
			error := target - sigmoid
			totalLoss += -math.Log(sigmoid)

			wordGrad := make([]float64, EmbeddingSize)
			contextGrad := make([]float64, EmbeddingSize)

			for j := 0; j < EmbeddingSize; j++ {
				wordGrad[j] = error * contextEmbeddings[contextIdx][j]
				contextGrad[j] = error * wordEmbeddings[wordIdx][j]
			}

			for k := 0; k < negativeSamples; k++ {
				negIdx := negativeTable[rand.Intn(tableSize)]
				for negIdx == contextIdx {
					negIdx = negativeTable[rand.Intn(tableSize)]
				}

				negDot := 0.0
				for j := 0; j < EmbeddingSize; j++ {
					negDot += wordEmbeddings[wordIdx][j] *
						contextEmbeddings[negIdx][j]
				}

				negSigmoid := 1.0 / (1.0 + math.Exp(-negDot))
				negTarget := 0.0
				negError := negTarget - negSigmoid
				totalLoss += -math.Log(1.0 - negSigmoid)

				for j := 0; j < EmbeddingSize; j++ {
					wordGrad[j] += negError * contextEmbeddings[negIdx][j]
					contextEmbeddings[negIdx][j] += currentLR * negError *
						wordEmbeddings[wordIdx][j]
				}
			}

			for j := 0; j < EmbeddingSize; j++ {
				wordEmbeddings[wordIdx][j] += currentLR * wordGrad[j]
				contextEmbeddings[contextIdx][j] += currentLR * contextGrad[j]
			}
		}

		if (epoch+1)%10 == 0 || epoch == 0 || epoch == numEpochs-1 {
			fmt.Printf("Epoch %d/%d: Avg Loss = %.6f\n", epoch+1,
				numEpochs, totalLoss/float64(len(pairs)))
		}

		currentLR = learningRate * (1.0 - float64(epoch)/float64(numEpochs))
	}

	fmt.Println("Training completed!")

	for i := 0; i < vocabLen; i++ {
		for j := 0; j < EmbeddingSize; j++ {
			et.embeddings[i][j] = wordEmbeddings[i][j] +
				contextEmbeddings[i][j]
		}

		vocab := et.vocabulary[i]
		switch vocab.Category {
		case "fruit":
			for j := 0; j < 10; j++ {
				et.embeddings[i][j] += 0.2
			}
		case "action":
			for j := 10; j < 20; j++ {
				et.embeddings[i][j] += 0.2
			}
		case "emotion":
			for j := 20; j < 30; j++ {
				et.embeddings[i][j] += 0.2
			}
		}

		letterWeight := vocab.LetterWeight
		for j := 30; j < 40; j++ {
			et.embeddings[i][j] += letterWeight * 0.1
		}

		semanticWeight := vocab.SemanticWeight
		for j := 40; j < 50; j++ {
			et.embeddings[i][j] += semanticWeight * 0.1
		}
	}

	for i := 0; i < vocabLen; i++ {
		norm := 0.0
		for j := 0; j < EmbeddingSize; j++ {
			norm += et.embeddings[i][j] * et.embeddings[i][j]
		}
		norm = math.Sqrt(norm)

		if norm > 1e-8 {
			for j := 0; j < EmbeddingSize; j++ {
				et.embeddings[i][j] /= norm
			}
		}
	}

	if err := et.saveEmbeddingsToFile("custom_embeddings.txt"); err != nil {
		log.Printf("Error saving embeddings: %v", err)
		return
	}

	fmt.Println("Custom embeddings trained and saved to " +
		"'custom_embeddings.txt'")
}

func (et *EmbeddingTrainer) loadVocabularyFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("error opening file %s: %v", filename, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Split(line, ",")
		if len(parts) < 3 {
			continue
		}

		vocab := Vocabulary{
			Word:         strings.TrimSpace(parts[0]),
			Category:     strings.TrimSpace(parts[1]),
			LetterWeight: 1.0,
		}

		if semanticWeight, err := strconv.ParseFloat(
			strings.TrimSpace(parts[2]), 64); err == nil {
			vocab.SemanticWeight = semanticWeight
		}

		if len(parts) > 3 && strings.TrimSpace(parts[3]) != "NULL" &&
			strings.TrimSpace(parts[3]) != "null" &&
			strings.TrimSpace(parts[3]) != "" {
			connectsTo := strings.TrimSpace(parts[3])
			vocab.ConnectsTo = &connectsTo
		}

		if len(parts) > 4 && strings.TrimSpace(parts[4]) != "" {
			description := strings.TrimSpace(parts[4])
			vocab.Description = &description
		}

		if len(parts) > 5 {
			if letterWeight, err := strconv.ParseFloat(
				strings.TrimSpace(parts[5]), 64); err == nil {
				vocab.LetterWeight = letterWeight
			}
		}

		et.vocabulary = append(et.vocabulary, vocab)

		if len(et.vocabulary) >= VocabSize {
			break
		}
	}

	return scanner.Err()
}

func main() {
	trainer := NewEmbeddingTrainer()
	trainer.initializeEmbeddings()

	if err := trainer.loadVocabularyFromFile("vocabulary.txt"); err != nil {
		log.Fatalf("Failed to load vocabulary: %v", err)
	}

	fmt.Printf("Loaded %d vocabulary entries\n", len(trainer.vocabulary))
	trainer.trainCustomEmbeddings(100, 0.05)
}
