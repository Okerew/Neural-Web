package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
)

type WordEntry struct {
	MEANINGS [][]interface{} `json:"MEANINGS"`
	ANTONYMS []string        `json:"ANTONYMS"`
	SYNONYMS []string        `json:"SYNONYMS"`
}

type VocabularyData map[string]WordEntry

// NOTE: These are the basic patterns for the example vocabulary given which can
// be read from the neural web if you want you can expand these patterns but this isn't necessary.
// That being said if you want to add more category maps you must also add them in the neural web.
var categoryPatterns = map[string]*regexp.Regexp{
	"fruit": regexp.MustCompile(`(?i)(apple|banana|cherry|date|elder|fig|
grape|honey|kiwi|lemon|mango|nect|orange|papaya|quince|rasp|straw|tang|
ugli|vanil|water|xigua|peach|lime|berry|melon)`),
	"vegetable": regexp.MustCompile(`(?i)(carrot|broccoli|potato|yam|
zucchini|bean|squash|lettuce|spinach|cabbage|onion|garlic|pepper|corn|
tomato|cucumber|peas|celery)`),
	"action": regexp.MustCompile(`(?i)(run|jump|sing|dance|write|move|go|
come|get|make|take|give|see|know|think|feel|work|play|live|try|ask|tell|
use|find|want|put|say|show|turn|start|bring|keep|hold|walk|talk|eat|
drink|sleep|wake|sit|stand|lie|fall|rise|open|close|push|pull|cut|hit)`),
	"emotion": regexp.MustCompile(`(?i)(love|hope|joy|fear|anger|sad|happy|
worry|excit|calm|stress|peace|proud|shame|guilt|anxious|nervous|
confident|surprised|confused|frustrated|jealous|grateful|lonely)`),
	"punctuation": regexp.MustCompile(`^[.,;:!?'"()\[\]{}\-]+$`),
	"adjective": regexp.MustCompile(`(?i)(fast|slow|big|small|high|low|
loud|quiet|hot|cold|new|old|good|bad|long|short|hard|soft|light|dark|
clean|dirty|easy|difficult|strong|weak|rich|poor|beautiful|ugly|
smart|stupid|kind|mean|funny|serious|important|dangerous)`),
	"preposition": regexp.MustCompile(`(?i)^(in|on|at|by|for|with|from|
to|of|about|under|over|through|between|among|during|before|after|
since|until|above|below|beside|behind|across|around|beyond|within|
without|against|toward|upon|beneath|throughout|underneath)$`),
	"pronoun": regexp.MustCompile(`(?i)^(i|you|he|she|it|we|they|me|him|
her|us|them|my|your|his|her|its|our|their|mine|yours|hers|ours|theirs|
myself|yourself|himself|herself|itself|ourselves|yourselves|themselves|
this|that|these|those|who|whom|whose|which|what)$`),
	"verb": regexp.MustCompile(`(?i)(be|is|am|are|was|were|been|being|
have|has|had|having|do|does|did|done|doing|will|would|shall|should|
may|might|can|could|must|ought)`),
}

func categorizeWord(word string) string {
	word = strings.ToLower(strings.TrimSpace(word))

	for category, pattern := range categoryPatterns {
		if pattern.MatchString(word) {
			return category
		}
	}

	if len(word) <= 3 && regexp.MustCompile(`^[a-z]+$`).MatchString(word) {
		return "common"
	}

	return "noun"
}

func calculateFrequency(word string, wordData WordEntry) string {
	score := 0.5

	wordLower := strings.ToLower(word)

	commonWords := map[string]float64{
		"the": 0.9, "be": 0.8, "to": 0.8, "of": 0.9, "and": 0.8, "a": 0.9,
		"in": 0.7, "that": 0.6, "have": 0.5, "it": 0.6, "for": 0.8,
		"not": 0.7, "on": 0.6, "with": 0.5, "he": 0.7, "as": 0.5,
		"you": 0.7, "do": 0.6, "at": 0.6, "are": 0.7, "is": 0.8,
		"was": 0.7, "but": 0.6, "his": 0.6, "from": 0.6, "they": 0.6,
		"we": 0.6, "say": 0.5, "her": 0.5, "she": 0.6, "or": 0.6,
		"an": 0.7, "will": 0.6, "my": 0.6, "one": 0.6, "all": 0.6,
		"would": 0.5, "there": 0.6, "their": 0.5,
	}

	if freq, exists := commonWords[wordLower]; exists {
		return fmt.Sprintf("%.1f", freq)
	}

	if len(wordData.MEANINGS) > 0 {
		score += 0.1
	}
	if len(wordData.MEANINGS) > 2 {
		score += 0.1
	}
	if len(wordData.SYNONYMS) > 0 {
		score += 0.05
	}
	if len(wordData.SYNONYMS) > 3 {
		score += 0.05
	}

	wordLen := len(word)
	switch {
	case wordLen <= 3:
		score += 0.2
	case wordLen <= 5:
		score += 0.1
	case wordLen > 10:
		score -= 0.1
	}

	if strings.Contains(strings.ToUpper(word), "ING") {
		score += 0.05
	}
	if strings.Contains(strings.ToUpper(word), "ED") {
		score += 0.05
	}
	if strings.Contains(strings.ToUpper(word), "S") &&
		len(word) > 1 && word[len(word)-1] == 'S' {
		score += 0.05
	}

	if score > 1.0 {
		score = 1.0
	}
	if score < 0.1 {
		score = 0.1
	}

	return fmt.Sprintf("%.1f", score)
}

func extractDefinition(meanings [][]interface{}) string {
	if len(meanings) == 0 {
		return "No definition available"
	}

	var definitions []string
	for i, meaning := range meanings {
		if i >= 2 {
			break
		}
		if len(meaning) >= 2 {
			if partOfSpeech, ok := meaning[0].(string); ok {
				if definition, ok := meaning[1].(string); ok {
					def := strings.TrimSpace(definition)
					if def != "" {
						if partOfSpeech != "" {
							definitions = append(definitions,
								fmt.Sprintf("%s: %s", partOfSpeech, def))
						} else {
							definitions = append(definitions, def)
						}
					}
				}
			}
		}
	}

	if len(definitions) == 0 {
		return "No definition available"
	}

	result := strings.Join(definitions, "; ")
	result = strings.ReplaceAll(result, ",", "")
	result = strings.ReplaceAll(result, "\n", " ")
	result = strings.ReplaceAll(result, "\r", "")
	result = strings.ReplaceAll(result, "  ", " ")

	if len(result) > 150 {
		words := strings.Fields(result)
		if len(words) > 20 {
			result = strings.Join(words[:20], " ") + "..."
		}
	}

	return result
}

func processVocabularyJSON(inputPath, outputPath string) error {
	inputFile, err := os.Open(inputPath)
	if err != nil {
		return fmt.Errorf("error opening input file: %v", err)
	}
	defer inputFile.Close()

	var vocabData VocabularyData
	decoder := json.NewDecoder(inputFile)
	if err := decoder.Decode(&vocabData); err != nil {
		return fmt.Errorf("error parsing JSON: %v", err)
	}

	outputFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("error creating output file: %v", err)
	}
	defer outputFile.Close()

	writer := bufio.NewWriter(outputFile)
	defer writer.Flush()

	processedCount := 0
	for word, wordData := range vocabData {
		word = strings.TrimSpace(word)
		if word == "" {
			continue
		}

		category := categorizeWord(word)
		frequency := calculateFrequency(word, wordData)
		definition := extractDefinition(wordData.MEANINGS)

		outputLine := fmt.Sprintf("%s,%s,%s,%s,0.0\n",
			word, category, frequency, definition)

		if _, err := writer.WriteString(outputLine); err != nil {
			return fmt.Errorf("error writing to output file: %v", err)
		}

		processedCount++
		if processedCount%5000 == 0 {
			fmt.Printf("Processed %d words...\n", processedCount)
		}
	}

	fmt.Printf("Successfully converted %d words\n", processedCount)
	return nil
}

func main() {
	if len(os.Args) != 3 {
		fmt.Printf("Usage: %s <input.json> <output.txt>\n", os.Args[0])
		fmt.Println("Converts JSON vocabulary files to TXT format")
		fmt.Println("Expected JSON format: {\"WORD\": {\"MEANINGS\": [[...]], \"ANTONYMS\": [...], \"SYNONYMS\": [...]}}")
		os.Exit(1)
	}

	inputFile := os.Args[1]
	outputFile := os.Args[2]

	if !strings.HasSuffix(strings.ToLower(inputFile), ".json") {
		log.Fatal("Input file must be a JSON file")
	}

	if !strings.HasSuffix(strings.ToLower(outputFile), ".txt") {
		log.Fatal("Output file must be a TXT file")
	}

	fmt.Printf("Converting vocabulary: %s -> %s\n", inputFile, outputFile)

	if err := processVocabularyJSON(inputFile, outputFile); err != nil {
		log.Fatalf("Conversion failed: %v", err)
	}

	fmt.Println("Conversion completed successfully!")
}
