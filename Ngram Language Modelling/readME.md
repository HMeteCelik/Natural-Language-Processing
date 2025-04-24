# Ngram Language Model (Unigram & Bigram)

## 📌 Assignment: Programming Assignment 2 (AIN442 / BBM497)

This repository contains a Python implementation of a variation of the **Ngram Language Model** (Unigram and Bigram) as described in **Programming Assignment 2** for **AIN442 Practicum in Natural Language Processing / BBM497 NLP Lab** at Hacettepe University.

## 🚀 Features

### ✅ Language Model Training
- Custom `ngramLM` class with support for unigram and bigram learning
- Learns from a file-based corpus encoded in UTF-8
- Turkish-aware lowercasing (e.g., I → ı, İ → i)
- Sentence tokenization with start (`<s>`) and end (`</s>`) tokens
- Tokenization via regular expressions

### ✅ Frequency and Probability Calculation
- Calculates unigram and bigram counts
- Computes unsmoothed and add-1 smoothed probabilities
- Handles unknown words for smoothing

### ✅ Sentence Probability Evaluation
- Computes probability of a sentence using smoothed bigram model
- Returns a probability score for a list of tokens

### ✅ Sentence Generation
- Generates sentences using probabilistic top-k sampling
- Supports constraints like max number of follow words and max sentence length
- Can generate the most probable sentence or random sentences

## 🧠 Example Usage
```python
lm = ngramLM()
lm.trainFromFile("hw02_tinyTestCorpus.txt")
print("Bigram Count:", lm.bigramCount(('a', 'b')))
print("Bigram Prob:", lm.bigramProb(('a', 'b')))
print("Generated Sentence:", lm.generateSentence())
```
