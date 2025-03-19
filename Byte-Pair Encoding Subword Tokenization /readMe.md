# Byte-Pair Encoding (BPE) Subword Tokenization  

## 📌 Assignment: Programming Assignment 1 (AIN442 / BBM497)  

This repository contains a Python implementation of a variation of the **Byte-Pair Encoding (BPE)** algorithm for subword tokenization. The implementation follows the specifications provided in **Programming Assignment 1** for **AIN442 Practicum in Natural Language Processing / BBM497 NLP Lab** at Hacettepe University.  

## 🚀 Features  

### ✅ Token Learner  
- Learns a vocabulary and merge rules from a given training corpus  
- Supports both direct string input and file-based corpus input  
- Returns a tuple containing the learned merges, vocabulary, and tokenized corpus  
- Implements a maximum merge count parameter to limit the number of merge operations  

### ✅ Token Segmenter  
- Tokenizes input text using the learned merge rules  
- Ensures consistency with the trained vocabulary  

### ✅ File Output Support  
- Saves the learned merges, vocabulary, and tokenized corpus to an output file  
