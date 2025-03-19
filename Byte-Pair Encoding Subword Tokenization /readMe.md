Byte-Pair Encoding (BPE) Subword Tokenization

📌 Assignment: Programming Assignment 1 (AIN442 / BBM497)

This repository contains a Python implementation of a variation of the Byte-Pair Encoding (BPE) algorithm for subword tokenization. The implementation follows the specifications provided in Programming Assignment 1 for AIN442 Practicum in Natural Language Processing / BBM497 NLP Lab at Hacettepe University.

🚀 Features

✅ Token Learner

Learns subword tokens from a given corpus (string or file).

Implements maximum merge count to limit the number of merges.

Returns (Merges, Vocabulary, TokenizedCorpus).

Respects whitespace boundaries—tokens from different words are not merged.

✅ Token Segmenter

Uses the learned merge rules to tokenize new input text.

✅ File Output Support

Can write results to an output file instead of returning them as output.

⚙️ Usage

🔹 Training the Token Learner

📂 Train using a text file

(Merges, Vocabulary, TokenizedCorpus) = bpeFN("hw01_tiny.txt", 10)

💾 Save the output to a file

bpeFNToFile("hw01_tiny.txt", 1000, "output.txt")

🔹 Tokenizing Text

Once the BPE merges have been learned, you can tokenize new text using:

tokenizedStr = bpeTokenize("sos sus ses sel fes araba", Merges)

