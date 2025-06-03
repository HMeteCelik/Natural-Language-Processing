#####

# Word2Vec Applications

This project explores word embedding applications using **Word2Vec** vectors through the `Gensim` library. It implements two main functionalities: word replacement with similar alternatives and sentence similarity comparison.

## :package: Requirements

```python
import gensim.downloader
import numpy as np
```

## :rocket: Setup

Load the pre-trained Word2Vec model:

```python
model = gensim.downloader.load("word2vec-google-news-300")
```

## :wrench: Functions

### 1. Word Replacement with Similar Words

```python
replace_with_similar(sentence, indices)
```

- Replaces words at specified indices with randomly selected similar words  
- Returns: `(new_sentence, similar_words_dict)`  
- Uses top 5 most similar words from Word2Vec model

**Example:**

```python
sentence = "I love AIN442 and BBM497 courses"
indices = [1, 5]
new_sentence, most_similar_dict = replace_with_similar(sentence, indices)
# Output: "I adore AIN442 and BBM497 classes"
```

---

### 2. Sentence Similarity Analysis

#### `sentence_vector(sentence)`

- Converts a sentence to a 300-dimensional vector (mean of word vectors)  
- Returns: `(vector_dict, sentence_vec)`  
- Handles out-of-vocabulary words using zero vectors

#### `most_similar_sentences(file_path, query)`

- Finds most similar sentences using cosine similarity  
- Reads sentences from file and compares them with a query  
- Returns: sorted list of `(sentence, similarity_score)` tuples

**Example:**

```python
query = "Which courses have you taken at Hacettepe University ?"
results = most_similar_sentences("sentences.txt", query)
# Returns sentences ranked by similarity
```

---

## :bar_chart: Key Features

- **Word Similarity**: Leverages `Word2Vec`'s `most_similar()` method  
- **Vector Operations**: Uses `NumPy` for efficient computations  
- **Cosine Similarity**: Calculates similarity using `numpy.dot()` and `numpy.linalg.norm()`  
- **Random Selection**: Randomly chooses from top 5 similar words

---

## :file_folder: Input Files

- `sentences.txt`: Contains 20 sentences for similarity comparison  
- Each line represents a separate sentence (no preprocessing required)

#####
