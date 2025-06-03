# IMDB Sentiment Classification

This project implements binary sentiment classification (positive/negative) on the IMDB Movie Review dataset using two models: **Naive Bayes** and **Logistic Regression**.

## 📁 Dataset

- **Source**: [IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- Loaded via `load_dataset("imdb")` from the `datasets` library.
- Split into `train` and `test`, each with 25,000 samples.

## 🧼 Preprocessing

The `preprocess_text(text)` function:
- Removes punctuation and numbers
- Converts text to lowercase
- Removes stopwords (`nltk`)
- Normalizes whitespace

## 📊 Models

### 1. Naive Bayes Classifier
- Custom class `NaiveBayesClassifier` with `fit()` and `predict()` methods
- Outputs: predicted class and log probabilities

### 2. Logistic Regression
- Uses top 10,000 words with highest bias scores as BoW features
- Trains 25 models with different `max_iter` values
- Plots accuracy for both train and test sets
