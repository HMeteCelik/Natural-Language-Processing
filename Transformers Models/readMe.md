# 📰 News Article Enrichment with BERT & Google Gemini

This project provides an end-to-end solution for enhancing news articles by automatically classifying them using a fine-tuned BERT model and generating refined headlines and polished content using Google Gemini's LLM API.

---

📌 Project Overview

The goal is to streamline news content processing for online publications. The workflow includes:

* **🧠 Category Classification** – Predicting the category of a news article using a BERT-based model.
* **✍️ Title & Article Generation** – Leveraging Google Gemini (or Gemma) to create a professional title and rewrite the article content.

Both tasks are seamlessly integrated into a single pipeline for automated processing.

---


🛠️ Tools & Requirements

* **Language**: Python
* **Environment**: Jupyter Notebook (💡 Google Colab with T4 GPU recommended)
* **Classifier**: BERT Base Uncased from HuggingFace's transformers
* **LLM Service**: Google Gemini (or Gemma API via Google AI Studio)

---

📁 Project Structure

The project consists of three main sections implemented in a single Jupyter Notebook:

### 1️⃣ Category Classification

* **🔍 EDA**: Visualizing class distributions and text lengths.
* **🧼 Preprocessing**: Cleaning data if needed.
* **🧪 Model Training**: Fine-tuning BERT with experimentation on hyperparameters (learning rate, epochs, etc.).
* **📊 Evaluation**: Calculating F1 score (expected > 0.90) using `test.csv`.
* **🔎 Prediction Examples**: Predicting and displaying results for the first 5 test samples.
* **💾 Model Saving**: Storing the trained model to skip retraining.

### 2️⃣ Title & Article Generation (LLM)

* **🔑 API Integration**: Authenticate using Google Gemini API key.
* **🧠 Prompt Engineering**: Structure prompts to yield clean and parsable output (avoid fluff like “Here’s your title:”).
* **📤 Output Extraction**: Programmatically extract the headline and rewritten article from the LLM response.

### 3️⃣ End-to-End Pipeline Integration

* **📈 Category Prediction**: Use the saved BERT model to predict and convert labels to category names.
* **✍️ LLM Enrichment**: Feed the article into the LLM to get the enhanced title and content.
* **🗂️ Output Formatting**: Return results in dictionary format:

    ```json
    {
      "category": "...",
      "title": "...",
      "article": "..."
    }
    ```

* **🧪 Pipeline Demo**: Process the last 5 samples from `train.csv` and print enriched results as a list of dictionaries.

---

🚀 Quick Start

1.  Clone the repo and open the notebook in Google Colab.
2.  Upload the AG News subset (`train.csv`, `test.csv`).
3.  Fine-tune BERT or load the saved model.
4.  Set up your Google Gemini API key.
5.  Run the end-to-end pipeline.

