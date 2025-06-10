<readme>
# News Article Enrichment with BERT and Google Gemini

This project provides an end-to-end solution for enriching news articles by classifying their categories using a fine-tuned BERT model and then generating polished headlines and articles using Google Gemini's API.

## Project Overview

The core idea is to automate the process of categorizing news articles and then refining their presentation for a news website. This involves two main tasks:

1.  **Category Classification**: Using a BERT-based model to predict the category of a given news article.
2.  **Title and Article Generation**: Leveraging a large language model (LLM) to extract a suitable headline and rewrite the article in a professional, publication-ready format.

Finally, these two tasks are combined into a single pipeline to process articles seamlessly.

## [cite_start]Dataset 

The project utilizes a subset of the "AG News" dataset. The original dataset contains 120,000 training examples and 7,600 test examples. For faster BERT training, the training set has been reduced to 30,000 examples. [cite_start]There are a total of 4 different categories in the dataset. 

## [cite_start]Tools and Requirements 

* [cite_start]**Programming Language**: Python 
* [cite_start]**Development Environment**: Jupyter Notebook [cite: 13] [cite_start](Google Colab with T4 GPU is recommended for hardware acceleration). 
* [cite_start]**Classifier Model**: BERT Base Uncased (from the `transformers` library, specifically `BertForSequenceClassification`). 
* **LLM Service**: Google Gemini API (or Gemma API). [cite_start]Google AI Studio can be used for API key configuration and prompt experimentation. 

## Project Structure

The project is structured into three main sections within a Jupyter Notebook:

### [cite_start]1. Category Classification 

[cite_start]This section focuses on training and evaluating a BERT model for classifying news articles into one of 4 categories. 

* [cite_start]**Exploratory Data Analysis (EDA)**: Initial analysis of the `train.csv` dataset, including class distributions and text length statistics. 
* [cite_start]**Preprocessing**: Application of necessary preprocessing techniques on the training data if deemed necessary. 
* [cite_start]**Model Training**: Fine-tuning the "BERT Base Uncased" model using `train.csv` (or a derived training/validation set).  [cite_start]Experimentation with hyperparameters (learning rate, epochs, etc.) is encouraged to achieve optimal performance.  [cite_start]Hardware acceleration (GPU via CUDA) is recommended. 
* [cite_start]**Model Evaluation**: Evaluating the trained model on `test.csv` by calculating the F1 score.  [cite_start]An F1 score greater than 0.90 is expected for this dataset. 
* [cite_start]**Prediction Examples**: Classifying the first 5 samples from `test.csv` and printing their predicted and true labels (categories). 
* [cite_start]**Model Saving**: Saving the trained BERT model to avoid retraining. 

### [cite_start]2. Title and Well-Written Article Generation 

[cite_start]This section focuses on using Google's LLM API services to generate a title and a polished version of a given news article. 

* [cite_start]**API Integration**: Obtaining an API key and implementing it within the script. 
* [cite_start]**Prompt Engineering**: Designing prompts to ensure the LLM returns the title and well-written article in a specific, easily extractable format (e.g., using one-shot or few-shot prompting). 
* [cite_start]**Output Extraction**: Demonstrating the extraction of the title and polished article from the LLM's response.  [cite_start]Avoid extra model-generated phrases like "Here is an example title below:". 

### [cite_start]3. Combining 2 Tasks 

[cite_start]This section integrates the trained BERT model with the LLM API to create a complete article enrichment pipeline. 

* **Category Prediction**: Using the pre-trained and saved BERT model to predict the category of an input article. [cite_start]The numerical label is converted to a text label. 
* [cite_start]**LLM Generation**: Employing the LLM API to generate a title and a well-written version of the article. 
* [cite_start]**Formatted Output**: Extracting the title and well-written article separately and printing them in a dictionary format with keys: "category", "title", and "article". 
* [cite_start]**Pipeline Demonstration**: Processing the last 5 samples from `train.csv` through all steps in the pipeline and printing the resulting output as a list of dictionaries. 
</readme>
