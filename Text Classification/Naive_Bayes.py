import pandas as pd
import re
import string
from nltk.corpus import stopwords
from datasets import load_dataset
from collections import Counter
import math
from sklearn.metrics import accuracy_score


stop = stopwords.words("english")

def preprocess_text(text):
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop]
    cleaned_text = ' '.join(tokens)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

class NaiveBayesClassifier:
    def __init__(self):
        self.total_pos_words = 0
        self.total_neg_words = 0
        self.vocab_size = 0
        self.prior_pos = 0
        self.prior_neg = 0
        self.pos_counter = Counter()
        self.neg_counter = Counter()

    def fit(self, train_df):
        for ind in range(len(train_df)):
            train_df.loc[ind, "text"] = preprocess_text(train_df.loc[ind, "text"])
        
        pos_df = train_df[train_df["label"] == 1].reset_index(drop=True)
        neg_df = train_df[train_df["label"] == 0].reset_index(drop=True)
        
        pos_vocab = self.pos_operations(pos_df, train_df)
        self.pos_counter = Counter(pos_vocab)
        
        neg_vocab = self.neg_operations(neg_df, train_df)
        self.neg_counter = Counter(neg_vocab)
        self.vocab_size = len(set(list(self.pos_counter.keys()) + list(self.neg_counter.keys())))
    
    def pos_operations(self, pos_df, train_df): 
        self.prior_pos = len(pos_df) / len(train_df)
        pos_vocab = []
        for ind in range(len(pos_df)):
            text = pos_df.loc[ind, "text"].split()
            self.total_pos_words += len(text)
            pos_vocab.extend(text)
        return pos_vocab

    def neg_operations(self, neg_df, train_df): 
        self.prior_neg = len(neg_df) / len(train_df)
        neg_vocab = []
        for ind in range(len(neg_df)):
            text = neg_df.loc[ind, "text"].split()
            self.total_neg_words += len(text)
            neg_vocab.extend(text)
        return neg_vocab
    
    def predict(self,text):
        text = preprocess_text(text).split()
        pos =  math.log(self.prior_pos)
        neg = math.log(self.prior_neg)
        for word in text:
            pos_numerator = (self.pos_counter[word] + 1)
            pos += math.log(pos_numerator / (self.total_pos_words + self.vocab_size))
            neg_numerator = (self.neg_counter[word] + 1)
            neg += math.log(neg_numerator / (self.total_neg_words + self.vocab_size))
        if pos > neg:
            return(1,pos,neg)
        return(0,pos,neg)



df = load_dataset("imdb")

train_df = pd.DataFrame(df["train"])

test_df = pd.DataFrame(df["test"])

nb = NaiveBayesClassifier()
nb.fit(train_df)
print(nb.total_pos_words)
print(nb.total_neg_words)
print(nb.vocab_size)
print(nb.prior_pos)
print(nb.prior_neg)
print(nb.pos_counter["great"])
print(nb.neg_counter["great"])
prediction1 = nb.predict(test_df.iloc[0]["text"])
prediction2 = nb.predict("This movie will be place at 1st in my favourite, movies!")
prediction3 = nb.predict("I couldn't wait for the movie to end, so I, turned it off halfway through. :D It was a complete disappointment.")
print(f"{'Positive' if prediction1[0] == 1 else 'Negative'}")
print(prediction1)
print(f"{'Positive' if prediction2[0] == 1 else 'Negative'}")
print(prediction2)
print(f"{'Positive' if prediction3[0] == 1 else 'Negative'}")
print(prediction3)
print(preprocess_text("This movie will be place at 1st in my favourite, movies!"))
print(preprocess_text("I couldn't wait for the movie to end, so I turned, it off halfway through. :D It was a complete disappointment."))
y_true = test_df['label'].values
y_pred = [nb.predict(text)[0] for text in test_df['text']]
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")