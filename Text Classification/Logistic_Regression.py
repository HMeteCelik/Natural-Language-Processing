import pandas as pd
import re
import string
from nltk.corpus import stopwords
from datasets import load_dataset
from collections import Counter
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

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




def pos_operations(pos_df): 
    pos_vocab = []
    for ind in range(len(pos_df)):
        text = pos_df.loc[ind, "text"].split()
        pos_vocab.extend(text)
    return pos_vocab

def neg_operations(neg_df): 
    neg_vocab = []
    for ind in range(len(neg_df)):
        text = neg_df.loc[ind, "text"].split()
        neg_vocab.extend(text)
    return neg_vocab


def bias_scores(train_df):
    for ind in range(len(train_df)):
        train_df.loc[ind, "text"] = preprocess_text(train_df.loc[ind, "text"])
    
    pos_df = train_df[train_df["label"] == 1].reset_index(drop=True)
    neg_df = train_df[train_df["label"] == 0].reset_index(drop=True)

    pos_vocab = pos_operations(pos_df)
    pos_counter = Counter(pos_vocab)
        
    neg_vocab = neg_operations(neg_df)
    neg_counter = Counter(neg_vocab)
        
    vocab = set(list(pos_counter.keys()) + list(neg_counter.keys()))

    bias_scores = []
    for word in vocab:
        fp = pos_counter[word]
        fn = neg_counter[word]
        ft = fp + fn
        num = abs(fp - fn)
        score = (num / ft) * math.log(ft)
        bias_scores.append((word, fp, fn, ft, score))

    bias_scores.sort(key=lambda x: (-x[4], x[0]))
        
    return bias_scores[:10000]


    


df = load_dataset("imdb")
train_df = pd.DataFrame(df["train"])
test_df = pd.DataFrame(df["test"])

scores = bias_scores(train_df)

top_words = [item[0] for item in scores]

vectorizer = CountVectorizer(vocabulary=top_words)

for ind in range(len(train_df)):
    train_df.loc[ind, "text"] = preprocess_text(train_df.loc[ind, "text"])
for ind in range(len(test_df)):
    test_df.loc[ind, "text"] = preprocess_text(test_df.loc[ind, "text"])


X_train = vectorizer.transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["label"]
y_test = test_df["label"]

train_accuracies = []
test_accuracies = []

for i in range(1, 26):
    lr_model = LogisticRegression(max_iter=i)
    lr_model.fit(X_train, y_train)
    
    y_train_pred = lr_model.predict(X_train)
    y_test_pred = lr_model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

print(scores[:2])
print(scores[-2:])

plt.figure(figsize=(10,6))
plt.plot(range(1, 26), train_accuracies, label="Train Accuracy", marker='o')
plt.plot(range(1, 26), test_accuracies, label="Test Accuracy", marker='o')
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy vs Number of Iterations")
plt.legend()
plt.show()


'''
As we can see from the plot, when the number of iterations increases, the training accuracy keeps getting better. However, the test accuracy does not improve much after about the 10th iteration.

To avoid overfitting and to have a model that works well on new data, I would choose the model at the 10th iteration. At this point, the model has good training and test accuracy without becoming too complex.
'''