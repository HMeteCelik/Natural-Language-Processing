import gensim.downloader 
import random
import numpy as np
from numpy.linalg import norm

model = gensim.downloader.load("word2vec-google-news-300")

def replace_with_similar(sentence, indices):
    tokenized_sentence = sentence.split()
    similar_words_dict = {}
    
    for i in indices:
        word = tokenized_sentence[i]
        similars =  model.most_similar(word, topn=5)
        similar_words_dict[word] = similars
        tokenized_sentence[i] = random.choice(similars)[0]

    new_sentence = ' '.join(tokenized_sentence)
    return new_sentence, similar_words_dict


def sentence_vector(sentence):
    tokenized_sentence = sentence.split()
    vector_dict = {}
    for word in tokenized_sentence:
        try:
            vector_dict[word] = model[word]
        except KeyError:
            vector_dict[word] = np.zeros(300)
    sentence_vec = sum(vector_dict.values()) / len(vector_dict.keys())
    
    return vector_dict, sentence_vec


def most_similar_sentences(file_path,query):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.read().split("\n")
    similarity = []
    _,query_vector = sentence_vector(query)
    for sentence in sentences:
        _,vec = sentence_vector(sentence)
        cosine = np.dot(query_vector,vec)/(norm(query_vector)*norm(vec))
        similarity.append((sentence,cosine))
    similarity = sorted(similarity,key=lambda x:(-x[1],x[0]))
    return similarity