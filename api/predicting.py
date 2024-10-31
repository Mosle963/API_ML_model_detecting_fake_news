from sklearn.svm import SVC
from nltk.corpus import stopwords
import gensim
import pickle
import numpy as np
import os
import re
import string
import nltk

def clean(text):

    # Download the stopwords if not already downloaded
    stop_words = set(stopwords.words('english'))

    # Add custom stop words (e.g., weekdays)
    custom_stop_words = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
    stop_words.update(custom_stop_words)
    text = text.lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    text_tokens = text.split()
    cleaned_text = ' '.join([word for word in text_tokens if word not in stop_words and len(word) > 2])
    return cleaned_text

class MyWord2Vec:
    def __init__(self, model):
        self.word2vecmodel = model

    def make_corpus_iterable(self, text_data):
        corpus_iterable = [gensim.utils.simple_preprocess(text) for text in text_data]
        return corpus_iterable

    def transform(self, text_data):
        corpus_iterable = self.make_corpus_iterable(text_data)
        vectors = [self.word2vecmodel.wv.get_mean_vector(text) for text in corpus_iterable]
        vectors_2d = np.stack(vectors)
        return vectors_2d

def predict(text):

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    w2v_file = os.path.join(BASE_DIR, 'api', 'w2v.pkl')
    svc_file = os.path.join(BASE_DIR, 'api', 'svc.pkl')

    with open(w2v_file, 'rb') as f:
        word2vec_model = pickle.load(f)

    with open(svc_file, 'rb') as f:
        svc_model = pickle.load(f)
    try:
        text = clean(text)
        text = [text]
        w2v = MyWord2Vec(word2vec_model)
        w2v_embeddings = w2v.transform(text)
        props = svc_model.predict_proba(w2v_embeddings)
        prob_true = props[:, 1]
        return prob_true[0]

    except Exception as e:
        return 0.0
