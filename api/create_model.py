import pandas as pd
import gensim
import numpy as np
import pickle
import os
from sklearn.svm import SVC


def save_embeddings(embeddings, filename):
    np.save(filename, embeddings)

def load_embeddings(filename):
    return np.load(filename)

class MyWord2Vec:
    def __init__(self, name, window_size=10, word_min_count=5, vector_size=300):
        self.window_size = window_size
        self.word_min_count = word_min_count
        self.vector_size = vector_size
        self.name = name
        self.word2vecmodel = gensim.models.Word2Vec(
            window=window_size,
            min_count=word_min_count,
            vector_size=vector_size
        )

    def make_corpus_iterable(self, text_data):
        corpus_iterable = [gensim.utils.simple_preprocess(text) for text in text_data]
        return corpus_iterable

    def fit_transform(self, text_data):
        corpus_iterable = self.make_corpus_iterable(text_data)
        self.word2vecmodel.build_vocab(corpus_iterable)
        self.word2vecmodel.train(corpus_iterable, total_examples=self.word2vecmodel.corpus_count, epochs=self.word2vecmodel.epochs)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        w2v_file = os.path.join(BASE_DIR, 'api', 'w2v.pkl')
        pickle.dump(self.word2vecmodel, open(w2v_file, 'wb'))
        return self.transform(text_data)

    def transform(self, text_data):
        corpus_iterable = self.make_corpus_iterable(text_data)
        vectors = [self.word2vecmodel.wv.get_mean_vector(text) for text in corpus_iterable]
        vectors_2d = np.stack(vectors)
        return vectors_2d


def createw2v():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_file = os.path.join(BASE_DIR, 'api', 'train_dataset.csv')
    train = pd.read_csv(dataset_file)
    X_train = train['clean']
    word2vec_model = MyWord2Vec(name='custom_word2vec')
    X_train_w2v_embeddings = word2vec_model.fit_transform(X_train)

    embedding_file = os.path.join(BASE_DIR, 'api', 'X_train_embeddings_word2vec.npy')
    save_embeddings(X_train_w2v_embeddings, embedding_file)

def createsvc():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_file = os.path.join(BASE_DIR, 'api', 'train_dataset.csv')
    train = pd.read_csv(dataset_file)
    y_train = train['label']
    embedding_file = os.path.join(BASE_DIR, 'api', 'X_train_embeddings_word2vec.npy')
    svc_file = os.path.join(BASE_DIR, 'api', 'svc.pkl')
    X_train_w2v_embeddings = load_embeddings(embedding_file)
    svc_pr_model = SVC(C=1.0, kernel='poly', probability=True)
    svc_pr_model.fit(X_train_w2v_embeddings, y_train)
    pickle.dump(svc_pr_model, open(svc_file, 'wb'))

