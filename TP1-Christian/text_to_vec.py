import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.data_loader import DataLoader
from gensim.models import KeyedVectors

from tensorflow.keras.layers import Embedding
from utils.isave import ISave


class TextToVec(ISave):

    def __init__(self):
        self.tokenizer = None
        self.sequence_max_length = 0
        self.embedding_matrix = None
        self.embedding_dim = 0
        self.nb_words = 0

    def generate_embedding_matrix(self, word2vec_path):
        oov = list()
        oov.append((np.random.rand(self.embedding_dim) * 2.0) - 1.0)
        oov = oov / np.linalg.norm(oov)

        word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

        for word, i in self.tokenizer.word_index.items():
            if word in word2vec.vocab:
                self.embedding_matrix[i] = word2vec.word_vec(word)
            else:
                self.embedding_matrix[i] = oov

    def fit(self, data_path: DataLoader, vocabulary: pd.Series, embedding_dim: int):
        self.tokenizer = Tokenizer(filters=' ')
        self.tokenizer.fit_on_texts(vocabulary)
        self.nb_words = len(self.tokenizer.word_index) + 1  # index start at 1
        self.sequence_max_length = 0

        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.zeros((self.nb_words, embedding_dim))
        self.generate_embedding_matrix(data_path.word2vec_path)

        sequences = self.tokenizer.texts_to_sequences(vocabulary)
        self.sequence_max_length = max(len(s) for s in sequences)

    def sequence_texts(self, text):
        sequences = self.tokenizer.texts_to_sequences(text)
        sequences = pad_sequences(sequences, maxlen=self.sequence_max_length)
        return sequences

    def get_embedding_layer(self, trainable: bool = False):
        return Embedding(self.nb_words, self.embedding_dim,
                         weights=[self.embedding_matrix],
                         input_length=self.sequence_max_length,
                         trainable=trainable,
                         name='embedding_layer')
