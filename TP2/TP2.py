import pandas as pd
import numpy as np
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM


class ModelLSTM:

    def __init__(self):
        self.df = None
        self.num_words = None
        self.embedding_dim = None
        self.lstm_out = None
        self.batch_size = None
        self.model = None
        self.X = None

    def load_data(self, path_to_data='train.txt', num_words=2500):
        self.df = pd.read_csv(path_to_data, sep='\t').set_index('id')
        self.num_words = num_words

        def concatenate_turn(t1, t2, t3):
            res = [f'{tt1} {tt2} {tt3}' for tt1, tt2, tt3 in zip(list(t1), list(t2), list(t3))]
            return res

        self.df['concatenate'] = concatenate_turn(self.df.turn1, self.df.turn2, self.df.turn3)
        tokenizer = Tokenizer(num_words=num_words, lower=True, split=' ')
        tokenizer.fit_on_texts(self.df.concatenate.values)
        self.X = tokenizer.texts_to_sequences(self.df.concatenate.values)
        self.X = pad_sequences(self.X)
    
    def compile(self, embedding_dim=128, lstm_out=200, batch_size=32):
        self.embedding_dim = embedding_dim
        self.lstm_out = lstm_out
        self.batch_size = batch_size
        model = Sequential()
        model.add(Embedding(self.num_words, embedding_dim, input_length = self.X.shape[1], dropout = 0.2))
        model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
        model.add(Dense(4,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        print(model.summary())
        self.model = model
    
    def fit_evaluate(self):
        Y = pd.get_dummies(self.df['label']).values
        X_train, X_valid, Y_train, Y_valid = train_test_split(self.X,Y, test_size = 0.20, random_state = 36)
        self.model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch = 1,  verbose = 5)
        score, acc = self.model.evaluate(X_valid, Y_valid, verbose=2, batch_size=self.batch_size)
        print(f'Score: {score}')
        print(f'Val accuracy: {acc}')
    
    def predict(self, test_data_path):
        df = pd.read_csv(test_data_path, sep='\t').set_index('id')
        num_words = self.num_words

        def concatenate_turn(t1, t2, t3):
            res = [f'{tt1} {tt2} {tt3}' for tt1, tt2, tt3 in zip(list(t1), list(t2), list(t3))]
            return res

        df['concatenate'] = concatenate_turn(df.turn1, df.turn2, df.turn3)
        tokenizer = Tokenizer(num_words=num_words, lower=True, split=' ')
        tokenizer.fit_on_texts(df.concatenate.values)
        X = tokenizer.texts_to_sequences(df.concatenate.values)
        X = pad_sequences(X)
        Y_pred = self.model.predict(X, batch_size=self.batch_size)
        return Y_pred