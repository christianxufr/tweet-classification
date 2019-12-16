from keras.layers import Input, Dense, Embedding, Concatenate, Activation, \
    Dropout, LSTM, Bidirectional, GlobalMaxPooling1D, GaussianNoise, RepeatVector, Permute, Reshape, multiply, Flatten
from keras.models import Model
import numpy as np


class CustomModel:
    def __init__(self, model_num):
        self.model_num = model_num
        self.model = None
    
    def build(self, embeddings_matrix, sequence_length, lstm_dim, hidden_layer_dim, num_classes, 
               noise=0.1, dropout_lstm=0.2, dropout=0.2):

        if self.model_num == 1:
            self.model = self.buildModel_1(embeddings_matrix, sequence_length, lstm_dim, hidden_layer_dim, num_classes, 
               noise, dropout_lstm, dropout)
        
        elif self.model_num == 2:
            self.model = self.buildModel_2(embeddings_matrix, sequence_length, lstm_dim, hidden_layer_dim, num_classes, 
               noise, dropout_lstm, dropout)
        
        else:
            self.model = self.buildModel_3(embeddings_matrix, sequence_length, lstm_dim, hidden_layer_dim, num_classes, 
               noise, dropout_lstm, dropout)
    
    def buildModel_1(self, embeddings_matrix, sequence_length, lstm_dim, hidden_layer_dim, num_classes, 
               noise=0.1, dropout_lstm=0.2, dropout=0.2):

        turn1_input = Input(shape=(sequence_length,), dtype='int32')
        turn2_input = Input(shape=(sequence_length,), dtype='int32')
        turn3_input = Input(shape=(sequence_length,), dtype='int32')
        embedding_dim = embeddings_matrix.shape[1]
        embeddingLayer = Embedding(embeddings_matrix.shape[0],
                                    embedding_dim,
                                    weights=[embeddings_matrix],
                                    input_length=sequence_length,
                                    trainable=False)
        
        turn1_branch = embeddingLayer(turn1_input)
        turn2_branch = embeddingLayer(turn2_input) 
        turn3_branch = embeddingLayer(turn3_input) 
        
        turn1_branch = GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim))(turn1_branch)
        turn2_branch = GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim))(turn2_branch)
        turn3_branch = GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim))(turn3_branch)

        lstm1 = Bidirectional(LSTM(lstm_dim, dropout=dropout_lstm))
        lstm2 = Bidirectional(LSTM(lstm_dim, dropout=dropout_lstm))
        
        turn1_branch = lstm1(turn1_branch)
        turn2_branch = lstm2(turn2_branch)
        turn3_branch = lstm1(turn3_branch)
        
        x = Concatenate(axis=-1)([turn1_branch, turn2_branch, turn3_branch])
        
        x = Dropout(dropout)(x)
        
        x = Dense(hidden_layer_dim, activation='relu')(x)
        
        output = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=[turn1_input, turn2_input, turn3_input], outputs=output)
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        
        return model

    
    def buildModel_2(self, embeddings_matrix, sequence_length, lstm_dim, hidden_layer_dim, num_classes, 
               noise=0.1, dropout_lstm=0.2, dropout=0.2):

        turn1_input = Input(shape=(sequence_length,), dtype='int32')
        turn2_input = Input(shape=(sequence_length,), dtype='int32')
        turn3_input = Input(shape=(sequence_length,), dtype='int32')
        embedding_dim = embeddings_matrix.shape[1]
        embeddingLayer = Embedding(embeddings_matrix.shape[0],
                                    embedding_dim,
                                    weights=[embeddings_matrix],
                                    input_length=sequence_length,
                                    trainable=False)
        
        turn1_branch = embeddingLayer(turn1_input)
        turn2_branch = embeddingLayer(turn2_input) 
        turn3_branch = embeddingLayer(turn3_input) 
        
        turn1_branch = GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim))(turn1_branch)
        turn2_branch = GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim))(turn2_branch)
        turn3_branch = GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim))(turn3_branch)

        lstm1 = Bidirectional(LSTM(lstm_dim, dropout=dropout_lstm))
        lstm2 = Bidirectional(LSTM(lstm_dim, dropout=dropout_lstm))
        
        turn1_branch = lstm1(turn1_branch)
        turn2_branch = lstm2(turn2_branch)
        turn3_branch = lstm1(turn3_branch)
        
        turn1_branch = RepeatVector(1)(turn1_branch)
        turn2_branch = RepeatVector(1)(turn2_branch)
        turn3_branch = RepeatVector(1)(turn3_branch)
        
        x = Concatenate(axis=1)([turn1_branch, turn2_branch, turn3_branch])

        x = LSTM(lstm_dim, dropout=dropout_lstm)(x)

        x = Dropout(dropout)(x)
        
        x = Dense(hidden_layer_dim, activation='relu')(x)
        
        output = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=[turn1_input, turn2_input, turn3_input], outputs=output)
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        
        return model

    
    def buildModel_3(self, embeddings_matrix, sequence_length, lstm_dim, hidden_layer_dim, num_classes, 
               noise=0.1, dropout_lstm=0.2, dropout=0.2):
        
        def attention_3d_block(inputs):
            input_dim = int(inputs.shape[2])
            a = Permute((2, 1))(inputs)
            a = Reshape((input_dim, 3))(a)
            a = Dense(3, activation='softmax')(a)
            a_probs = Permute((2, 1), name='attention_vec')(a)
            output_attention_mul = multiply([inputs, a_probs])
            return output_attention_mul
        
        turn1_input = Input(shape=(sequence_length,), dtype='int32')
        turn2_input = Input(shape=(sequence_length,), dtype='int32')
        turn3_input = Input(shape=(sequence_length,), dtype='int32')
        embedding_dim = embeddings_matrix.shape[1]
        embeddingLayer = Embedding(embeddings_matrix.shape[0],
                                    embedding_dim,
                                    weights=[embeddings_matrix],
                                    input_length=sequence_length,
                                    trainable=False)
        
        turn1_branch = embeddingLayer(turn1_input)
        turn2_branch = embeddingLayer(turn2_input) 
        turn3_branch = embeddingLayer(turn3_input) 
        
        turn1_branch = GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim))(turn1_branch)
        turn2_branch = GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim))(turn2_branch)
        turn3_branch = GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim))(turn3_branch)

        lstm1 = Bidirectional(LSTM(lstm_dim, dropout=dropout_lstm))
        lstm2 = Bidirectional(LSTM(lstm_dim, dropout=dropout_lstm))
        
        turn1_branch = lstm1(turn1_branch)
        turn2_branch = lstm2(turn2_branch)
        turn3_branch = lstm1(turn3_branch)
        
        turn1_branch = RepeatVector(1)(turn1_branch)
        turn2_branch = RepeatVector(1)(turn2_branch)
        turn3_branch = RepeatVector(1)(turn3_branch)
        
        x = Concatenate(axis=1)([turn1_branch, turn2_branch, turn3_branch])

        x = LSTM(lstm_dim, dropout=dropout_lstm, return_sequences=True)(x)

        x = attention_3d_block(x)
        
        x = Flatten()(x)
        
        x = Dropout(dropout)(x)
        
        x = Dense(hidden_layer_dim, activation='relu')(x)
        
        output = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=[turn1_input, turn2_input, turn3_input], outputs=output)
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        
        return model

    def summary(self):
        print(self.model.summary())

    def train(self, message_first_message_train, message_second_message_train, message_third_message_train,
        labels_categorical_train, message_first_message_val, message_second_message_val, message_third_message_val,
        labels_categorical_val):

        epochs = 20 if self.model_num == 1 else 9

        return self.model.fit([message_first_message_train, message_second_message_train, message_third_message_train],
                    np.array(labels_categorical_train),
                    validation_data=(
                        [message_first_message_val, message_second_message_val, message_third_message_val],
                        np.array(labels_categorical_val)
                    ),
                    epochs=epochs,
                    batch_size=200)

    def predict(self, message_first_message_dev, message_second_message_dev, message_third_message_dev):
        return self.model.predict([message_first_message_dev, message_second_message_dev, message_third_message_dev])

    