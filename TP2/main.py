from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

from embedding import Embedding
from preprocess import Preprocess
from model import CustomModel
from Metrics import evaluation


def get_sequences(texts, sequence_length, tokenizer):
    message_first = pad_sequences(tokenizer.texts_to_sequences(texts[:, 0]), sequence_length)
    message_second = pad_sequences(tokenizer.texts_to_sequences(texts[:, 1]), sequence_length)
    message_third = pad_sequences(tokenizer.texts_to_sequences(texts[:, 2]), sequence_length)
    return message_first, message_second, message_third

def main(model_num=1):

    preprocess = Preprocess()

    texts_train, labels_train = preprocess.preprocessData('../projet2/train.txt', mode="train")
    texts_dev, labels_dev = preprocess.preprocessData('../projet2/dev.txt', mode="train")

    MAX_SEQUENCE_LENGTH = 24
    LSTM_DIM = 64
    HIDDEN_LAYER_DIM = 30
    NUM_CLASSES = 4
    GAUSSIAN_NOISE = 0.1
    DROPOUT = 0.2
    DROPOUT_LSTM = 0.2
    BATCH_SIZE = 200

    X_train, X_val, y_train, y_val = train_test_split(texts_train, labels_train, test_size=0.2, random_state=42)

    labels_categorical_train = to_categorical(np.asarray(y_train))
    labels_categorical_val = to_categorical(np.asarray(y_val))
    labels_categorical_dev = to_categorical(np.asarray(labels_dev))

    embedding = Embedding('../projet2/emosense.300d.txt')
    embeddings = embedding.getMatrix()
    tokenizer = embedding.getTokenizer()

    message_first_message_train, message_second_message_train, message_third_message_train = get_sequences(X_train, MAX_SEQUENCE_LENGTH, tokenizer)
    message_first_message_val, message_second_message_val, message_third_message_val = get_sequences(X_val, MAX_SEQUENCE_LENGTH, tokenizer)
    message_first_message_dev, message_second_message_dev, message_third_message_dev = get_sequences(texts_dev, MAX_SEQUENCE_LENGTH, tokenizer)
    

    model = CustomModel(model_num)
    model.build(embeddings, MAX_SEQUENCE_LENGTH, LSTM_DIM, HIDDEN_LAYER_DIM, NUM_CLASSES, 
               noise=GAUSSIAN_NOISE, dropout_lstm=DROPOUT_LSTM, dropout=DROPOUT)
    model.summary()
    history = model.train(message_first_message_train, message_second_message_train, message_third_message_train,
                    labels_categorical_train, message_first_message_val, message_second_message_val, message_third_message_val,
                    labels_categorical_val)
                    
    y_pred = model.predict([message_first_message_dev, message_second_message_dev, message_third_message_dev])
    #evaluation(y_pred, labels_categorical_dev)

if __name__ == "__main__":
    main()

