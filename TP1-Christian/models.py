from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional


def model3_blstm(embedding_layer, saving):
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model_name = "blstm_3"
    saving.save_model(model, model_name)
    return model, model_name


def model3_lstm(embedding_layer, saving):
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(3, activation='softmax'))
    model_name = "lstm_3_2"
    saving.save_model(model, model_name)
    return model, model_name


def model3_gru(embedding_layer, saving):
    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(32))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model_name = "model4"
    saving.save_model(model, model_name)
    return model, model_name


def model7(model3, saving, trainable: bool):
    model7 = model3
    model7.pop()
    for layer in model7.layers:
        layer.trainable = trainable
    model7.add(Dense(64, activation='relu', name='dense1'))
    model7.add(Dropout(0.2))
    model7.add(Dense(32, activation='relu', name='dense2'))
    model7.add(Dropout(0.2))
    model7.add(Dense(7, activation='softmax', name='dense3'))
    model_name = "model7"
    saving.save_model(model7, model_name)
    return model7, model_name
