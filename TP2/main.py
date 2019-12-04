from utils.data_loader import DataLoader
from text_to_vec import TextToVec
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils.visualize import Visualize
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from utils.saving import Saving
from preprocess import Preprocess
import numpy as np
import pandas as pd
from models import *


def generate_text_preprocessing(data_loader, preprocess: Preprocess):
    data = data_loader.load_tweet_v2(data_loader.train_7_v2_path) \
        .append(data_loader.load_tweet_v2(data_loader.val_7_v2_path)) \
        .append(data_loader.load_tweet(data_loader.train_3_path)) \
        .append(data_loader.load_tweet(data_loader.train_7_path))

    vocab = data['text'].apply(preprocess.text_standardization)

    text_processing = TextToVec()
    text_processing.fit(data_loader, vocab, 300)
    text_processing.save("save/tp-all-300.save")


def train(model, model_name: str, X_train, X_val, y_train, y_val,
          saving: Saving):

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'],
                  )
    model.summary()
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=25,
                        batch_size=100,
                        callbacks=[*saving.get_callbacks(model_name,
                                                       tensorboard=False,
                                                       model_checkpoint=False),])
                                   #EarlyStopping(monitor='val_loss')])
    Visualize.plot_history(history.history, output="/tmp")
    return history


def preprocessing(data, preprocess, text2vec, output_dim = None):
    texts = data["text"].values
    texts = np.vectorize(preprocess.text_standardization)(texts)
    texts = text2vec.sequence_texts(texts)
    if output_dim is not None:
        label = to_categorical(data["class"], output_dim)
        return texts, label
    return texts, None


def main(data_loader: DataLoader, saving: Saving, preprocess: Preprocess, text2vec: TextToVec):
    train_7_v2 = data_loader.load_tweet_v2(data_loader.train_7_v2_path)
    val_7_v2 = data_loader.load_tweet_v2(data_loader.val_7_v2_path)
    train_3 = data_loader.load_tweet(data_loader.train_3_path)
    train_7 = data_loader.load_tweet(data_loader.train_7_path)

    texts, labels = preprocessing(train_7_v2, preprocess, text2vec, 7)
    val, vlabels = preprocessing(val_7_v2, preprocess, text2vec, 7)
    #texts, labels = shuffle(texts, labels)

    #X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2)


    embedding_layer = text2vec.get_embedding_layer(trainable=False)

    model3, model3_name = model3_gru(embedding_layer, saving)

    model3 = saving.load_model(model3_name)

    model, model7_name = model7(model3, saving, trainable=False)

    train(model, model7_name, texts, val, labels, vlabels, saving)
    saving.save_model(model, model7_name, override=True)

    #model, model_name = model3_gru(embedding_layer, saving)
    #train(model, model_name, train_3, saving, preprocess, text2vec)


def predict(data_loader: DataLoader, saving: Saving, preprocess: Preprocess, text2vec: TextToVec):
    model = saving.load_model("model7")
    test_data = data_loader.load_tweet_v2(data_loader.val_7_v2_path)
    texts, _ = preprocessing(test_data, preprocess, text2vec, 7)
    test_data["class"] = np.argmax(model.predict(texts), axis=1) - 3
    data_loader.write_tweet_v2(test_data, "predict.csv")


if __name__ == "__main__":
    data_loader = DataLoader("../data") #"/home/epita/cardamin/drive-reader/result")
    saving = Saving(directory="results")
    preprocess = Preprocess(data_loader)
    text2vec = TextToVec.load("save/tp-all-300.save")
    #main(data_loader, saving, preprocess, text2vec)
    predict(data_loader, saving, preprocess, text2vec)
    #generate_text_preprocessing(data_loader, preprocess)

