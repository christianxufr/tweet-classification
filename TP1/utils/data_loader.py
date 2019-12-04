import os
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, data_root: str):
        self.root = data_root
        self.train_3_path = os.path.join(self.root, "data_train_3.csv")
        self.train_7_path = os.path.join(self.root, "data_train_7.csv")
        self.test_3_path = os.path.join(self.root, "test.txt")
        self.emoji_path = os.path.join(self.root, "emoji.py")
        self.word2vec_path = os.path.join(self.root, "GoogleNews-vectors-negative300.bin")
        self.val_7_v2_path = os.path.join(self.root, "2018-Valence-oc-En-dev.txt")
        self.train_7_v2_path = os.path.join(self.root, "2018-Valence-oc-En-train.txt")
        self.test_7_v2_path = os.path.join(self.root, "2018-Valence-oc-En-test.txt")

    @staticmethod
    def load_tweet(path: str):
        data = pd.read_csv(path,
                           encoding="utf-8",
                           sep="\t",
                           names=["id", "class", "text"])
        tweet_class = {
            "positive": 1,
            "neutral": 0,
            "negative": -1
        }
        data['class'] = data['class'].apply(lambda x: tweet_class.get(x, x))
        data = data[["id", "text", "class"]]
        return data

    @staticmethod
    def load_tweet_v2(path: str):
        data = pd.read_csv(path,
                           encoding="utf-8",
                           sep="\t",
                           names=["id", "text", "dim", "class"], header=0)
        if data['class'][0] != "NONE":
            data['class'] = data['class'].apply(lambda x: int(x.split(':')[0]))
        data = data[["id", "text", "class"]]
        return data

    @staticmethod
    def load_tweet_v2_no_label(path: str):
        data = pd.read_csv(path,
                           encoding="utf-8",
                           sep="\t",
                           names=["id", "text", "dim"], header=0)
        data = data[["id", "text"]]
        return data

    @staticmethod
    def write_tweet_v2(tweet: pd.DataFrame, path: str):
        classes = np.array(['0: neutral or mixed emotional state can be inferred',
                            '1: slightly positive emotional state can be inferred',
                            '2: moderately positive emotional state can be inferred',
                            '3: very positive emotional state can be inferred',
                            '-3: very negative emotional state can be inferred',
                            '-2: moderately negative emotional state can be inferred',
                            '-1: slightly negative emotional state can be inferred',
                            ])
        tweet = tweet.copy(deep=True)
        tweet['dim'] = ['valence'] * tweet.shape[0]
        tweet['class'] = classes[tweet['class']]
        tweet = tweet[['id', 'text', 'dim', 'class']]
        tweet.columns = ['ID', 'Tweet', 'Affect Dimension', 'Intensify Class']
        tweet.to_csv(path, sep='\t', index=False)
