import os
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self):
        self.root = '../projet2'
        self.train_path = os.path.join(self.root, "train.txt")
        self.eval_path = os.path.join(self.root, "dev.txt")
        self.datastories_path = os.path.join(self.root, "datastories.twitter.100d.txt")
        self.word2vec_path = os.path.join(self.root, "GoogleNews-vectors-negative300.bin")
        self.emoji_path = os.path.join(self.root, "emoji.py")

    @staticmethod
    def load_tweet(path: str):
        data = pd.read_csv(path,
                           encoding="utf-8",
                           sep="\t",
                           header=None,
                           skiprows=[0],
                           names=["id", "turn1", "turn2", "turn3", "label"])
        tweet_class = {
            "happy": 0,
            "sad": 1,
            "angry": 2,
            "others": 3,
        }
        data['class'] = data['label'].apply(lambda x: tweet_class.get(x, x))
        data = data[["id", "turn1", "turn2", "turn3", "class"]]
        return data