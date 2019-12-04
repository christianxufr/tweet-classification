import nltk
import re
import importlib.util
import numpy as np
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from tokenizer.tokenizer import TweetTokenizer
from string import punctuation

from utils.data_loader import DataLoader

class Preprocess:

    def __init__(self, data_path: DataLoader):
        self.data = data_path
        self.spec = importlib.util.spec_from_file_location("emoji", self.data.emoji_path)
        self.emoji = importlib.util.module_from_spec(self.spec)
        self.spec.loader.exec_module(self.emoji)

        nltk.download('stopwords')
        nltk.download('wordnet')
        self.tknzr = TweetTokenizer(preserve_case=False,
                                    preserve_handles=False,
                                    preserve_hashes=False,
                                    preserve_url=False,
                                    preserve_len=False,
                                    regularize=True)
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words("english")) - {"against", "very", "all", "such", "not", "no"}

    def text_standardization(self, tweet):
        tweet = re.sub(r"\\u2019", "'", tweet)
        tweet = re.sub(r"\\u002c", ",", tweet)
        tweet = ' '.join(self.emoji.str2emoji(unidecode(tweet).lower().split()))
        tweet = re.sub(r"\'ve", " have", tweet)
        tweet = re.sub(r" can\'t", " cannot", tweet)
        tweet = re.sub(r"n\'t", " not", tweet)
        tweet = re.sub(r"\'re", " are", tweet)
        tweet = re.sub(r"\'d", " would", tweet)
        tweet = re.sub(r"\'ll", " will", tweet)
        tweet = re.sub(r"\'s", "", tweet)
        tweet = re.sub(r"\'n", "", tweet)
        tweet = re.sub(r"\'m", " am", tweet)
        tweet = self.tknzr.tokenize(tweet)
        tweet = [self.lemmatizer.lemmatize(i)
                 for i in tweet if (i not in self.stopwords) and (i not in punctuation)]
        tweet = ' '.join(tweet)
        return tweet
