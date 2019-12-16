from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import numpy as np

import re
import io


class Preprocess:

    def __init__(self):
        self.label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
        self.emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}

        self.emoticons_additional = {
            '(^・^)': '<happy>', ':‑c': '<sad>', '=‑d': '<happy>', ":'‑)": '<happy>', ':‑d': '<laugh>',
            ':‑(': '<sad>', ';‑)': '<happy>', ':‑)': '<happy>', ':\\/': '<sad>', 'd=<': '<annoyed>',
            ':‑/': '<annoyed>', ';‑]': '<happy>', '(^�^)': '<happy>', 'angru': 'angry', "d‑':":
                '<annoyed>', ":'‑(": '<sad>', ":‑[": '<annoyed>', '(�?�)': '<happy>', 'x‑d': '<laugh>',
        }

        self.text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                    'time', 'url', 'date', 'number'],
            # terms that will be annotated
            annotate={"hashtag", "allcaps", "elongated", "repeated",
                    'emphasis', 'censored'},
            fix_html=True,  # fix HTML tokens
            # corpus from which the word statistics are going to be used 
            # for word segmentation 
            segmenter="twitter",
            # corpus from which the word statistics are going to be used 
            # for spell correction
            corrector="twitter",
            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=True,  # spell correction for elongated words
            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons, self.emoticons_additional]
)
    def tokenize(self, text):
        text = " ".join(self.text_processor.pre_process_doc(text))
        return text


    def preprocessData(self, dataFilePath, mode):
        conversations = []
        labels = []
        with io.open(dataFilePath, encoding="utf8") as finput:
            finput.readline()
            for line in finput:
                line = line.strip().split('\t')
                for i in range(1, 4):
                    line[i] = self.tokenize(line[i])
                if mode == "train":
                    labels.append(self.emotion2label[line[4]])
                conv = line[1:4]
                conversations.append(conv)
        if mode == "train":
            return np.array(conversations), np.array(labels)
        else:
            return np.array(conversations)
