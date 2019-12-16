from keras.preprocessing.text import Tokenizer
import io
import numpy as np



class Embedding:
    def __init__(self, embeddings_file_path):
        self.embeddings_file_path = embeddings_file_path
        self.tokenizer = None

    def getEmbeddings(self):
        embeddingsIndex = {}
        dim = 0
        with io.open(self.embeddings_file_path, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                embeddingVector = np.asarray(values[1:], dtype='float32')
                embeddingsIndex[word] = embeddingVector 
                dim = len(embeddingVector)
        return embeddingsIndex, dim


    def getEmbeddingMatrix(self, wordIndex, embeddings, dim):
        embeddingMatrix = np.zeros((len(wordIndex) + 1, dim))
        for word, i in wordIndex.items():
            embeddingMatrix[i] = embeddings.get(word)
        return embeddingMatrix
    
    def getMatrix(self):
        embeddings, dim = self.getEmbeddings()
        self.tokenizer = Tokenizer(filters='')
        self.tokenizer.fit_on_texts([' '.join(list(embeddings.keys()))])
        wordIndex = self.tokenizer.word_index
        embeddings_matrix = self.getEmbeddingMatrix(wordIndex, embeddings, dim)
        return embeddings_matrix
    
    def getTokenizer(self):
        return self.tokenizer