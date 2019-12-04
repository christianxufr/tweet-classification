import pickle


class ISave:

    @staticmethod
    def load(path: str):
        with open(path, "br") as file:
            return pickle.load(file)

    def save(self, path: str):
        with open(path, "bw") as file:
            pickle.dump(self, file)
