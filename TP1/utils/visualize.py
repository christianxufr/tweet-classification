import matplotlib.pyplot as plt
import pickle
import os


class Visualize:

    def __init__(self, path: str):
        file = open(path, 'br')
        self.history = pickle.load(file)
        file.close()

    @staticmethod
    def plot_history(history, output: str = "."):
        keys = list(history.keys())
        val = [k for k in keys if 'val' in k]
        train = [k for k in keys if k not in val]
        val = sorted(val)
        train = sorted(train)
        for train, val in zip(train, val):
            title = str.title(' '.join(train.split('_')))
            plt.plot(history[train])
            plt.plot(history[val])
            plt.title('Model ' + title)
            plt.ylabel(title)
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'])
            if output is not None:
                plt.savefig(os.path.join(output, train + '.png'))
            plt.show()

    def plot(self, output: str = "."):
        Visualize.plot_history(self.history, output)
