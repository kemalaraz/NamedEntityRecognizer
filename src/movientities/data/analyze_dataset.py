import os
import seaborn as sns
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt

class Analyzer:

    @staticmethod
    def get_labels(path:str):
        pass

    @staticmethod
    def count_labels(path:str, without_o:bool=False):
        """
        Count labels.

        Args:
            path (str): Path of the file.
            without_o (bool, optional): If we dont want to count O or not. Defaults to False.

        Returns:
            [type]: [description]
        """
        assert os.path.exists(path), "Given path doesn't exists."
        labels = {}
        with open(path) as f:
            for line in f.readlines():
                if line != "\n":
                    label = line.split("\t")[0]
                    if label not in labels.keys():
                        labels[label] = 1
                    else:
                        labels[label] += 1
        if without_o:
            labels.pop("O", None)
        return labels

    @staticmethod
    def plot_data(plot_dict:Dict):
        labels = list(plot_dict.keys())
        counts = list(plot_dict.values())
        plt.barh(labels, counts, color=sns.color_palette("husl", 8))
        for index, value in enumerate(counts):
            plt.text(value, index, str(value))
        plt.show()

plt.savefig(os.path.join('test.png'), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures

if __name__ == "__main__":
    train_labels = Analyzer.count_labels("/home/karaz/Desktop/MovieEntityRecognizer/data/raw/engtrain.bio", without_o=True)
    Analyzer.plot_data(train_labels)
    test_labels = Analyzer.count_labels("/home/karaz/Desktop/MovieEntityRecognizer/data/raw/engtest.bio", without_o=True)
    Analyzer.plot_data(test_labels)