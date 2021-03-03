import os
from typing import Dict

import torch

class BuildData:

    __datanames__ = ["train.txt", "dev.txt", "test.txt"]

    @staticmethod
    def create_finaldata(source:str, outpath:str):
        """
        Transform the data to the conll format.

        Args:
            source (str): The raw data path
            outpath (str): modified data path
        """
        assert outpath.split("/")[-1] not in BuildData.__datanames__,\
                f"Finalized data name must be one of these -> {BuildData.__datanames__}"

        with open(source, "r") as f:
            with open(outpath, "w") as fw:
                for line in f.readlines():
                    if line != "\n":
                        line = line.split("\t")
                        text = line[1].strip("\n")
                        entity = line[0]
                        fw.write(text + " " + entity + "\n")
                    else:
                        fw.write("\n")

    @staticmethod
    def create_dataset(filepath:str) -> Dict:
        """
        Create the dataset where the dict has instances and corresponding labels.

        Args:
            filepath (str): the folder that has __dataset__ files.

        Returns:
            Dict: Dataset dictionary.
        """

        assert os.path.isdir(filepath), f"The given folder path is not a directory {filepath}"
        path = list(*os.walk(filepath))
        assert any(True for p in path[2] if p in BuildData.__datanames__),\
                    f"Files that are not in {BuildData.__datanames__}"
        dataset = {}
        for file in path[2]:
            text, label = [], []
            texts, labels = [], []
            with open(os.path.join(path[0], file)) as f:
                for line in f.readlines():
                    if line != "\n":
                        line = line.split(" ")
                        text.append(line[0])
                        label.append(line[-1][:-1])
                    else:
                        texts.append(text)
                        labels.append(label)
                        text, label = [], []
                name = file.split(".")[0]
                dataset[name + "_instances"] = texts
                dataset[name + "_labels"] = labels
        return dataset

    @staticmethod
    def prepare_data_bilstmcrf(sequence:List, word2idx:Dict) -> torch.Tensor:
        idxs = [word2idx[w] for w in sequence]
        return torch.tensor(idxs, dtype=torch.long)


if __name__ == "__main__":
    dataset = BuildData.create_dataset("/home/kemalaraz/Desktop/MovieEntityRecognizer/data/modified/mitmovies")
    print(list(zip(dataset["train_instances"][0], dataset["train_labels"][0])))
    print(list(zip(dataset["test_instances"][0], dataset["test_labels"][0])))