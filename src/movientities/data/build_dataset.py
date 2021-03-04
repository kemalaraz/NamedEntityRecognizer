import os
from typing import Dict, List

import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import SequenceTaggingDataset

class BuildData:

    __datanames__ = ["train.txt", "dev.txt", "test.txt"]

    @staticmethod
    def create_finaldata(source:str, outpath:str, splits:str=" "):
        """
        Transform the data to the conll format.

        Args:
            source (str): The raw data path
            outpath (str): modified data path
        """
        assert outpath.split("/")[-1] in BuildData.__datanames__,\
                f"Finalized data name must be one of these -> {BuildData.__datanames__}"

        with open(source, "r") as f:
            with open(outpath, "w") as fw:
                for line in f.readlines():
                    if line != "\n":
                        line = line.split("\t")
                        text = line[1].strip("\n")
                        entity = line[0]
                        fw.write(text + splits + entity + "\n")
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
            if file in BuildData.__datanames__:
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

class Corpus(object):

  def __init__(self, input_folder, min_word_freq, batch_size):
    # list all the fields
    self.word_field = Field(lower=True)
    self.tag_field = Field(unk_token=None)
    # create dataset using built-in parser from torchtext
    self.train_dataset, self.test_dataset = SequenceTaggingDataset.splits(
        path=input_folder,
        train="train.txt",
        test="test.txt",
        fields=(("word", self.word_field), ("tag", self.tag_field))
    )
    # convert fields to vocabulary list
    self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
    self.tag_field.build_vocab(self.train_dataset.tag)
    # create iterator for batch input
    self.train_iter, self.test_iter = BucketIterator.splits(
        datasets=(self.train_dataset, self.test_dataset),
        batch_size=batch_size
    )
    # prepare padding index to be ignored during model training/evaluation
    self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
    self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]


if __name__ == "__main__":
    dataset = BuildData.create_dataset("/home/kemalaraz/Desktop/MovieEntityRecognizer/data/modified/mitmovies")
    print(list(zip(dataset["train_instances"][0], dataset["train_labels"][0])))
    print(list(zip(dataset["test_instances"][0], dataset["test_labels"][0])))
