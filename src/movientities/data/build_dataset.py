import os
import gensim
from collections import Counter
from typing import Dict, List

import torch
from torchtext.vocab import Vocab
from torchtext.data import Field, BucketIterator, NestedField
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

class CharCorpus(object):

    def __init__(self, input_folder, min_word_freq, batch_size, wv_file=None):
        # list all the fields
        self.word_field = Field(lower=True)  # [sent len, batch_size]
        self.tag_field = Field(unk_token=None)  # [sent len, batch_size]
        ### BEGIN MODIFIED SECTION: CHARACTER EMBEDDING ###
        self.char_nesting_field = Field(tokenize=list)
        self.char_field = NestedField(self.char_nesting_field)  # [batch_size, sent len, word len]
        # create dataset using built-in parser from torchtext
        self.train_dataset, self.test_dataset = SequenceTaggingDataset.splits(
            path=input_folder,
            train="train.txt",
            test="test.txt",
            fields=(
                (("word", "char"), (self.word_field, self.char_field)),
                ("tag", self.tag_field)
            )
        )
        ### END MODIFIED SECTION ###
        # convert fields to vocabulary list
        if wv_file:
            self.wv_model = gensim.models.word2vec.Word2Vec.load(wv_file)
            self.embedding_dim = self.wv_model.vector_size
            word_freq = {word: self.wv_model.wv.vocab[word].count for word in self.wv_model.wv.vocab}
            word_counter = Counter(word_freq)
            self.word_field.vocab = Vocab(word_counter, min_freq=min_word_freq)
            vectors = []
            for word, idx in self.word_field.vocab.stoi.items():
                if word in self.wv_model.wv.vocab.keys():
                    vectors.append(torch.as_tensor(self.wv_model.wv[word].tolist()))
                else:
                    vectors.append(torch.zeros(self.embedding_dim))
            self.word_field.vocab.set_vectors(
                stoi=self.word_field.vocab.stoi,
                vectors=vectors,
                dim=self.embedding_dim
            )
        else:
            self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
        # build vocab for tag and characters
        self.char_field.build_vocab(self.train_dataset.char)  # NEWLY ADDED
        self.tag_field.build_vocab(self.train_dataset.tag)
        # create iterator for batch input
        self.train_iter, self.test_iter = BucketIterator.splits(
            datasets=(self.train_dataset, self.test_dataset),
            batch_size=batch_size
        )
        # prepare padding index to be ignored during model training/evaluation
        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.char_pad_idx = self.char_field.vocab.stoi[self.char_field.pad_token]  # NEWLY ADDED
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]


if __name__ == "__main__":
    dataset = BuildData.create_dataset("/home/kemalaraz/Desktop/MovieEntityRecognizer/data/modified/mitmovies")
    print(list(zip(dataset["train_instances"][0], dataset["train_labels"][0])))
    print(list(zip(dataset["test_instances"][0], dataset["test_labels"][0])))
