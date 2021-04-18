import os
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
from spacy.lang.en import English
from seqeval.metrics import classification_report

import torch
from torch.utils import tensorboard
from .utils.diceloss import DiceLoss
from .base.base_trainer import Trainer
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class TrainerBiLstm(Trainer):
    """Standard trainer of BiLSTM"""

    def __init__(self, model, data, optimizer_cls, loss_fn_cls, log_name:str):
        self.model = model
        self.data = data
        self.optimizer = optimizer_cls(model.parameters())
        self.loss_fn = loss_fn_cls(ignore_index=self.data.tag_pad_idx)
        self.writer = tensorboard.SummaryWriter(log_name)
        self.train_global = 0
        self.test_global = 0

    @staticmethod
    def epoch_time(start_time:float, end_time:float) -> Tuple[float, float]:
        """
        Eplased epoch time.

        Args:
            start_time (float): Start time
            end_time (float): End time

        Returns:
            Tuple[float,float]: Eplased mins and seconds.
        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def accuracy(self, preds:torch.Tensor, y:torch.Tensor) -> float:
        """
        The function for calculating accuracy. ACC = (TP+TN)/(TP+TN+FP+FN)

        Args:
            preds (torch.Tensor): The tensor of predictions.
            y (torch.Tensor): Ground truths of the same inputs.

        Returns:
            float: The accuracy score.
        """
        max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
        non_pad_elements = (y != self.data.tag_pad_idx).nonzero()  # prepare masking for paddings
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])

        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

    def epoch(self) -> Tuple[float, float]:
        """
        A function that runs one epoch.

        Returns:
            Tuple[float, float]: The loss and accuracy of the epoch.
        """
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()

        for batch in self.data.train_iter:
            self.train_global += 1
            text = batch.word

        true_tags = batch.tag
        self.optimizer.zero_grad()
        pred_tags = self.model(text)

        pred_tags = pred_tags.view(-1, pred_tags.shape[-1])

        true_tags = true_tags.view(-1)
        batch_loss = self.loss_fn(pred_tags, true_tags)
        batch_acc = self.accuracy(pred_tags, true_tags)

        self.writer.add_scalar("Loss/Train", batch_loss, self.train_global)
        self.writer.add_scalar("Accuracy/Train", batch_acc, self.train_global)

        batch_loss.backward()
        self.optimizer.step()

        epoch_loss += batch_loss.item()
        epoch_acc += batch_acc.item()

        return epoch_loss / len(self.data.train_iter), epoch_acc / len(self.data.train_iter)

    @torch.no_grad()
    def evaluate(self, iterator) -> Tuple[float, float, float, float, float, float]:
        """
        The function for evaluation.

        Args:
            iterator ([torch.utils.data.DataLoader]): The DataLoader iterator of the evaluation

        Returns:
            Tuple[float, float, float, float, float, float]: Performance metrics -> loss, acc, precision, recall, f1 mic, f1 mac
        """
        epoch_loss = 0
        epoch_acc = 0
        epoch_pre = 0
        epoch_rec = 0
        epoch_f1mac = 0
        epoch_f1mic = 0

        self.model.eval()

        for batch in iterator:
            self.test_global += 1
            text = batch.word

            true_tags = batch.tag
            pred_tags = self.model(text)

            pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
            true_tags = true_tags.view(-1)

            np_pred = pred_tags.argmax(dim=1, keepdim=True).detach().cpu().numpy()
            np_true = true_tags.detach().cpu().numpy()

            batch_loss = self.loss_fn(pred_tags, true_tags)
            batch_acc = self.accuracy(pred_tags, true_tags)
            batch_pre = precision_score(np_true, np_pred, labels=np.unique(np_pred), average='macro')
            batch_rec = recall_score(np_true, np_pred, labels=np.unique(np_pred), average='macro', zero_division=1)
            batch_f1mac = f1_score(np_true, np_pred, labels=np.unique(np_pred), average='macro')
            batch_f1mic = f1_score(np_true, np_pred, labels=np.unique(np_pred), average='micro')

            self.writer.add_scalar("Loss/test", batch_loss, self.test_global)
            self.writer.add_scalar("Accuracy/test", batch_acc, self.test_global)
            self.writer.add_scalar("F1Macro/test", batch_f1mac, self.test_global)
            self.writer.add_scalar("F1Micro/test", batch_f1mic, self.test_global)

            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
            epoch_pre += batch_pre
            epoch_rec += batch_rec
            epoch_f1mac += batch_f1mac
            epoch_f1mic += batch_f1mic

        return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_pre / len(iterator), \
                epoch_rec / len(iterator), epoch_f1mac / len(iterator), epoch_f1mic / len(iterator)

    # main training sequence
    def train(self, n_epochs:int):
        """
        The entry function for training.

        Args:
            n_epochs ([int]): Number of epochs to run
        """
        # TODO: Change prints with logging or maybe tqdm
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc = self.epoch()
            end_time = time.time()

            epoch_mins, epoch_secs = TrainerBiLstm.epoch_time(start_time, end_time)

            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrn Loss: {train_loss:.3f} | Trn Acc: {train_acc * 100:.2f}%")
            val_loss, val_acc, val_pre, val_rec, val_f1mac, val_f1mic = self.evaluate(self.data.test_iter)

            print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}% | Val Precision: {val_pre * 100:.2f}% \
| Val Recall: {val_rec * 100:.2f}% | Val F1 Macro: {val_f1mac * 100:.2f}% | Val F1 Micro: {val_f1mic * 100:.2f}%")

    @torch.no_grad()
    def infer(self, sentence:str, true_tags=None) -> Tuple:
        """
        The inference of the model.

        Args:
            sentence (str): The sentence that the prediction will be made.
            true_tags ([List], optional): The ground truth tags. Defaults to None.

        Returns:
            Tuple[]: Return tuple that consists of tokens, entity tags and unk tokens.
        """
        self.model.eval()
        # tokenize sentence
        nlp = English()
        tokens = [token.text.lower() for token in nlp(sentence)]

        # transform to indices based on corpus vocab
        numericalized_tokens = [self.data.word_field.vocab.stoi[t] for t in tokens]

        # find unknown words
        unk_idx = self.data.word_field.vocab.stoi[self.data.word_field.unk_token]
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]

        # begin prediction
        token_tensor = torch.LongTensor(numericalized_tokens)
        token_tensor = token_tensor.unsqueeze(-1)
        predictions = self.model(token_tensor)

        # convert results to tags
        top_predictions = predictions.argmax(-1)
        predicted_tags = [self.data.tag_field.vocab.itos[t.item()] for t in top_predictions]

        # print inferred tags
        max_len_token = max([len(token) for token in tokens] + [len("word")])
        max_len_tag = max([len(tag) for tag in predicted_tags] + [len("pred")])

        print(
            f"{'word'.ljust(max_len_token)}\t{'unk'.ljust(max_len_token)}\t{'pred tag'.ljust(max_len_tag)}"
            + ("\ttrue tag" if true_tags else "")
            )

        for i, token in enumerate(tokens):
          is_unk = "✓" if token in unks else ""

        print(
            f"{token.ljust(max_len_token)}\t{is_unk.ljust(max_len_token)}\t{predicted_tags[i].ljust(max_len_tag)}"
            + (f"\t{true_tags[i]}" if true_tags else "")
            )

        return tokens, predicted_tags, unks


class CharBilstmTrainer(object):
    """Standard trainer of CharBiLSTM"""

    def __init__(self, model, data, optimizer_cls, loss_fn_cls, log_file):
        self.model = model
        self.data = data
        self.optimizer = optimizer_cls(model.parameters())
        self.loss_fn = loss_fn_cls(ignore_index=self.data.tag_pad_idx)
        self.train_global = 0
        self.test_global = 0
        self.writer = tensorboard.SummaryWriter(log_file)

    @staticmethod
    def epoch_time(start_time:float, end_time:float) -> Tuple[float, float]:
        """
        Eplased epoch time.

        Args:
            start_time (float): Start time
            end_time (float): End time

        Returns:
            Tuple[float,float]: Eplased mins and seconds.
        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def accuracy(self, preds:torch.Tensor, y:torch.Tensor) -> float:
        """
        The function for calculating accuracy. ACC = (TP+TN)/(TP+TN+FP+FN)

        Args:
            preds (torch.Tensor): The tensor of predictions.
            y (torch.Tensor): Ground truths of the same inputs.

        Returns:
            float: The accuracy score.
        """
        max_preds = preds.argmax(dim=1, keepdim=True)
        non_pad_elements = (y != self.data.tag_pad_idx).nonzero()
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])

        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

    def epoch(self) -> Tuple[float, float]:
        """
        A function that runs one epoch.

        Returns:
            Tuple[float, float]: The loss and accuracy of the epoch.
        """
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()
        self.train_global += 1

        for batch in self.data.train_iter:
            words = batch.word
            chars = batch.char
            true_tags = batch.tag

            self.optimizer.zero_grad()
            pred_tags = self.model(words, chars)

            pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
            true_tags = true_tags.view(-1)

            batch_loss = self.loss_fn(pred_tags, true_tags)
            batch_acc = self.accuracy(pred_tags, true_tags)

            self.writer.add_scalar("Loss/Train", batch_loss, self.train_global)
            self.writer.add_scalar("Accuracy/Train", batch_acc, self.train_global)

            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()

        return epoch_loss / len(self.data.train_iter), epoch_acc / len(self.data.train_iter)

    @torch.no_grad()
    def evaluate(self, iterator) -> Tuple[float, float, float, float, float, float]:
        """
        The function for evaluation.

        Args:
            iterator ([torch.utils.data.DataLoader]): The DataLoader iterator of the evaluation

        Returns:
            Tuple[float, float, float, float, float, float]: Performance metrics -> loss, acc, precision, recall, f1 mic, f1 mac
        """
        epoch_loss = 0
        epoch_acc = 0
        epoch_pre = 0
        epoch_rec = 0
        epoch_f1mac = 0
        epoch_f1mic = 0

        self.model.eval()

        for batch in iterator:
            self.test_global += 1
            words = batch.word
            chars = batch.char
            true_tags = batch.tag

            pred_tags = self.model(words, chars)
            pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
            true_tags = true_tags.view(-1)

            np_pred = pred_tags.argmax(dim=1, keepdim=True).detach().cpu().numpy()
            np_true = true_tags.detach().cpu().numpy()

            batch_loss = self.loss_fn(pred_tags, true_tags)
            batch_acc = self.accuracy(pred_tags, true_tags)
            batch_pre = precision_score(np_true, np_pred, labels=np.unique(np_pred), average='macro')
            batch_rec = recall_score(np_true, np_pred, labels=np.unique(np_pred), average='macro', zero_division=1)
            batch_f1mac = f1_score(np_true, np_pred, labels=np.unique(np_pred), average='macro')
            batch_f1mic = f1_score(np_true, np_pred, labels=np.unique(np_pred), average='micro')

            self.writer.add_scalar("Loss/test", batch_loss, self.test_global)
            self.writer.add_scalar("Accuracy/test", batch_acc, self.test_global)
            self.writer.add_scalar("F1Macro/test", batch_f1mac, self.test_global)
            self.writer.add_scalar("F1Micro/test", batch_f1mic, self.test_global)

            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
            epoch_pre += batch_pre
            epoch_rec += batch_rec
            epoch_f1mac += batch_f1mac
            epoch_f1mic += batch_f1mic

        return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_pre / len(iterator), \
              epoch_rec / len(iterator), epoch_f1mac / len(iterator), epoch_f1mic / len(iterator)

    # main training sequence
    def train(self, n_epochs:int):
        """
        The entry function for training.

        Args:
            n_epochs ([int]): Number of epochs to run
        """
        # TODO: Change prints with logging info or maybe with tqdm
        for epoch in range(n_epochs):
            start_time = time.time()

            train_loss, train_acc = self.epoch()
            end_time = time.time()

            epoch_mins, epoch_secs = CharBilstmTrainer.epoch_time(start_time, end_time)

            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrn Loss: {train_loss:.3f} | Trn Acc: {train_acc * 100:.2f}%")
            val_loss, val_acc, val_pre, val_rec, val_f1mac, val_f1mic = self.evaluate(self.data.test_iter)

            print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}% | Val Precision: {val_pre * 100:.2f}% \
| Val Recall: {val_rec * 100:.2f}% | Val F1 Macro: {val_f1mac * 100:.2f}% | Val F1 Micro: {val_f1mic * 100:.2f}%")

    @torch.no_grad()
    def infer(self, sentence:str, true_tags=None) -> Tuple:
        """
        The inference of the model.

        Args:
            sentence (str): The sentence that the prediction will be made.
            true_tags ([List], optional): The ground truth tags. Defaults to None.

        Returns:
            Tuple[]: Return tuple that consists of tokens, entity tags and unk tokens.
        """
        self.model.eval()
        # tokenize sentence
        nlp = English()
        tokens = [token.text for token in nlp(sentence)]
        max_word_len = max([len(token) for token in tokens])

        # transform to indices based on corpus vocab
        numericalized_tokens = [self.data.word_field.vocab.stoi[token.lower()] for token in tokens]
        numericalized_chars = []
        char_pad_id = self.data.char_pad_idx

        for token in tokens:
            numericalized_chars.append(
                [self.data.char_field.vocab.stoi[char] for char in token]
                + [char_pad_id for _ in range(max_word_len - len(token))]
            )

        unk_idx = self.data.word_field.vocab.stoi[self.data.word_field.unk_token]
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]

        # begin prediction
        token_tensor = torch.as_tensor(numericalized_tokens)
        token_tensor = token_tensor.unsqueeze(-1)
        char_tensor = torch.as_tensor(numericalized_chars)
        char_tensor = char_tensor.unsqueeze(0)
        predictions = self.model(token_tensor, char_tensor)  # MODIFIED

        # convert results to tags
        top_predictions = predictions.argmax(-1)
        predicted_tags = [self.data.tag_field.vocab.itos[t.item()] for t in top_predictions]

        # print inferred tags
        max_len_token = max([len(token) for token in tokens] + [len('word')])
        max_len_tag = max([len(tag) for tag in predicted_tags] + [len('pred')])

        print(
            f"{'word'.ljust(max_len_token)}\t{'unk'.ljust(max_len_token)}\t{'pred tag'.ljust(max_len_tag)}"
            + ("\ttrue tag" if true_tags else "")
        )

        for i, token in enumerate(tokens):
            is_unk = "✓" if token in unks else ""

            print(
                f"{token.ljust(max_len_token)}\t{is_unk.ljust(max_len_token)}\t{predicted_tags[i].ljust(max_len_tag)}"
                + (f"\t{true_tags[i]}" if true_tags else "")
            )

        return tokens, predicted_tags, unks