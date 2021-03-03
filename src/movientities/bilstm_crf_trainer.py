import torch
from .models.lstm import BiLSTM_CRF
from .utils.diceloss import DiceLoss
from .base.base_trainer import Trainer

class Trainer_BiLstm(Trainer):

    def __init__(self, embedding_dim:int, hidden_dim:int, tokenizer:Dict, label_map:Dict,
                model=None, dice_loss:bool=False, optimizer_name:str="SGD", lr:int=0.1, weight_decay:int=1e-4):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_map = label_map
        if dice_loss:
            self.loss_fn = DiceLoss()

        if not model:
            self.model = BiLSTM_CRF
        else:
            self.model = model

        if optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr, weight_decay)
        elif optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay)

    def train(self):
        pass

    def evaluate(self):
        pass
