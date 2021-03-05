import torch
from torch import nn

class CharBilstm(nn.Module):

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 char_emb_dim,  # NEWLY ADDED
                 char_input_dim,  # NEWLY ADDED
                 char_cnn_filter_num,  # NEWLY ADDED
                 char_cnn_kernel_size,  # NEWLY ADDED
                 hidden_dim,
                 output_dim,
                 lstm_layers,
                 emb_dropout,
                 cnn_dropout,  # NEWLY ADDED
                 lstm_dropout,
                 fc_dropout,
                 word_pad_idx,
                 char_pad_idx):  # NEWLY ADDED
        super().__init__()
        # LAYER 1A: Word Embedding
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=word_pad_idx
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        ### BEGIN MODIFIED SECTION: CHARACTER EMBEDDING ###
        # LAYER 1B: Char Embedding-CNN
        self.char_emb_dim = char_emb_dim
        self.char_emb = nn.Embedding(
            num_embeddings=char_input_dim,
            embedding_dim=char_emb_dim,
            padding_idx=char_pad_idx
        )
        self.char_cnn = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=char_emb_dim * char_cnn_filter_num,
            kernel_size=char_cnn_kernel_size,
            groups=char_emb_dim  # different 1d conv for each embedding dim
        )
        self.cnn_dropout = nn.Dropout(cnn_dropout)
        ### END MODIFIED SECTION ###
        # LAYER 2: BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim + (char_emb_dim * char_cnn_filter_num),
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )
        # LAYER 3: Fully-connected
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # times 2 for bidirectional
        # init weights from normal distribution
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def forward(self, words, chars):
        # words = [sentence length, batch size]
        # chars = [batch size, sentence length, word length)
        # embedding_out = [sentence length, batch size, embedding dim]
        embedding_out = self.emb_dropout(self.embedding(words))
        char_emb_out = self.emb_dropout(self.char_emb(chars))
        batch_size, sent_len, word_len, char_emb_dim = char_emb_out.shape
        char_cnn_max_out = torch.zeros(batch_size, sent_len, self.char_cnn.out_channels)
        # for character embedding, we need to iterate over sentences
        for sent_i in range(sent_len):
            # sent_char_emb = [batch size, word length, char emb dim]
            sent_char_emb = char_emb_out[:, sent_i, :, :]  # get the character field of sent i
            # sent_char_emb_p = [batch size, char emb dim, word length]
            sent_char_emb_p = sent_char_emb.permute(0, 2, 1)  # the channel (char emb dim) has to be the last dimension
            # char_cnn_sent_out = [batch size, out channels * char emb dim, word length - kernel size + 1]
            char_cnn_sent_out = self.char_cnn(sent_char_emb_p)
            char_cnn_max_out[:, sent_i, :], _ = torch.max(char_cnn_sent_out, dim=2)  # max pooling over the word length dimension
        char_cnn = self.cnn_dropout(char_cnn_max_out)
        # concat word and char embedding
        # char_cnn_p = [sentence length, batch size, char emb dim * num filter]
        char_cnn_p = char_cnn.permute(1, 0, 2)
        word_features = torch.cat((embedding_out, char_cnn_p), dim=2)
        ### END MODIFIED SECTION ###
        # lstm_out = [sentence length, batch size, hidden dim * 2]
        lstm_out, _ = self.lstm(word_features)
        # ner_out = [sentence length, batch size, output dim]
        ner_out = self.fc(self.fc_dropout(lstm_out))
        return ner_out

    def init_embeddings(self, char_pad_idx, word_pad_idx, pretrained=None, freeze=True):
        # initialize embedding for padding as zero
        self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)
        self.char_emb.weight.data[char_pad_idx] = torch.zeros(self.char_emb_dim)
        if pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=torch.as_tensor(pretrained),
                padding_idx=word_pad_idx,
                freeze=freeze
            )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)