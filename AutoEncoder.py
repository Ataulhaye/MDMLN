import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# Building the encoder
class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim1,
        hidden_dim2,
        embedding_dim,
    ):
        super().__init__()
        # encoder part
        self.relu = nn.ReLU()
        self.enc_l1 = nn.Linear(input_dim, hidden_dim1)
        # self.enc_l1 = nn.Conv1d(input_dim, hidden_dim1)
        self.enc_l2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.enc_l3 = nn.Linear(hidden_dim2, embedding_dim)

        # decoder part
        self.dec_l1 = nn.Linear(embedding_dim, hidden_dim2)
        self.dec_l2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.dec_l3 = nn.Linear(hidden_dim1, input_dim)

    def encoder(self, x):
        x = self.relu(self.enc_l1(x))
        x = self.relu(self.enc_l2(x))
        x = self.enc_l3(x)
        return x

    def decoder(self, x):
        x = self.relu(self.dec_l1(x))
        x = self.relu(self.dec_l2(x))
        x = torch.sigmoid(self.dec_l3(x))
        return x

    def forward(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return encoding, decoding
        # return Codings(encoding, decoding)
