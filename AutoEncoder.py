import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from Codings import Codings


# Building the encoder
class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim1,
        hidden_dim2,
        hidden_dim3,
        hidden_dim4,
        embedding_dim,
    ):
        super().__init__()
        # encoder part
        # self.fc1 = nn.Linear(input_dim, h_dim1)
        # self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.relu = nn.ReLU()
        self.enc_l1 = nn.Linear(input_dim, hidden_dim1)
        self.enc_l2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.enc_l3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.enc_l4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.enc_l5 = nn.Linear(hidden_dim4, embedding_dim)

        # decoder part
        # self.fc3 = nn.Linear(h_dim2, h_dim1)
        # self.fc4 = nn.Linear(h_dim1, input_dim)
        self.dec_l1 = nn.Linear(embedding_dim, hidden_dim4)
        self.dec_l2 = nn.Linear(hidden_dim4, hidden_dim3)
        self.dec_l3 = nn.Linear(hidden_dim3, hidden_dim2)
        self.dec_l4 = nn.Linear(hidden_dim2, hidden_dim1)
        self.dec_l5 = nn.Linear(hidden_dim1, input_dim)

    def encoder(self, x):
        # x = torch.sigmoid(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        x = self.relu(self.enc_l1(x))
        x = self.relu(self.enc_l2(x))
        x = self.relu(self.enc_l3(x))
        x = self.relu(self.enc_l4(x))
        x = self.enc_l5(x)
        # x = torch.sigmoid(self.enc_l5(x))
        return x

    def decoder(self, x):
        # x = torch.sigmoid(self.fc3(x))
        # x = torch.sigmoid(self.fc4(x))
        x = self.relu(self.dec_l1(x))
        x = self.relu(self.dec_l2(x))
        x = self.relu(self.dec_l3(x))
        x = self.relu(self.dec_l4(x))
        x = torch.sigmoid(self.dec_l5(x))
        return x

    def forward(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return Codings(encoding, decoding)
