import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
import torchvision
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# import torch.nn as nn
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)


class Encoder(nn.Module):
    def __init__(
        self, input_dim=28 * 28, hidden_dim1=128, hidden_dim2=16, embedding_dim=2
    ):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim1)
        self.l2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.l3 = nn.Linear(hidden_dim2, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, output_dim=28 * 28, hidden_dim1=128, hidden_dim2=16, embedding_dim=2
    ):
        super().__init__()
        self.l1 = nn.Linear(embedding_dim, hidden_dim2)
        self.l2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.l3 = nn.Linear(hidden_dim1, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# enc = None
# dec = None
# if torch.cuda.is_available():
#    enc = Encoder().cuda()
#    dec = Encoder().cuda()
# else:
#    enc = Encoder().cpu()
#    dec = Encoder().cpu()

enc = Encoder().to(device)
dec = Decoder().to(device)

train_dl = DataLoader(train_data, batch_size=100)

loss_fn = nn.MSELoss()
optimizer_enc = torch.optim.Adam(enc.parameters())
optimizer_dec = torch.optim.Adam(dec.parameters())

train_loss = []
num_epochs = 10

for epoch in range(num_epochs):
    train_epoch_loss = 0
    for imgs, labels in train_dl:
        imgs = imgs.to(device)
        # 100 , 1 , 28 , 28 ---> (100 , 28*28)
        imgs = imgs.flatten(1)
        embedding = enc(imgs)
        output = dec(embedding)
        loss = loss_fn(output, imgs)
        # Calculate Loss
        train_epoch_loss += loss.cpu().detach().numpy()
        # Clear the gradients
        # clear out the gradients of all Variables
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        # Calculate gradients
        loss.backward()
        # Update Weights
        optimizer_enc.step()
        optimizer_dec.step()
    train_loss.append(train_epoch_loss)

# plt.plot(train_loss)
print("")

encodings = None
all_labels = []

for imgs, labels in train_dl:
    imgs = imgs.to(device)
    imgs = imgs.flatten(1)
    all_labels.extend(list(labels.numpy()))
    encoding = enc(imgs)
    if encodings is None:
        encodings = encoding.cpu()
    else:
        encodings = torch.vstack([encodings, encoding.cpu()])

# cmap = plt.get_cmap("viridis", 10)


all_labels = np.array(all_labels)
encodings = encodings.detach().numpy()

# pc = plt.scatter(encodings[:, 0], encodings[:, 1], c=all_labels, cmap=cmap)
# plt.colorbar(pc)
# plt.savefig("Test1.svg", dpi=700)
# T-Distributed Stochastic Neighbor Embedding (t-SNE)
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(encodings)
print("t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))

df_tsne = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2"])
# Add labels
df_tsne["labels"] = all_labels

# df_tsne["TSNE1"] = tsne_results[:, 0]
# df_tsne["TSNE2"] = tsne_results[:, 1]
print(df_tsne)
plt.figure(figsize=(16, 10))
sns.scatterplot(
    data=df_tsne,
    x="TSNE1",
    y="TSNE2",
    hue="labels",
    palette=sns.color_palette("hls", 10),
    legend="full",
    alpha=0.3,
)


fig = px.scatter(df_tsne, x="TSNE1", y="TSNE2", color="labels")
fig.show()

# pc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap=cmap)
# plt.colorbar(pc)
# plt.savefig("Testtsne.svg", dpi=700)
#######################
fig = px.scatter_3d(
    df_tsne, x="TSNE1", y="TSNE2", color=df_tsne["labels"], labels={"color": "labels"}
)

fig.update_traces(marker_size=8)
fig.show()

#######################
fig = px.scatter_3d(df_tsne, x="TSNE1", y="TSNE2", color="labels")
fig.show()


print("")
