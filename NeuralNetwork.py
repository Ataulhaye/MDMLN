# import the Torch package to save your day
# transforms are used to preprocess the images, e.g. crop, rotate, normalize, etc
# package we used to manipulate matrix
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# package we used for image processing
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

from AutoEncoder import Autoencoder
from Brain import Brain
from BrainDataConfig import BrainDataConfig
from BrainDataLabel import BrainDataLabel
from DataTraining import DataTraining
from ExportData import ExportData
from TrainingConfig import TrainingConfig

# specific the data path in which you would like to store the downloaded files
# here, we save it to the folder called "mnist_data"
# ToTensor() here is used to convert data type to tensor, so that can be used in network

train_dataset = datasets.MNIST(
    root="./mnist_data/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="./mnist_data/", train=False, transform=transforms.ToTensor(), download=True
)

print(train_dataset)
print(test_dataset)
batchSize = 128

# only after packed in DataLoader, can we feed the data into the neural network iteratively
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batchSize, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batchSize, shuffle=False
)

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_hidden_1 = 128  # 256  # 1st layer num features
num_hidden_2 = 16  # 128  # 2nd layer num features
embedding_dim = 2  # (the latent dim)

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_hidden_1 = 256  # 256  # 1st layer num features
num_hidden_2 = 128  # 128  # 2nd layer num features
num_hidden_3 = 64
num_hidden_4 = 32
embedding_dim = 9  # (the latent dim)


model = Autoencoder(
    num_input, num_hidden_1, num_hidden_2, num_hidden_3, num_hidden_4, embedding_dim
)
# If using GPU, model need to be set on cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


def train_model(model, device, training_data, epochs=10):
    # define loss and parameters
    optimizer = optim.Adam(model.parameters())
    # MSE loss will calculate Mean Squared Error between the inputs
    loss_function = nn.MSELoss()
    train_loss = []
    accuracies = []
    print("====Training start====")
    model.train()
    for epoch in range(epochs):
        epoc_accuracy = 0
        train_epoch_loss = 0.0
        for data, labels in training_data:
            # prepare input data
            data = data.to(device)
            inputs = torch.reshape(
                data, (-1, 784)
            )  # -1 can be any value. So when reshape, it will satisfy 784 first
            # set gradient to zero
            optimizer.zero_grad()

            # feed inputs into model
            model_output = model(inputs)

            # calculating loss
            loss = loss_function(model_output.decoding, inputs)
            correct = torch.sum(
                labels == torch.argmax(model_output.decoding.detach().cpu(), dim=1)
            ).item()
            epoc_accuracy += correct
            # calculate gradient of each parameter
            loss.backward()
            train_epoch_loss += loss.item()

            # update the weight based on the gradient calculated
            optimizer.step()
        train_loss.append(train_epoch_loss)
        accuracies.append(epoc_accuracy)
    print("====Training finish====")
    return train_loss


def test_model(model, device, testing_data, batches=39):
    # define loss and parameters
    optimizer = optim.Adam(model.parameters())
    # MSE loss will calculate Mean Squared Error between the inputs
    loss_function = nn.MSELoss()
    validation_loss = []
    embeddings = []
    all_labels = []
    print("====Testing start====")
    model.eval()
    for batch in range(batches):
        valid_batch_loss = 0.0
        for data, labels in testing_data:
            # prepare input data
            data = data.to(device)
            inputs = torch.reshape(
                data, (-1, 784)
            )  # -1 can be any value. So when reshape, it will satisfy 784 first

            model_output = model(inputs)

            # calculating loss
            loss = loss_function(model_output.decoding, inputs)

            valid_batch_loss += loss.item()

            for i, emb in enumerate(model_output.encoding):
                embeddings.append(emb.cpu().detach().numpy())
                all_labels.append(labels[i].item())
        validation_loss.append(valid_batch_loss)
    print("====Testing finish====")
    return validation_loss, embeddings, all_labels


def mnist_classification(classifiers, strategies, t_config: TrainingConfig, X, y):
    brain = Brain(area="MNIST")
    brain.voxels = np.array(X)
    la = BrainDataLabel(name="mnist_labels", popmean=(1 / 9), labels=np.array(y))
    brain.current_labels = la

    training = DataTraining()

    export_data = training.brain_data_classification(
        brain, t_config, strategies, classifiers
    )

    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=export_data,
        sheet_name=f"{brain.area}-Results",
        title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
    )


t_loss = train_model(model=model, device=device, training_data=train_loader, epochs=80)
plt.plot(t_loss)
v_loss, X, y = test_model(
    model=model, device=device, testing_data=test_loader, batches=40
)
plt.plot(v_loss)

strategies = ["Nothing"]
classifiers = ["SVM", "MLP", "LinearDiscriminant"]
t_config = TrainingConfig()
# t_config.nan_classifiers = []
# t_config.folds = 1
# t_config.explain = True
t_config.dimension_reduction = False
t_config.predefined_split = False

mnist_classification(classifiers, strategies, t_config, X, y)
