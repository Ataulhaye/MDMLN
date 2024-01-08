import os
import tempfile
from functools import partial

import numpy as np
import ray.cloudpickle as pickle
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from ray import train, tune

# from ray.air import Checkpoint, session
from ray.air import session
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import random_split
from torchvision import datasets, transforms

from AutoEncoder import Autoencoder
from Brain import Brain
from BrainDataLabel import BrainDataLabel
from DataTraining import DataTraining
from ExportData import ExportData
from TrainingConfig import TrainingConfig


def train_and_validate_mnist_ray_tune(config, data_dir=None):
    print("Config:Tes", config)
    # net = Autoencoder(config["input_dim"],config["hidden_dim1"],config["hidden_dim2"],config["hidden_dim3"],config["hidden_dim4"],config["embedding_dim"],)
    net = Autoencoder(
        config["input_dim"],
        config["hidden_dim1"],
        config["hidden_dim2"],
        config["hidden_dim3"],
        config["hidden_dim4"],
        config["embedding_dim"],
    )
    # net = Autoencoder(784, 512, 128, 64, 32, 10)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])

    # checkpoint = session.get_checkpoint()

    # if checkpoint:
    # checkpoint_state = checkpoint.to_dict()
    # start_epoch = checkpoint_state["epoch"]
    # net.load_state_dict(checkpoint_state["net_state_dict"])
    # optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    # else:
    # start_epoch = 0

    # start_epoch = 0
    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    # trainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    # valloader = torch.utils.data.DataLoader(val_subset, batch_size=128, shuffle=True)
    net.train()
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            inputs = torch.reshape(inputs, (-1, 784))
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(inputs)
            loss = loss_function(output, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        net.eval()
        for i, data in enumerate(valloader):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = torch.reshape(inputs, (-1, 784))
                output = net(inputs)
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = loss_function(output, inputs)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # checkpoint_data = {"epoch": epoch,"net_state_dict": net.state_dict(),"optimizer_state_dict": optimizer.state_dict(),}
        # checkpoint = Checkpoint.from_dict(checkpoint_data)

        # session.report({"loss": val_loss / val_steps, "accuracy": correct / total},checkpoint=checkpoint,)
        metrics = {"loss": val_loss / val_steps, "accuracy": correct / total}

        # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        # rank = train.get_context().get_world_rank()
        # torch.save(
        # (net.state_dict(), optimizer.state_dict()),
        # os.path.join(temp_checkpoint_dir, f"model-rank={rank}.pt"),
        # )
        # checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        # train.report(metrics, checkpoint=checkpoint)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None

            should_checkpoint = epoch % config.get("checkpoint_freq", 1) == 0
            # In standard DDP training, where the model is the same across all ranks,
            # only the global rank 0 worker needs to save and report the checkpoint
            if train.get_context().get_world_rank() == 0 and should_checkpoint:
                torch.save(
                    (net.state_dict(), optimizer.state_dict()),
                    os.path.join(temp_checkpoint_dir, "model.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            train.report(metrics, checkpoint=checkpoint)

    print("Finished Training")


def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images = torch.reshape(images, (-1, 784))
            output = net(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def load_data(data_dir="./mnist_data/"):
    trainset = datasets.MNIST(
        root=data_dir, train=True, transform=transforms.ToTensor(), download=True
    )
    testset = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )

    return trainset, testset
