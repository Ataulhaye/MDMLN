import os
import pickle
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ray import train
from ray.train import Checkpoint
from torch.utils.data import DataLoader, TensorDataset

from AutoEncoder import Autoencoder
from ExportData import ExportData
from TestTrainingSet import TestTrainingTensorDataset


def test_autoencode_braindata(net, testset: TensorDataset, device="cpu"):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    loss_function = nn.MSELoss()
    acc_loss = 0.0
    net.eval()
    cnt = 0
    test_encodings = []
    test_labels = []
    with torch.no_grad():
        for data in testloader:
            cnt = cnt + 1
            voxels, labels = data
            voxels = voxels.to(device)
            output = net(voxels)
            encodings = net.encoder(voxels)
            for encoding in encodings:
                test_encodings.append(encoding.cpu().tolist())
            for label in labels:
                test_labels.append(label.item())
            loss = loss_function(output, voxels)
            acc_loss += loss.item()
    print("Counter:", cnt)
    print("testset:", testset.tensors[0].shape)
    print("Best trial test set validation loss: {}".format(acc_loss))
    return test_encodings, test_labels


def generate_model(config):
    model = Autoencoder(
        config["input_dim"],
        config["hidden_dim1"],
        config["hidden_dim2"],
        config["hidden_dim3"],
        config["hidden_dim4"],
        config["embedding_dim"],
    )
    return model


def train_and_validate_autoencode_braindata(
    config, tensor_set: TestTrainingTensorDataset
):
    model = generate_model(config)
    print("Training Config", config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    trainloader = DataLoader(
        tensor_set.train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4,
    )
    valloader = DataLoader(
        tensor_set.val_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4,
    )

    train_loss = []
    valid_loss = []
    train_encodings = []
    train_labels = []
    train_steps = 0
    model.train()
    for epoch in range(config["epochs"]):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            voxels, labels = data
            voxels = voxels.to(device)
            optimizer.zero_grad()

            output = model(voxels)
            encodings = model.encoder(voxels)
            for encoding in encodings:
                train_encodings.append(encoding.cpu().tolist())
            for label in labels:
                train_labels.append(label.item())
            loss = loss_function(output, voxels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_steps += 1

        train_loss.append(running_loss)

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        model.eval()
        for i, data in enumerate(valloader):
            with torch.no_grad():
                voxels, labels = data
                voxels = voxels.to(device)
                output = model(voxels)

                encodings = model.encoder(voxels)
                for encoding in encodings:
                    train_encodings.append(encoding.cpu().tolist())
                for label in labels:
                    train_labels.append(label.item())

                loss = loss_function(output, voxels)
                val_loss += loss.item()
                val_steps += 1
            valid_loss.append(val_loss)

    model_config = {
        "lr": config["lr"],
        "input_dim": config["input_dim"],
        "hidden_dim1": config["hidden_dim1"],
        "hidden_dim2": config["hidden_dim2"],
        "hidden_dim3": config["hidden_dim3"],
        "hidden_dim4": config["hidden_dim4"],
        "embedding_dim": config["embedding_dim"],
        "batch_size": config["batch_size"],
    }
    name = ExportData.get_file_name("_model.pt", config["brain_area"])
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "t_loss": running_loss / train_steps,
            "v_loss": val_loss / val_steps,
            "t_loss_list": train_loss,
            "v_loss_list": valid_loss,
            "train_steps": train_steps,
            "val_steps": val_steps,
            "config": model_config,
            "train_set": tensor_set.train_set.tensors[0].shape,
            "validation_set": tensor_set.val_set.tensors[0].shape,
        },
        name,
    )
    print("Model Saved")

    print("Finished Training")
    print("Training loss", train_loss)
    # plt.plot(train_loss)
    # plt.savefig("test", dpi=700)
    # plt.close()
    return (model, train_encodings, train_labels)


def load_bestmodel_and_test(model_path, device, gpus_per_trial):
    checkpoint_path = model_path + "/model.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    best_trained_model = generate_model(checkpoint["config"])

    if device == "cuda:0" and gpus_per_trial > 1:
        best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_trained_model.load_state_dict(checkpoint["model_state"])

    file_name = ""
    static_dataset = None
    with open(file_name, "rb") as data:
        static_dataset = pickle.load(data)
    test_acc = test_autoencode_braindata(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))
