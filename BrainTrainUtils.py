import os
import pickle
import tempfile
from math import ceil

import torch
import torch.nn as nn
import torch.optim as optim
from ray import train
from ray.train import Checkpoint
from torch.utils.data import DataLoader, TensorDataset

from AutoEncoder import Autoencoder
from Brain import Brain
from BrainDataConfig import BrainDataConfig
from BrainDataLabel import BrainDataLabel
from DataTraining import DataTraining
from TestTrainingSet import TestTrainingSet, TestTrainingTensorDataset
from TrainingConfig import TrainingConfig


def train_and_validate_brain_voxels_ray(config, set: TestTrainingTensorDataset):
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

    train_subset = torch.Tensor(set.X_train)
    val_subset = torch.Tensor(set.X_test)

    train_subset = TensorDataset(train_subset)
    val_subset = TensorDataset(val_subset)

    trainloader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4,
    )
    valloader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4,
    )

    train_loss = []
    valid_loss = []
    train_steps = 0
    model.train()
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs = data[0].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(inputs)
            loss = loss_function(output, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_steps += 1

        train_loss.append(running_loss)

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        model.eval()
        for i, data in enumerate(valloader):
            with torch.no_grad():
                inputs = data[0].to(device)
                output = model(inputs)

                loss = loss_function(output, inputs)
                val_loss += loss.cpu().numpy()
                val_steps += 1
            valid_loss.append(val_loss)

        metrics = {
            "t_loss": running_loss / train_steps,
            "v_loss": val_loss / val_steps,
            "epoch": epoch,
        }

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "t_loss": running_loss / train_steps,
                    "v_loss": val_loss / val_steps,
                    "t_loss_list": train_loss,
                    "v_loss_list": valid_loss,
                },
                os.path.join(temp_checkpoint_dir, "model.pt"),
            )
            print("Checkpoint Saved")
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(metrics, checkpoint=checkpoint)
            print("Checkpoint Reported")

    print("Finished Training")


def test_accuracy(net, testset, device="cpu"):
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    loss_function = nn.MSELoss()
    acc_loss = 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            inputs = data[0].to(device)
            output = net(inputs)
            loss = loss_function(output, inputs)
            acc_loss += loss.cpu().numpy()
            tloss = loss.cpu().numpy()
            t1loss = loss.item()
    print("Best trial test set validation loss: {}".format(acc_loss))
    return acc_loss


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


def train_and_validate_brain_voxels(config, tensor_set: TestTrainingTensorDataset):
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
        tensor_set.test_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4,
    )

    train_loss = []
    valid_loss = []
    train_steps = 0
    model.train()
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs = data[0].to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(inputs)
            loss = loss_function(output, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_steps += 1

        train_loss.append(running_loss)

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        model.eval()
        for i, data in enumerate(valloader):
            with torch.no_grad():
                inputs = data[0].to(device)
                output = model(inputs)

                loss = loss_function(output, inputs)
                val_loss += loss.cpu().numpy()
                val_steps += 1
            valid_loss.append(val_loss)

        metrics = {
            "t_loss": running_loss / train_steps,
            "v_loss": val_loss / val_steps,
            "epoch": epoch,
        }

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "t_loss": running_loss / train_steps,
                    "v_loss": val_loss / val_steps,
                    "t_loss_list": train_loss,
                    "v_loss_list": valid_loss,
                },
                os.path.join(temp_checkpoint_dir, "model.pt"),
            )
            print("Checkpoint Saved")
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(metrics, checkpoint=checkpoint)
            print("Checkpoint Reported")

    print("Finished Training")


def tes():
    bd_config = BrainDataConfig()
    brain = Brain(
        area=bd_config.STG,
        data_path=bd_config.STG_path,
        load_labels=True,
        load_int_labels=True,
    )
    brain.current_labels = brain.subject_labels_int

    train_config = TrainingConfig()
    train_config.strategy = "mean"

    tt_set = DataTraining().premeditate_random_train_test_split_n(brain, train_config)

    brain.normalize_data_safely(strategy=train_config.strategy, data_set=tt_set)

    file_name = f"{brain.area}_{train_config.strategy}_static_wholeSet.pickle"
    with open(file_name, "wb") as output:
        pickle.dump(tt_set, output)

    # with open(file_name, "rb") as data:
    # static_dataset = pickle.load(data)

    modify_brain = Brain()
    modify_brain.voxels = tt_set.X_train
    current_labels = BrainDataLabel(
        brain.current_labels.name, brain.current_labels.popmean, tt_set.y_train
    )
    modify_brain.current_labels = current_labels
    modify_bd_config = BrainDataConfig()
    modify_patients = []
    for subset_size in bd_config.patients:
        n_test = ceil(subset_size * train_config.test_size)
        modify_patients.append(subset_size - n_test)
    modify_bd_config.patients = modify_patients

    modify_tt_set = DataTraining().premeditate_random_train_test_split_n(
        modify_brain, train_config, modify_bd_config
    )

    file_name = f"{brain.area}_{train_config.strategy}_static_subSet.pickle"
    with open(file_name, "wb") as output:
        pickle.dump(modify_tt_set, output)

    XT_train = torch.Tensor(modify_tt_set.X_train)
    XT_test = torch.Tensor(modify_tt_set.X_test)
    yT_train = torch.Tensor(modify_tt_set.y_train)
    yT_test = torch.Tensor(modify_tt_set.y_test)

    tr_set = TensorDataset(XT_train, yT_train)
    ts_set = TensorDataset(XT_test, yT_test)

    return TestTrainingTensorDataset(train_set=tr_set, test_set=ts_set)
