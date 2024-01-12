import os
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from ray import train, tune

# from ray.air import Checkpoint, session
from ray.air import session
from ray.train import Checkpoint
from torch.utils.data import random_split
from torchvision import datasets, transforms

from AutoEncoder import Autoencoder


def train_and_validate_mnist_Test_method(config, data_dir=None):
    #'input_dim': 784, 'hidden_dim1': 64, 'hidden_dim2': 128, 'hidden_dim3': 256, 'hidden_dim4': 256, 'embedding_dim': 8, 'lr': 0.0009231555955597683, 'batch_size': 2
    net = Autoencoder(784, 64, 128, 256, 256, 8)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0009231555955597683)

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
    batch_size = 128
    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=True
    )

    # Load existing checkpoint
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    train_loss = []
    valid_loss = []
    net.train()
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):
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

            running_loss += loss.item()

            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        train_loss.append(running_loss)

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total_labels = 0
        correct = 0
        net.eval()
        for i, data in enumerate(valloader):
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                inputs, labels = data
                inputs = inputs.to(device)
                inputs = torch.reshape(inputs, (-1, 784))
                output = net(inputs)
                # _, predicted = torch.max(output, 1)
                predictions = torch.argmax(output.cpu(), dim=1)
                # correct += torch.sum(labels == predictions).item()
                correct += (predictions == labels).sum().item()
                total_labels += labels.size(0)

                loss = loss_function(output, inputs)
                val_loss += loss.cpu().numpy()
                valid_loss.append(val_loss)
                val_steps += 1

        metrics = {"loss": val_loss / val_steps, "accuracy": correct / total_labels}

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            rank = train.get_context().get_world_rank()
            torch.save(
                (net.state_dict(), optimizer.state_dict()),
                os.path.join(temp_checkpoint_dir, f"model-rank={rank}.pt"),
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        train.report(metrics, checkpoint=checkpoint)

    # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
    # checkpoint = None

    # should_checkpoint = epoch % config.get("checkpoint_freq", 1) == 0
    # In standard DDP training, where the model is the same across all ranks,
    # only the global rank 0 worker needs to save and report the checkpoint
    # if train.get_context().get_world_rank() == 0 and should_checkpoint:
    # torch.save(
    # (net.state_dict(), optimizer.state_dict()),
    # os.path.join(temp_checkpoint_dir, "model.pt"),
    # )
    # checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

    # train.report(metrics, checkpoint=checkpoint)

    print("Finished Training")


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
    # Best trial config: {'input_dim': 784, 'hidden_dim1': 64, 'hidden_dim2': 256, 'hidden_dim3': 16, 'hidden_dim4': 8, 'embedding_dim': 1, 'lr': 0.0006111085649326119, 'batch_size': 64}
    # Best trial final validation loss: 0.04955726166434111
    # Best trial final validation accuracy: 0.0
    # net = Autoencoder(784, 64, 256, 16, 8, 1)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])

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

    # this should work but how to get the path
    # checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "model.pt")
    # checkpoint = torch.load(checkpoint_path)

    # problem is with this
    checkpoint = train.get_checkpoint()
    print("Train checkpoint", checkpoint)
    if checkpoint:
        with checkpoint.as_directory() as loaded_checkpoint_dir:
            ck_dict = torch.load(os.path.join(loaded_checkpoint_dir, "model.pt"))
            start_epoch = int(ck_dict["epoch"]) + 1
            net.load_state_dict(ck_dict["model_state"])
            optimizer.load_state_dict(ck_dict["optimizer_state"])
    else:
        start_epoch = 0

    train_loss = []
    valid_loss = []
    train_steps = 0
    net.train()
    for epoch in range(
        start_epoch, config["epochs"]
    ):  # loop over the dataset multiple times
        running_loss = 0.0
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
            train_steps += 1

        train_loss.append(running_loss)

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        net.eval()
        for i, data in enumerate(valloader):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = torch.reshape(inputs, (-1, 784))
                output = net(inputs)

                loss = loss_function(output, inputs)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        metrics = {
            "t_loss": running_loss / train_steps,
            "v_loss": val_loss / val_steps,
            "epoch": epoch,
        }

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            # torch.save((net.state_dict(), optimizer.state_dict()),os.path.join(temp_checkpoint_dir, "model.pt"),)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": net.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "t_loss": running_loss / train_steps,
                    "v_loss": val_loss / val_steps,
                },
                os.path.join(temp_checkpoint_dir, "model.pt"),
            )
            print("Checkpoint Saved")
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(metrics, checkpoint=checkpoint)
            print("Checkpoint Reported")

    print("Finished Training")


def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )
    loss_function = nn.MSELoss()
    acc_loss = 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images = torch.reshape(images, (-1, 784))
            output = net(images)
            loss = loss_function(output, images)
            acc_loss += loss.cpu().numpy()
            tloss = loss.cpu().numpy()
            t1loss = loss.item()
    print("Best trial test set validation loss: {}".format(acc_loss))
    return acc_loss


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


config = {
    "input_dim": 784,
    "hidden_dim1": tune.choice([2**i for i in range(10)]),
    "hidden_dim2": tune.choice([2**i for i in range(10)]),
    "hidden_dim3": tune.choice([2**i for i in range(10)]),
    "hidden_dim4": tune.choice([2**i for i in range(10)]),
    "embedding_dim": tune.choice([2**i for i in range(5)]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16, 32, 64, 128]),
}
# train_and_validate_mnist(config, "./mnist_data/")

# test_accuracy(Autoencoder(784, 64, 128, 256, 256, 8), device="cpu")
