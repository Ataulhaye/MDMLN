import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from AutoEncoderN import AutoencoderN
from ExportData import ExportData
from TestTrainingSet import TestTrainingTensorDataset


def test_autoencoder_braindataN(net, testset: TensorDataset, device="cpu"):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    loss_function = nn.MSELoss()
    acc_loss = 0.0
    loss_list = []
    net.eval()
    with torch.no_grad():
        for data in testloader:
            inputs = data[0].to(device)
            _, output = net(inputs)
            loss = loss_function(output, inputs)
            acc_loss += loss.item()
            loss_list.append(loss.item())

    test_encodings_tensor, _ = net(testset.tensors[0].to(device))
    test_encodings = test_encodings_tensor.cpu().detach().numpy()
    test_labels = testset.tensors[1].numpy()

    print("testset:", testset.tensors[0].shape)
    print("Test set loss: {}".format(acc_loss / len(testset)))
    return acc_loss, test_encodings, test_labels


def generate_modelN(config):
    model = AutoencoderN(
        config["input_dim"],
        config["hidden_dim1"],
        config["hidden_dim2"],
        config["embedding_dim"],
    )
    return model


def train_autoencoder_braindataN(config, tensor_set: TestTrainingTensorDataset):
    # model = generate_model(config)
    model = generate_modelN(config)
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

    train_losses = []
    model.train()
    # training loop
    for epoch in range(config["epochs"]):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            voxels, _ = data
            voxels = voxels.to(device)

            optimizer.zero_grad()

            _, output = model(voxels)
            # following code is to get the latents (Encoded data)
            # encodings = model.encoder(voxels)
            # for encoding in encodings:
            # train_encodings.append(encoding.cpu().tolist())
            # for label in labels:
            # train_labels.append(label.item())

            loss = loss_function(output, voxels)
            print("Loss", loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_losses.append(running_loss / len(tensor_set.train_set))

    model_config = {
        "lr": config["lr"],
        "input_dim": config["input_dim"],
        "hidden_dim1": config["hidden_dim1"],
        "hidden_dim2": config["hidden_dim2"],
        "embedding_dim": config["embedding_dim"],
        "batch_size": config["batch_size"],
    }
    name = ExportData.get_file_name("_model.pt", config["brain_area"])
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": running_loss / len(tensor_set.train_set),
            "train_loss_list": train_losses,
            "config": model_config,
            "train_set": tensor_set.train_set.tensors[0].shape,
        },
        name,
    )
    print("Model Saved")

    # encoding
    model.eval()
    train_encodings_tensor, _ = model(tensor_set.train_set.tensors[0].to(device))
    train_encodings = train_encodings_tensor.cpu().detach().numpy()

    train_labels = tensor_set.train_set.tensors[1].numpy()
    print("Finished Training")
    print("Training loss", train_losses)
    # plt.plot(train_loss)
    # plt.savefig("test", dpi=700)
    # plt.close()
    return (model, train_encodings, train_labels)


def load_bestmodel_and_test(model_path, device, gpus_per_trial):
    checkpoint_path = model_path + "/model.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    best_trained_model = generate_modelN(checkpoint["config"])

    if device == "cuda:0" and gpus_per_trial > 1:
        best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_trained_model.load_state_dict(checkpoint["model_state"])

    file_name = ""
    static_dataset = None
    with open(file_name, "rb") as data:
        static_dataset = pickle.load(data)
    test_acc = test_autoencoder_braindataN(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))
