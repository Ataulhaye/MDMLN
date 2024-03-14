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

from AutoEncoderN import AutoencoderN
from Brain import Brain
from BrainDataConfig import BrainDataConfig
from BrainDataLabel import BrainDataLabel
from DataTraining import DataTraining
from TestTrainingSet import TestTrainingSet, TestTrainingTensorDataset
from TrainingConfig import TrainingConfig


def train_and_validate_brain_voxels_rayN(config, tensor_set: TestTrainingTensorDataset):
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

    train_loss_list = []
    model.train()
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        train_steps = 0
        for i, data in enumerate(trainloader):

            inputs = data[0].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            _, output = model(inputs)
            loss = loss_function(output, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_steps += 1

        # train_loss_list.append(running_loss / train_steps)
        train_loss_list.append(running_loss / len(tensor_set.train_set))

        metrics = {
            # "train_loss": running_loss / train_steps,
            "train_loss": running_loss / len(tensor_set.train_set),
            "loss_list": train_loss_list,
            "epoch": epoch,
        }
        model_config = {
            "lr": config["lr"],
            "input_dim": config["input_dim"],
            "hidden_dim1": config["hidden_dim1"],
            "hidden_dim2": config["hidden_dim2"],
            "embedding_dim": config["embedding_dim"],
            "batch_size": config["batch_size"],
        }
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    # "train_loss": running_loss / train_steps,
                    "train_loss": running_loss / len(tensor_set.train_set),
                    "t_loss_list": train_loss_list,
                    "config": model_config,
                    "train_set": tensor_set.train_set.tensors[0].shape,
                },
                os.path.join(temp_checkpoint_dir, "model.pt"),
            )
            print("Checkpoint Saved")
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(metrics, checkpoint=checkpoint)
            print("Checkpoint Reported")

    print("Finished Training")


def test_voxels_accuracyN(net, testset: TensorDataset, device="cpu"):
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
    print("Best trial test set validation loss: {}".format(acc_loss / len(testset)))
    return acc_loss, test_encodings, test_labels


def generate_modelN(config):
    model = AutoencoderN(
        config["input_dim"],
        config["hidden_dim1"],
        config["hidden_dim2"],
        config["embedding_dim"],
    )
    return model


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
    test_acc = test_voxels_accuracyN(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


def get_voxel_tensor_datasetsN():
    bd_config = BrainDataConfig()
    brain = Brain(
        # area=bd_config.STG,
        area=bd_config.IFG,
        # data_path=bd_config.STG_path,
        data_path="C://Users//ataul//source//Uni//BachelorThesis//poc//ROI_aal_wfupick_left44_45.rex.mat",
        # data_path="C://Users//ataul//source//Uni//BachelorThesis//poc//left_STG_MTG_AALlable_ROI.rex.mat",
        load_labels=True,
        load_int_labels=True,
    )
    brain.current_labels = brain.subject_labels_int

    train_config = TrainingConfig()
    train_config.strategy = "mean"

    tt_set = DataTraining().premeditate_random_train_test_split(brain, train_config)

    brain.normalize_data_safely(strategy=train_config.strategy, data_set=tt_set)

    # file_name = f"{brain.area}_{train_config.strategy}_static_wholeSet.pickle"
    # with open(file_name, "wb") as output:
    # pickle.dump(tt_set, output)

    # with open(file_name, "rb") as data:
    # static_dataset = pickle.load(data)

    # file_name = f"{brain.area}_{train_config.strategy}_static_subSet.pickle"
    # with open(file_name, "wb") as output:
    # pickle.dump(modify_tt_set, output)

    tr_set = TensorDataset(
        torch.Tensor(tt_set.X_train), torch.tensor(tt_set.y_train, dtype=torch.int)
    )
    ts_set = TensorDataset(
        torch.Tensor(tt_set.X_test), torch.tensor(tt_set.y_test, dtype=torch.int)
    )

    sets = TestTrainingTensorDataset(train_set=tr_set, test_set=ts_set)

    file_name = f"{brain.area}_{train_config.strategy}_static_wholeSet.pickle"
    with open(file_name, "wb") as output:
        pickle.dump(sets, output)

    return sets


def get_tensor_datasets(
    brain: Brain, train_config: TrainingConfig, tt_set: TestTrainingSet
):
    bd_config = BrainDataConfig()
    # train_config.strategy = "mean"

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

    modify_tt_set = DataTraining().premeditate_random_train_test_split(
        modify_brain, train_config, modify_bd_config
    )

    XT_train = torch.Tensor(modify_tt_set.X_train)
    XT_val = torch.Tensor(modify_tt_set.X_test)
    yT_train = torch.Tensor(modify_tt_set.y_train)
    yT_val = torch.Tensor(modify_tt_set.y_test)

    tr_set = TensorDataset(XT_train, yT_train)
    vl_set = TensorDataset(XT_val, yT_val)
    ts_set = TensorDataset(torch.Tensor(tt_set.X_test), torch.Tensor(tt_set.y_test))

    sets = TestTrainingTensorDataset(train_set=tr_set, val_set=vl_set, test_set=ts_set)

    # file_name = f"{brain.area}_{train_config.strategy}_static_wholeSet.pickle"
    # with open(file_name, "wb") as output:
    # pickle.dump(sets, output)

    return sets
