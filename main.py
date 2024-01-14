import os
import pickle
from functools import partial
from math import ceil

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from torchvision import datasets

from AutoEncoder import Autoencoder
from Brain import Brain
from BrainDataConfig import BrainDataConfig
from BrainDataLabel import BrainDataLabel
from BrainTrainUtils import (
    generate_model,
    tes,
    train_and_validate_brain_voxels,
    train_and_validate_brain_voxels_ray,
)
from DataTraining import DataTraining
from EvaluateTrainingModel import EvaluateTrainingModel
from ExportData import ExportData
from PlotData import VisualizeData
from TrainingConfig import TrainingConfig
from TrainUtlis import load_data, test_accuracy, train_and_validate_mnist_ray_tune


def run_evaluation():
    X, y = datasets.load_iris(return_X_y=True)
    print("Original set shape:", X.shape, y.shape)

    # split the data set randomly into test and train sets
    # random_state=some number will always output the same sets by every execution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    # classifier
    clf = svm.SVC(kernel="linear", C=1, random_state=42)
    # cross validation score take cares of fiting the model
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print("----------------------------")
    print("Scores using sklearn cmd:", scores)

    print(
        "%0.2f accuracy with a standard deviation of %0.2f"
        % (scores.mean(), scores.std())
    )

    t_statistic, p_value = stats.ttest_1samp(a=scores, popmean=0.3)
    # p value less then 0.05 consider to be significant, greater then 0.05 is consider not to be significant
    print("t_statistic , p_value", t_statistic, p_value)

    clf = svm.SVC(kernel="linear", C=1).fit(X_train, y_train)
    clf.score(X_test, y_test)

    print("----------------------------")
    svm_clf = svm.SVC(kernel="linear", C=1)
    svm_scores = DataTraining.training_prediction_using_cross_validation(svm_clf, X, y)

    print("----------------------------")
    dtree_clf = DecisionTreeClassifier(random_state=0)
    dtree_scores = DataTraining.training_prediction_using_cross_validation(
        dtree_clf, X, y
    )

    print("----------------------------")
    knc = KNeighborsClassifier(n_neighbors=3)
    knc_scores = DataTraining.training_prediction_using_cross_validation(knc, X, y)

    print("----------------------------------------------------")
    EvaluateTrainingModel.evaluate_models(svm_scores, svm_clf, dtree_scores, dtree_clf)
    print("----------------------------------------------------")
    EvaluateTrainingModel.evaluate_models(svm_scores, svm_clf, knc_scores, knc)
    print("----------------------------------------------------")


def classify_iris():
    x, y = datasets.load_iris(return_X_y=True)
    iris_label = BrainDataLabel(name="IRIS", popmean=0.33, labels=y)
    result = DataTraining().train_and_test_model_accuracy(
        x=x,
        y=iris_label,
        classifier="LGBM",
        test_size=0.2,
        popmean=iris_label.popmean,
        folds=5,
        predefined_split=True,
        strategy="m",
    )
    print(result)


def analyse_nans():
    config = BrainDataConfig()
    stg = Brain(area=config.STG, data_path=config.STG_path, load_labels=True)
    nans_column_wise = stg.calculate_nans_voxel_wise(stg.voxels)
    print("stg nans_column_wise", len(nans_column_wise))
    nans_voxel_wise = stg.calculate_nans_trail_wise(stg.voxels)
    print("stg nans_voxel_wise", len(nans_voxel_wise))
    print("------------")

    ifg = Brain(area=config.IFG, data_path=config.IFG_path, load_labels=True)
    nans_column_wise_ifg = ifg.calculate_nans_voxel_wise(ifg.voxels)
    print("IFG nans_column_wise", len(nans_column_wise_ifg))
    nans_voxel_wise_ifg = ifg.calculate_nans_trail_wise(ifg.voxels)
    print("IFG nans_voxel_wise", len(nans_voxel_wise_ifg))
    print("------------")


def visualize_nans():
    config = BrainDataConfig()
    stg = Brain(area=config.STG, data_path=config.STG_path, load_labels=True)
    ifg = Brain(area=config.IFG, data_path=config.IFG_path, load_labels=True)
    data_list = [stg, ifg]
    for data in data_list:
        nans_column_wise = stg.calculate_nans_voxel_wise(data.voxels)
        columns = [i for i in range(data.voxels.shape[1])]
        VisualizeData.plot_bar_graph(
            ("Columns", columns),
            ("nans-length-column-wise", nans_column_wise),
            title=data.area,
        )

        nans_voxel_wise = stg.calculate_nans_trail_wise(data.voxels)
        rows = [i for i in range(data.voxels.shape[0])]
        VisualizeData.plot_bar_graph(
            ("nans-length-voxel-wise", nans_voxel_wise),
            ("rows", rows),
            bar_color="red",
            title=data.area,
        )

    # VisualizeData.plot_data_bar(np.array(x), np.array(nans_column_wise))


def ifg_classification(classifiers, strategies, t_config: TrainingConfig):
    config = BrainDataConfig()
    brain = Brain(
        area=config.IFG,
        data_path=config.IFG_path,
        load_labels=True,
        load_int_labels=True,
    )

    training = DataTraining()

    brain.current_labels = brain.subject_labels_int
    export_data = training.brain_data_classification(
        brain, t_config, strategies, classifiers
    )

    brain.current_labels = brain.image_labels_int
    e_data = training.brain_data_classification(
        brain, t_config, strategies, classifiers
    )
    export_data.extend(e_data)

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


def stg_binary_classification(classifiers, strategies, t_config: TrainingConfig):
    config = BrainDataConfig()
    brain = Brain(
        area=config.STG,
        data_path=config.STG_path,
        load_labels=True,
        load_int_labels=True,
    )

    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    stg_subject_binary_data = brain.binary_data(config, brain.subject_labels_int)

    t_config.dimension_reduction = True
    t_config.explain = True
    t_config.folds = 1
    t_config.predefined_split = False

    for bd in stg_subject_binary_data:
        training = DataTraining()
        export_data = training.brain_data_classification(
            bd,
            t_config,
            strategies,
            classifiers,
        )
        export = ExportData()
        note = export.create_note(t_config)
        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=f"{brain.area}-Results",
            title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
        )

    stg_image_binary_data = brain.binary_data(config, brain.image_labels_int)

    for bd in stg_image_binary_data:
        training = DataTraining()
        export_data = training.brain_data_classification(
            bd,
            t_config,
            strategies,
            classifiers,
        )
        export = ExportData()
        note = export.create_note(t_config)
        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=f"{brain.area}-Results",
            title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
        )


def stg_classification(classifiers, strategies, t_config: TrainingConfig):
    config = BrainDataConfig()
    brain = Brain(
        area=config.STG,
        data_path=config.STG_path,
        load_labels=True,
        load_int_labels=True,
    )

    brain.current_labels = brain.subject_labels_int

    training = DataTraining()

    export_data = training.brain_data_classification(
        brain, t_config, strategies, classifiers
    )

    brain.current_labels = brain.image_labels_int
    e_data = training.brain_data_classification(
        brain, t_config, strategies, classifiers
    )
    export_data.extend(e_data)

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


def train_valid_mnist(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    import ray

    ray.init(local_mode=True)
    # data_dir = os.path.abspath("./data")
    data_dir = os.path.abspath("./mnist_data/")
    load_data(data_dir)

    config = {
        "input_dim": 784,
        "hidden_dim1": tune.choice([2**i for i in range(10)]),
        "hidden_dim2": tune.choice([2**i for i in range(10)]),
        "hidden_dim3": tune.choice([2**i for i in range(10)]),
        "hidden_dim4": tune.choice([2**i for i in range(10)]),
        "embedding_dim": tune.choice([2**i for i in range(5)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16, 32, 64, 128]),
        "epochs": 10,
    }

    # config = {
    # "input_dim": 784,
    # "hidden_dim1": tune.choice([2**i for i in range(2)]),
    # "hidden_dim2": tune.choice([2**i for i in range(2)]),
    # "hidden_dim3": tune.choice([2**i for i in range(2)]),
    # "hidden_dim4": tune.choice([2**i for i in range(2)]),
    # "embedding_dim": tune.choice([2**i for i in range(2)]),
    # "lr": tune.loguniform(1e-4, 1e-1),
    # "batch_size": tune.choice([64, 128]),
    # "epochs": 10,
    # }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                partial(train_and_validate_mnist_ray_tune, data_dir=data_dir)
            ),
            # tune.with_parameters(train_and_validate_mnist_ray_tune),
            resources={"cpu": 6, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            metric="v_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    best_result = results.get_best_result("v_loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["v_loss"]))
    print("Best trial final training loss: {}".format(best_result.metrics["t_loss"]))
    print("Best trial epoch: {}".format(best_result.metrics["epoch"]))
    print("Best model path", best_result.path)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "model.pt")

    checkpoint = torch.load(checkpoint_path)

    # Best trial config: {'input_dim': 784, 'hidden_dim1': 128, 'hidden_dim2': 4, 'hidden_dim3': 16, 'hidden_dim4': 32, 'embedding_dim': 4, 'lr': 0.002424992195342828, 'batch_size': 64, 'epochs': 10}
    best_trained_model = Autoencoder(
        best_result.config["input_dim"],
        best_result.config["hidden_dim1"],
        best_result.config["hidden_dim2"],
        best_result.config["hidden_dim3"],
        best_result.config["hidden_dim4"],
        best_result.config["embedding_dim"],
    )

    best_trained_model.load_state_dict(checkpoint["model_state"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


def train_valid_voxels(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    import ray

    ray.init(local_mode=True)
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

    modify_tt_set.y_train = np.array([])
    modify_tt_set.y_test = np.array([])

    config = {
        "input_dim": modify_tt_set.X_train.shape[1],
        "hidden_dim1": tune.choice([2**i for i in range(13)]),
        "hidden_dim2": tune.choice([2**i for i in range(13)]),
        "hidden_dim3": tune.choice([2**i for i in range(13)]),
        "hidden_dim4": tune.choice([2**i for i in range(13)]),
        "embedding_dim": tune.choice([2**i for i in range(5)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16, 32, 64, 128]),
        "epochs": 10,
    }
    # scheduler = ASHAScheduler(max_t=max_num_epochs,grace_period=1,reduction_factor=2,)
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                partial(train_and_validate_brain_voxels_ray, set=modify_tt_set)
            ),
            # tune.with_parameters(train_and_validate_mnist_ray_tune),
            resources={"cpu": 6, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            metric="v_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    # ray.put()
    results = tuner.fit()
    best_result = results.get_best_result("v_loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["v_loss"]))
    print("Best trial final training loss: {}".format(best_result.metrics["t_loss"]))
    print("Best trial epoch: {}".format(best_result.metrics["epoch"]))
    print("Best model path", best_result.path)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "model.pt")

    checkpoint = torch.load(checkpoint_path)

    best_trained_model = generate_model(best_result.config)

    best_trained_model.load_state_dict(checkpoint["model_state"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


def train_valid_voxels_test():
    modify_tt_set = tes()

    config = {
        "input_dim": modify_tt_set.X_train.tensors[0].shape[1],
        "hidden_dim1": 1250,
        "hidden_dim2": 750,
        "hidden_dim3": 500,
        "hidden_dim4": 250,
        "embedding_dim": 8,
        "lr": 0.0009231555955597683,
        "batch_size": 8,
        "epochs": 10,
    }

    train_and_validate_brain_voxels(config, modify_tt_set)


def main():
    # analyse_nans()
    # visualize_nans()
    # classify_iris()
    # You can change the number of GPUs per trial here:
    # train_valid_voxels()
    train_valid_voxels_test()
    # train_valid_mnist(num_samples=2, max_num_epochs=1, gpus_per_trial=1)
    strategies = [
        None,
        "mean",
        "median",
        "most_frequent",
        "constant",
        "remove-voxels",
        "n_neighbors",
    ]
    classifiers = [
        # "XGBoost" # not working, Invalid classes inferred from unique values of `y`.  Expected: [0 1 2], got ['D' 'N' 'S']
        "CatBoost",
        "LGBM",
        "DecisionTree",
        # "HistGradientBoosting",
        "SVM",
        "KNearestNeighbors",
        "GaussianNaiveBayes",
        "LinearDiscriminant",
        "MLP",
        "LogisticRegression",
        "RandomForest",
    ]
    strategies = ["mean", "remove-voxels", "median"]
    classifiers = ["SVM", "MLP", "LinearDiscriminant"]
    t_config = TrainingConfig()
    # t_config.folds = 1
    # t_config.explain = True
    t_config.dimension_reduction = True
    t_config.predefined_split = True

    # stg_binary_classification(classifiers, strategies, t_config)
    # stg_classification(classifiers, strategies, t_config)
    # ifg_classification(classifiers, strategies, t_config)

    t_config.predefined_split = False

    # stg_classification(classifiers, strategies, t_config)
    # ifg_classification(classifiers, strategies, t_config)


if __name__ == "__main__":
    # Set-ExecutionPolicy Unrestricted -Scope Process
    # ./activate.ps1
    # Best trial config: {'input_dim': 784, 'hidden_dim1': 64, 'hidden_dim2': 128, 'hidden_dim3': 256, 'hidden_dim4': 256, 'embedding_dim': 8, 'lr': 0.0009231555955597683, 'batch_size': 2}
    # Best trial final validation loss: 0.02403166469299079
    # Best trial final validation accuracy: 0.0

    main()
