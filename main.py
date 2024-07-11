import csv
import glob
import os
from functools import partial
from pathlib import Path

import moviepy.editor as mp
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from ray import tune
from ray.train import CheckpointConfig, RunConfig
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
from BrainTrainUtils import load_bestmodel_and_test
from DataTraining import DataTraining
from Enums import Lobe, Strategy
from EvaluateTrainingModel import EvaluateTrainingModel
from ExportData import ExportData
from FMRIAnalyser import FMRIAnalyser
from HyperParameterSearch import (
    get_voxel_tensor_datasets,
    train_and_validate_brain_voxels_ray,
)
from PlotData import VisualizeData
from RSAConfig import RSAConfig
from TrainingConfig import TrainingConfig
from TrainUtlis import load_data, test_accuracy, train_and_validate_mnist_ray_tune
from VideoToText import VideoToText


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
    stg = Brain(lobe=Lobe.STG, data_path=config.STG_path)
    nans_column_wise = stg.calculate_nans_voxel_wise(stg.voxels)
    print("stg nans_column_wise", len(nans_column_wise))
    nans_voxel_wise = stg.calculate_nans_trail_wise(stg.voxels)
    print("stg nans_voxel_wise", len(nans_voxel_wise))
    print("------------")

    ifg = Brain(lobe=Lobe.IFG, data_path=config.IFG_path)
    nans_column_wise_ifg = ifg.calculate_nans_voxel_wise(ifg.voxels)
    print("IFG nans_column_wise", len(nans_column_wise_ifg))
    nans_voxel_wise_ifg = ifg.calculate_nans_trail_wise(ifg.voxels)
    print("IFG nans_voxel_wise", len(nans_voxel_wise_ifg))
    print("------------")


def visualize_nans():
    config = BrainDataConfig()
    stg = Brain(lobe=Lobe.STG, data_path=config.STG_path)
    ifg = Brain(lobe=Lobe.IFG, data_path=config.IFG_path)
    data_list = [stg, ifg]
    for data in data_list:
        nans_column_wise = stg.calculate_nans_voxel_wise(data.voxels)
        print("---------------------------------------------------------")
        print(
            f"Indexes of {data.lobe.name} where whole column is NAN: ",
            nans_column_wise.count(488),
        )
        total_nans = sum(nans_column_wise)
        print("Total NANs: ", total_nans)
        print(
            "Total NANs in all data: {:0.2f}%".format(
                ((total_nans / (data.voxels.shape[0] * data.voxels.shape[1])) * 100)
            )
        )
        print("-------------------------------------------------------------")
        columns = [i for i in range(data.voxels.shape[1])]
        VisualizeData.plot_bar_graph(
            ("Columns", columns),
            ("nans-length-column-wise", nans_column_wise),
            title=data.lobe.name,
        )

        nans_voxel_wise = stg.calculate_nans_trail_wise(data.voxels)
        rows = [i for i in range(data.voxels.shape[0])]
        VisualizeData.plot_bar_graph(
            ("nans-length-voxel-wise", nans_voxel_wise),
            ("rows", rows),
            bar_color="red",
            title=data.lobe.name,
        )

    # VisualizeData.plot_data_bar(np.array(x), np.array(nans_column_wise))


def ifg_classification(classifiers, strategies, t_config: TrainingConfig):
    config = BrainDataConfig()
    brain = Brain(
        lobe=Lobe.IFG,
        data_path=config.IFG_path,
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

    t_config.lobe = brain.lobe.name
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=export_data,
        sheet_name=f"{brain.lobe.name}-Results",
        title=f"{brain.lobe.name}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
    )


def stg_binary_classification_with_shap(
    classifiers, strategies, t_config: TrainingConfig
):
    config = BrainDataConfig()
    brain = Brain(lobe=Lobe.STG, data_path=config.STG_path)

    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    brain.current_labels = brain.subject_labels_int
    stg_subject_binary_data = brain.binarize_fmri_image_or_subject(config)

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
        t_config.lobe = brain.lobe.name
        export = ExportData()
        note = export.create_note(t_config)
        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=f"{brain.lobe.name}-Results",
            title=f"{brain.lobe.name}-{t_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
        )
    brain.current_labels = brain.image_labels_int
    stg_image_binary_data = brain.binarize_fmri_image_or_subject(config)

    for bd in stg_image_binary_data:
        training = DataTraining()
        export_data = training.brain_data_classification(
            bd,
            t_config,
            strategies,
            classifiers,
        )
        t_config.lobe = brain.lobe.name
        export = ExportData()
        note = export.create_note(t_config)
        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=f"{brain.lobe.name}-Results",
            title=f"{brain.lobe.name}-{t_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
        )


def stg_subject_binary_classification(
    classifiers, strategies, t_config: TrainingConfig
):
    config = BrainDataConfig()
    brain = Brain(lobe=Lobe.STG, data_path=config.STG_path)

    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    brain.current_labels = brain.subject_labels
    stg_subject_binary_data = brain.binarize_fmri_image_or_subject(config)

    t_config.dimension_reduction = True
    # t_config.explain = True
    # t_config.folds = 1
    # t_config.predefined_split = False
    all_export_data = []

    for bd in stg_subject_binary_data:
        training = DataTraining()
        export_data = training.brain_data_classification(
            bd,
            t_config,
            strategies,
            classifiers,
        )
        all_export_data.extend(export_data)
    t_config.lobe = brain.lobe.name
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=all_export_data,
        sheet_name=f"{brain.lobe.name}-Results",
        title=f"{brain.lobe.name}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
        single_label=True,
    )


def ifg_binary_subject_classification(
    classifiers, strategies, t_config: TrainingConfig
):
    config = BrainDataConfig()
    brain = Brain(lobe=Lobe.IFG, data_path=config.IFG_path)

    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    brain.current_labels = brain.subject_labels
    ifg_subject_binary_data = brain.binarize_fmri_image_or_subject(config)

    # t_config.dimension_reduction = True

    all_export_data = []

    for bd in ifg_subject_binary_data:
        training = DataTraining()
        export_data = training.brain_data_classification(
            bd,
            t_config,
            strategies,
            classifiers,
        )
        all_export_data.extend(export_data)
    t_config.lobe = brain.lobe.name
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=all_export_data,
        sheet_name=f"{brain.lobe.name}-Results",
        title=f"{brain.lobe.name}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
        single_label=True,
    )


def stg_binary_trails_classification(classifiers, strategies, t_config: TrainingConfig):
    config = BrainDataConfig()
    brain = Brain(lobe=Lobe.STG, data_path=config.STG_path)

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    stg_subject_binary_data = brain.binary_fmri_image_trails()

    t_config.analyze_binary_trails = True
    t_config.analyze_concatenated_trails = False

    all_data = []

    for bd in stg_subject_binary_data:
        training = DataTraining()
        export_data = training.brain_data_classification(
            bd,
            t_config,
            strategies,
            classifiers,
        )
        all_data.extend(export_data)

    t_config.lobe = brain.lobe.name
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=all_data,
        sheet_name=f"{brain.lobe.name}-Results",
        title=f"{brain.lobe.name}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
        single_label=True,
    )


def stg_concatenated_trails_classification(
    classifiers, strategies, t_config: TrainingConfig
):
    config = BrainDataConfig()
    brain = Brain(lobe=Lobe.STG, data_path=config.STG_path)

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    stg_subject_binary_data = brain.concatenate_fmri_image_trails()

    t_config.analyze_binary_trails = False
    t_config.analyze_concatenated_trails = True

    all_data = []

    for bd in stg_subject_binary_data:
        training = DataTraining()
        export_data = training.brain_data_classification(
            bd,
            t_config,
            strategies,
            classifiers,
        )
        all_data.extend(export_data)
    t_config.lobe = brain.lobe.name
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=all_data,
        sheet_name=f"{brain.lobe.name}-Results",
        title=f"{brain.lobe.name}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
        single_label=True,
    )


def ifg_concatenated_trails_classification(
    classifiers, strategies, t_config: TrainingConfig
):
    config = BrainDataConfig()
    brain = Brain(lobe=Lobe.IFG, data_path=config.IFG_path)

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    ifg_subject_binary_data = brain.concatenate_fmri_image_trails()

    t_config.analyze_binary_trails = False
    t_config.analyze_concatenated_trails = True

    all_data = []

    for bd in ifg_subject_binary_data:
        training = DataTraining()
        export_data = training.brain_data_classification(
            bd,
            t_config,
            strategies,
            classifiers,
        )
        all_data.extend(export_data)
    t_config.lobe = brain.lobe.name
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=all_data,
        sheet_name=f"{brain.lobe.name}-Results",
        title=f"{brain.lobe.name}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
        single_label=True,
    )


def ifg_concatenated_binary_subjects_trails_classification(
    classifiers, strategies, t_config: TrainingConfig
):
    config = BrainDataConfig()
    brain = Brain(lobe=Lobe.IFG, data_path=config.IFG_path)

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    brains = brain.binary_subject_image_based_data()
    for m_brain in brains:
        ifg_subject_binary_data = m_brain.concatenate_fmri_image_trails()

        t_config.analyze_binary_trails = False
        t_config.analyze_concatenated_trails = True

        all_data = []

        for bd in ifg_subject_binary_data:
            training = DataTraining()
            export_data = training.brain_data_classification(
                bd,
                t_config,
                strategies,
                classifiers,
            )
            all_data.extend(export_data)
        t_config.lobe = brain.lobe.name
        export = ExportData()
        note = export.create_note(t_config)
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{brain.lobe.name}-Results",
            title=f"{brain.lobe.name}-{t_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
            single_label=True,
        )


def stg_concatenated_binary_subjects_trails_classification(
    classifiers, strategies, t_config: TrainingConfig
):
    config = BrainDataConfig()
    brain = Brain(lobe=Lobe.STG, data_path=config.STG_path)

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    brains = brain.binary_subject_image_based_data()
    for m_brain in brains:
        ifg_subject_binary_data = m_brain.concatenate_fmri_image_trails()

        t_config.analyze_binary_trails = False
        t_config.analyze_concatenated_trails = True

        all_data = []

        for bd in ifg_subject_binary_data:
            training = DataTraining()
            export_data = training.brain_data_classification(
                bd,
                t_config,
                strategies,
                classifiers,
            )
            all_data.extend(export_data)
        t_config.lobe = brain.lobe.name
        export = ExportData()
        note = export.create_note(t_config)
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{brain.lobe.name}-Results",
            title=f"{brain.lobe.name}-{t_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
            single_label=True,
        )


def ifg_binary_trails_classification(classifiers, strategies, t_config: TrainingConfig):
    config = BrainDataConfig()
    brain = Brain(lobe=Lobe.IFG, data_path=config.IFG_path)

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    ifg_subject_binary_data = brain.binary_fmri_image_trails()

    t_config.analyze_binary_trails = True
    t_config.analyze_concatenated_trails = False

    all_data = []

    for bd in ifg_subject_binary_data:
        training = DataTraining()
        export_data = training.brain_data_classification(
            bd,
            t_config,
            strategies,
            classifiers,
        )
        all_data.extend(export_data)

    t_config.lobe = brain.lobe.name
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=all_data,
        sheet_name=f"{brain.lobe.name}-Results",
        title=f"{brain.lobe.name}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
        single_label=True,
    )


def stg_classification(classifiers, strategies, t_config: TrainingConfig):
    config = BrainDataConfig()
    brain = Brain(lobe=Lobe.STG, data_path=config.STG_path)

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

    t_config.lobe = brain.lobe.name
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=export_data,
        sheet_name=f"{brain.lobe.name}-Results",
        title=f"{brain.lobe.name}-{t_config.folds}-Folds-{split}-Clf",
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
        "embedding_dim": tune.choice([2**i for i in range(5)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16, 32, 64, 128]),
        "epochs": 10,
    }
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


def hyper_parameter_search_braindata(
    lobe,
    num_samples=25,
    max_num_epochs=30,
    gpus_per_trial=1,
):
    import ray

    ray.init(local_mode=True)

    for i in range(5):

        voxel_sets = get_voxel_tensor_datasets(lobe)

        config = {
            "input_dim": voxel_sets.train_set.tensors[0].shape[1],
            "hidden_dim1": tune.choice([2**i for i in range(13)]),
            "hidden_dim2": tune.choice([2**i for i in range(13)]),
            "embedding_dim": tune.choice([2**i for i in range(5)]),
            "lr": tune.loguniform(1e-4, 1e-1),
            # "batch_size": tune.choice([128, 256, 512]),
            # "batch_size": 384,
            "batch_size": voxel_sets.train_set.tensors[0].shape[0],
            "epochs": 30,
        }
        # scheduler = ASHAScheduler(max_t=max_num_epochs,grace_period=1,reduction_factor=2,)
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2,
        )
        # run_config = RunConfig(
        # checkpoint_config=CheckpointConfig(
        # num_to_keep=5,
        # *Best* checkpoints are determined by these params:
        # checkpoint_score_attribute="train_loss",
        # checkpoint_score_order="min",
        # ),
        # This will store checkpoints on S3.
        # storage_path="s3://remote-bucket/location",
        # )
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    partial(train_and_validate_brain_voxels_ray, tensor_set=voxel_sets)
                ),
                # tune.with_parameters(train_and_validate_mnist_ray_tune),
                resources={"cpu": 10, "gpu": gpus_per_trial},
            ),
            tune_config=tune.TuneConfig(
                metric="train_loss",
                mode="min",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            param_space=config,
            # run_config=run_config,
        )

        # ray.put()
        results = tuner.fit()
        best_result = results.get_best_result("train_loss", "min")

        print("Best trial config: {}".format(best_result.config))
        print(
            "Best trial final training loss: {}".format(
                best_result.metrics["train_loss"]
            )
        )
        print("Best trial epoch: {}".format(best_result.metrics["epoch"]))
        print("Best model path", best_result.path)

        file_name = ExportData.get_file_name(".txt", f"BestTrainConfig-{lobe}")
        file_path = f"C://Users//ataul//source//Uni//BachelorThesis//poc//{file_name}"
        with open(file_path, "w") as text_file:
            text_file.write("----------------------------------------\n")
            text_file.write("Best trial config: {}\n".format(best_result.config))
            text_file.write(
                "Best trial final training loss: {}\n".format(
                    best_result.metrics["train_loss"]
                )
            )
            text_file.write(
                "Best trial epoch: {}\n".format(best_result.metrics["epoch"])
            )
            text_file.write("Best model path {}\n".format(best_result.path))
            text_file.write("----------------------------------------\n")

    # device = "cpu"
    # if torch.cuda.is_available():
    # device = "cuda:0"

    # checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "model.pt")

    # checkpoint = torch.load(checkpoint_path, map_location=device)

    # best_trained_model = generate_model(best_result.config)

    # if device == "cuda:0" and gpus_per_trial > 1:
    # best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # best_trained_model.load_state_dict(checkpoint["model_state"])

    # test_acc = test_autoencode_braindata(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))


def test_load_model():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    path = r"C:\Users\ataul\ray_results\Best-IFG\tune_with_parameters_2024-03-16_17-40-53\tune_with_parameters_f4bd1_00011_11_embedding_dim=2,hidden_dim1=1024,hidden_dim2=8,lr=0.0016_2024-03-16_18-09-11\checkpoint_000024"
    # path = r"C:\Users\ataul\source\Uni\BachelorThesis\poc\STG_BestModel_With_TSNE\STG_mean_18-03-2024_09-57-37_209122_model.pt"
    # path = r"C:\Users\ataul\source\Uni\BachelorThesis\poc\STG_BestModel_With_TSNE\STG_mean_18-03-2024_14-02-01_780484_model.pt"
    model_path = path.replace(os.sep, "/")
    # lobe = "IFG"
    lobe = "STG"
    load_bestmodel_and_test(lobe, model_path, device, gpus_per_trial=1)


def analyze_STG():
    analyser = FMRIAnalyser(Lobe.STG)
    # analyser.training_config.dimension_reduction = True
    # analyser.training_config.has_fix_components = (True, 10)
    analyser.binary_subject_classification()
    analyser = FMRIAnalyser(Lobe.STG)
    # analyser.training_config.dimension_reduction = True
    analyser.binary_image_classification()
    analyser = FMRIAnalyser(Lobe.STG)
    # analyser.training_config.dimension_reduction = True
    analyser.unary_subject_binary_image_classification()
    analyser = FMRIAnalyser(Lobe.STG)
    # analyser.training_config.dimension_reduction = True
    analyser.binary_subject_concatenated_image_classification()
    analyser = FMRIAnalyser(Lobe.STG)
    # analyser.training_config.dimension_reduction = True
    analyser.binary_subject_binary_image_classification()
    analyser = FMRIAnalyser(Lobe.STG)
    # analyser.training_config.dimension_reduction = True
    analyser.binary_subject_unary_image_classification()
    analyser = FMRIAnalyser(Lobe.STG)
    # analyser.training_config.dimension_reduction = True
    analyser.subject_concatenated_image_classification()
    analyser = FMRIAnalyser(Lobe.STG)
    # analyser.training_config.dimension_reduction = True
    analyser.subject_binary_image_classification()


def analyze_IFG():
    analyser = FMRIAnalyser(Lobe.IFG)
    # analyser.training_config.dimension_reduction = True
    # analyser.training_config.has_fix_components = (True, 10)
    analyser.binary_subject_classification()
    analyser = FMRIAnalyser(Lobe.IFG)
    # analyser.training_config.dimension_reduction = True
    analyser.binary_image_classification()
    analyser = FMRIAnalyser(Lobe.IFG)
    # analyser.training_config.dimension_reduction = True
    analyser.unary_subject_binary_image_classification()
    analyser = FMRIAnalyser(Lobe.IFG)
    # analyser.training_config.dimension_reduction = True
    analyser.binary_subject_concatenated_image_classification()
    analyser = FMRIAnalyser(Lobe.IFG)
    # analyser.training_config.dimension_reduction = True
    analyser.binary_subject_binary_image_classification()
    analyser = FMRIAnalyser(Lobe.IFG)
    # analyser.training_config.dimension_reduction = True
    analyser.binary_subject_unary_image_classification()
    analyser = FMRIAnalyser(Lobe.IFG)
    # analyser.training_config.dimension_reduction = True
    analyser.subject_concatenated_image_classification()
    analyser = FMRIAnalyser(Lobe.IFG)
    # analyser.training_config.dimension_reduction = True
    analyser.subject_binary_image_classification()


def generate_text_file_from_videos():
    vt = VideoToText()
    vt.video_to_audio_transcript()


# "C:\Users\ataul\source\Uni\BachelorThesis\poc\AudioTranscriptsSet1_EN.csv"
def get_set_files(set_path, set_name):
    set_keys = []
    with open(set_path) as f:
        for line in f:
            k = line.split(" ")[0]
            assert k.split(".")[1].replace('"', "") in "mpg"
            set_keys.append(k)

    all_scripts = r"C:\Users\ataul\source\Uni\BachelorThesis\poc\Result_Files\AudioTranscripts_corrected.csv"

    set_transcripts = []
    scripts_d = []
    with open(all_scripts) as f:
        scripts_d = f.readlines()

    for key in set_keys:
        for script in scripts_d:
            if key in script:
                set_transcripts.append(script)
                break

    assert len(set_transcripts) == len(set_keys)
    dest_file = f"AudioTranscriptsSet{set_name}.csv"

    f = open(dest_file, "w")
    for trans in set_transcripts:
        f.write(trans)
    f.close()


def copy_en_txt(source_en_file, source_de_file, set_name):
    de_txt = None
    with open(source_de_file) as f:
        de_txt = f.readlines()

    en_txt = None
    with open(source_en_file) as f:
        en_txt = f.readlines()

    assert len(en_txt) == len(de_txt)

    set_transcripts = []

    for txt in de_txt:
        de_en = get_translation(txt, en_txt)
        if de_en is not None:
            set_transcripts.append(de_en)

    assert len(set_transcripts) == len(de_txt)

    dest_file = f"AudioTranscriptsSet{set_name}_DE_EN.csv"

    f = open(dest_file, "w")
    for trans in set_transcripts:
        f.write(trans)
    f.close()


def get_translation(txt, en_txt):
    de_en = None
    for script in en_txt:
        s_script = script.strip("\n")
        txt_seg = txt.split(";")
        if txt_seg[0] in script:
            de_en = txt.strip("\n") + ";" + s_script.split(";")[1].strip(" ") + "\n"
            break
    if de_en is None:
        print("This must not happened")
    return de_en


def load_embeddings():
    path = r"C:\Users\ataul\source\Uni\BachelorThesis\poc\PickleFiles\MCQ_Embeddings_Set1.pickle"
    embeddings = pd.read_pickle(path)
    for embed in embeddings:
        key, text_embed, answer_embed, bridge_embed, vid_embed, en_de_txt = embed
        print(key)
    print("End")


def mpg_to_mp4():
    path = r"C:\Users\ataul\source\Uni\BachelorThesis\Stimuli"
    directory_path = Path("Stimuli_mp4").absolute().resolve()
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    for video_file_path in glob.glob(os.path.join(path, "*.mpg")):
        file_name = video_file_path.split("\\")[-1].split(".")[0]
        final_file_name = f"{file_name}.mp4"
        save_path = os.path.join(directory_path, final_file_name)
        clip = mp.VideoFileClip(video_file_path, fps_source="fps")
        clip.write_videofile(save_path)


def main():
    # load_embeddings()
    # mpg_to_mp4()
    # generate_text_file_from_videos()
    # copy_en_txt(r"C:\Users\ataul\source\Uni\BachelorThesis\poc\AudioTranscriptsSet1_EN.csv",r"C:\Users\ataul\source\Uni\BachelorThesis\poc\AudioTranscriptsSet1.csv","1")
    # get_set_files(r"C:\Users\ataul\source\Uni\BachelorThesis\Stimuli_logs\Verstehen_Set1.txt", "1")

    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_Text_Embeddings(set_name="Set1")
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_Bridge_Embeddings(set_name="Set1")
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_Video_Embeddings(set_name="Set1")

    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_brain_difference_Text_Embeddings_RDM(set_name="Set1")
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_brain_difference_Bridge_Embeddings_RDM(set_name="Set1")
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_brain_difference_Video_Embeddings_RDM(set_name="Set1")

    rsa_con = RSAConfig()
    rsa_con.strategy = Strategy.mice.name
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.RSA_Abstract_Concrete_RDM()
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.RSA_Related_Unrelated_RDM()
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.RSA_brain_difference_Related_Unrelated_RDM()
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.RSA_brain_difference_Abstract_Concrete_RDM()

    # analyser.RSA_brain_difference_Abstract_Concrete_RDM()

    analyser = FMRIAnalyser(Lobe.STG, rsa_config=rsa_con)
    analyser.RSA_Related_Unrelated_RDM()

    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.RSA_Related_Unrelated_RDM()

    analyser = FMRIAnalyser(Lobe.STG)
    analyser.RSA_Related_Unrelated_RDM()

    rsa_con = RSAConfig()
    rsa_con.normalize = True
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.RSA_Abstract_Concrete_RDM()

    analyser = FMRIAnalyser(Lobe.STG, rsa_config=rsa_con)
    analyser.RSA_Abstract_Concrete_RDM()

    analyser = FMRIAnalyser(Lobe.IFG)
    analyser.training_config.dimension_reduction = True
    analyser.training_config.folds = 2
    analyser.strategies.insert(0, "mice")

    analyser.strategies = ["mice", "mean"]
    analyser.classifiers = ["SVM", "MLP"]
    analyser.unary_subject_binary_image_classification()

    analyser = FMRIAnalyser(Lobe.STG)
    analyser.training_config.dimension_reduction = True
    # analyser.training_config.folds = 1
    analyser.strategies.insert(0, "mice")
    # analyser.strategies = ["mice", "mean"]
    # analyser.classifiers = ["SVM", "MLP"]
    analyser.unary_subject_binary_image_classification()
    # analyze_IFG()
    # analyze_STG()

    # ifg.ifg_binary_subject_classification()
    # analyse_nans()
    # visualize_nans()
    # classify_iris()
    # You can change the number of GPUs per trial here:
    # test_load_model()
    # train_valid_voxels()
    # train_valid_voxels_test()
    # train_valid_mnist(num_samples=2, max_num_epochs=1, gpus_per_trial=1)
    lobe = "STG"
    # lobe = "IFG"
    # hyper_parameter_search_braindata(lobe)

    strategies = [
        None,
        "mean",
        "median",
        "most_frequent",
        "constant",
        "remove_voxels",
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
    strategies = ["mean", "remove_voxels", "median"]
    # classifiers = ["SVM", "MLP", "LinearDiscriminant"]
    classifiers = ["SVM", "MLP", "LinearDiscriminant", "LGBM"]
    # classifiers = ["LinearDiscriminant"]
    # strategies = ["mean"]
    # classifiers = ["SVM"]
    t_config = TrainingConfig()
    # t_config.folds = 10
    # t_config.explain = True

    t_config.dimension_reduction = True
    # stg_concatenated_trails_classification(classifiers, strategies, t_config)
    # stg_binary_trails_classification(classifiers, strategies, t_config)

    # ifg_concatenated_trails_classification(classifiers, strategies, t_config)
    # ifg_binary_trails_classification(classifiers, strategies, t_config)

    # t_config.has_fix_components = True
    # t_config.use_autoencoder = True
    # t_config.tsne = True
    # stg_binary_classification_with_shap(classifiers, strategies, t_config)
    # stg_subject_binary_classification(classifiers, strategies, t_config)
    # ifg_binary_subject_classification(classifiers, strategies, t_config)
    # stg_concatenated_trails_classification(classifiers, strategies, t_config)
    # stg_binary_trails_classification(classifiers, strategies, t_config)

    t_config.dimension_reduction = True
    t_config.has_fix_components = True
    # ifg_concatenated_binary_subjects_trails_classification(classifiers, strategies, t_config)
    # stg_concatenated_binary_subjects_trails_classification(classifiers, strategies, t_config)
    # stg_concatenated_trails_classification(classifiers, strategies, t_config)
    # stg_binary_trails_classification(classifiers, strategies, t_config)
    # ifg_concatenated_trails_classification(classifiers, strategies, t_config)
    # ifg_binary_trails_classification(classifiers, strategies, t_config)
    # stg_classification(classifiers, strategies, t_config)
    # ifg_classification(classifiers, strategies, t_config)

    # t_config.predefined_split = False
    # stg_classification(classifiers, strategies, t_config)# was using to create the results for stg

    ###############################
    # = best_autoencoder_config_STG
    t_config.best_autoencoder_config = {
        "input_dim": 7238,
        "hidden_dim1": 4096,
        "hidden_dim2": 512,
        "embedding_dim": 2,
        "lr": 0.00024641847887259374,
        "batch_size": 384,
        "epochs": 25,
        "lobe": "STG",
    }
    # t_config.best_autoencoder_config["epochs"] = 1
    # t_config.folds = 2
    # stg_classification(classifiers, strategies, t_config)
    #####################################
    # best_autoencoder_config_IFG
    t_config.best_autoencoder_config = {
        "input_dim": 523,
        "hidden_dim1": 1024,
        "hidden_dim2": 8,
        "embedding_dim": 2,
        "lr": 0.0016028928095361706,
        "batch_size": 384,
        "epochs": 25,
        "lobe": "IFG",
    }

    # ifg_classification(classifiers, strategies, t_config)


# Matlab stuff
# How to make .nii file from the .mat file
# https://de.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
# nii_image = load("C:\Users\ataul\source\Uni\BachelorThesis\poc\All_Brain_Data_Raw\AAL_all_lobes_ROI.rex.mat")
# matrix = cell2mat(struct2cell(nii_image))
# nii_image1 = make_nii(matrix)
# save_nii(nii_image1, "m1.nii")
# how to make gz file
# gzip("C:\Users\ataul\source\Uni\BachelorThesis\poc\All_Brain_Data_Raw\m1.nii")

if __name__ == "__main__":
    # Set-ExecutionPolicy Unrestricted -Scope Process
    # Location:
    # C:\Users\ataul\source\Uni\BachelorThesis\poc>
    # venv\Scripts\activate
    # deactivate
    # ./activate.ps1 #this may not needed
    main()
