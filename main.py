import os
from functools import partial

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
from EvaluateTrainingModel import EvaluateTrainingModel
from ExportData import ExportData
from HyperParameterSearch import (
    get_voxel_tensor_datasets,
    train_and_validate_brain_voxels_ray,
)
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

    t_config.brain_area = brain.area
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

    brain.current_labels = brain.subject_labels_int
    stg_subject_binary_data = brain.binary_data(config)

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
        t_config.brain_area = brain.area
        export = ExportData()
        note = export.create_note(t_config)
        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=f"{brain.area}-Results",
            title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
        )
    brain.current_labels = brain.image_labels_int
    stg_image_binary_data = brain.binary_data(config)

    for bd in stg_image_binary_data:
        training = DataTraining()
        export_data = training.brain_data_classification(
            bd,
            t_config,
            strategies,
            classifiers,
        )
        t_config.brain_area = brain.area
        export = ExportData()
        note = export.create_note(t_config)
        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=f"{brain.area}-Results",
            title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
        )


def stg_concatenated_trails_classification(
    classifiers, strategies, t_config: TrainingConfig
):
    config = BrainDataConfig()
    brain = Brain(
        area=config.STG,
        data_path=config.STG_path,
        load_labels=True,
        load_int_labels=True,
    )

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    stg_subject_binary_data = brain.concatenate_fmri_data_trails(brain)

    t_config.analyze_concatenated_trails = True
    t_config.analyze_binary_trails = False

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
    t_config.brain_area = brain.area
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=all_data,
        sheet_name=f"{brain.area}-Results",
        title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
        single_label=True,
    )


def stg_binary_trails_classification(classifiers, strategies, t_config: TrainingConfig):
    config = BrainDataConfig()
    brain = Brain(
        area=config.STG,
        data_path=config.STG_path,
        load_labels=True,
        load_int_labels=True,
    )

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    stg_subject_binary_data = brain.binary_fmri_data_trails()

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

    t_config.brain_area = brain.area
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=all_data,
        sheet_name=f"{brain.area}-Results",
        title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
        single_label=True,
    )


def stg_concatenated_trails_classification(
    classifiers, strategies, t_config: TrainingConfig
):
    config = BrainDataConfig()
    brain = Brain(
        area=config.STG,
        data_path=config.STG_path,
        load_labels=True,
        load_int_labels=True,
    )

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    stg_subject_binary_data = brain.concatenate_fmri_data_trails()

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
    t_config.brain_area = brain.area
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=all_data,
        sheet_name=f"{brain.area}-Results",
        title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
        single_label=True,
    )


def ifg_concatenated_trails_classification(
    classifiers, strategies, t_config: TrainingConfig
):
    config = BrainDataConfig()
    brain = Brain(
        area=config.IFG,
        data_path=config.IFG_path,
        load_labels=True,
        load_int_labels=True,
    )

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    ifg_subject_binary_data = brain.concatenate_fmri_data_trails()

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
    t_config.brain_area = brain.area
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=all_data,
        sheet_name=f"{brain.area}-Results",
        title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
        single_label=True,
    )


def ifg_concatenated_binary_subjects_trails_classification(
    classifiers, strategies, t_config: TrainingConfig
):
    config = BrainDataConfig()
    brain = Brain(
        area=config.IFG,
        data_path=config.IFG_path,
        load_labels=True,
        load_int_labels=True,
    )

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    brains = brain.binary_subject_image_based_data()
    for m_brain in brains:
        ifg_subject_binary_data = m_brain.concatenate_fmri_data_trails()

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
        t_config.brain_area = brain.area
        export = ExportData()
        note = export.create_note(t_config)
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{brain.area}-Results",
            title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
            single_label=True,
        )


def stg_concatenated_binary_subjects_trails_classification(
    classifiers, strategies, t_config: TrainingConfig
):
    config = BrainDataConfig()
    brain = Brain(
        area=config.STG,
        data_path=config.STG_path,
        load_labels=True,
        load_int_labels=True,
    )

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    brains = brain.binary_subject_image_based_data()
    for m_brain in brains:
        ifg_subject_binary_data = m_brain.concatenate_fmri_data_trails()

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
        t_config.brain_area = brain.area
        export = ExportData()
        note = export.create_note(t_config)
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{brain.area}-Results",
            title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
            single_label=True,
        )


def ifg_binary_trails_classification(classifiers, strategies, t_config: TrainingConfig):
    config = BrainDataConfig()
    brain = Brain(
        area=config.IFG,
        data_path=config.IFG_path,
        load_labels=True,
        load_int_labels=True,
    )

    brain.current_labels = brain.subject_labels
    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    ifg_subject_binary_data = brain.binary_fmri_data_trails()

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

    t_config.brain_area = brain.area
    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=all_data,
        sheet_name=f"{brain.area}-Results",
        title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
        single_label=True,
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

    t_config.brain_area = brain.area
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
    area,
    num_samples=25,
    max_num_epochs=30,
    gpus_per_trial=1,
):
    import ray

    ray.init(local_mode=True)

    for i in range(5):

        voxel_sets = get_voxel_tensor_datasets(area)

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

        file_name = ExportData.get_file_name(".txt", f"BestTrainConfig-{area}")
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
    # brain_area = "IFG"
    brain_area = "STG"
    load_bestmodel_and_test(brain_area, model_path, device, gpus_per_trial=1)


def main():
    # analyse_nans()
    # visualize_nans()
    # classify_iris()
    # You can change the number of GPUs per trial here:
    # test_load_model()
    # train_valid_voxels()
    # train_valid_voxels_test()
    # train_valid_mnist(num_samples=2, max_num_epochs=1, gpus_per_trial=1)
    area = "STG"
    # area = "IFG"
    # hyper_parameter_search_braindata(area)

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
    stg_binary_classification(classifiers, strategies, t_config)
    # stg_concatenated_trails_classification(classifiers, strategies, t_config)
    # stg_binary_trails_classification(classifiers, strategies, t_config)

    t_config.dimension_reduction = True
    t_config.pca_fix_components = 10
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
        "brain_area": "STG",
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
        "brain_area": "IFG",
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
