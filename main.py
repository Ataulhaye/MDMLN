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

from Autoencoder import Autoencoder
from AutoenocderTrainingFMRI import load_bestmodel_and_test
from Brain import Brain
from BrainDataConfig import BrainDataConfig
from BrainDataLabel import BrainDataLabel
from DataTraining import DataTraining
from Enums import Lobe, Strategy
from EvaluateTrainingModel import EvaluateTrainingModel
from ExportData import ExportData
from FMRIAnalyser import FMRIAnalyser
from HyperParameterSearch import (
    get_voxel_tensor_datasets,
    train_and_validate_brain_voxels_ray,
)
from RSAConfig import RSAConfig
from TrainingConfig import TrainingConfig
from VideoToText import VideoToText
from Visualization import VisualizeData

# from TrainUtlis import load_data, test_accuracy, train_and_validate_mnist_ray_tune


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


def all_SLA():
    ##############################################################################
    # MCQ Section
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_Text_Embeddings()
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_Bridge_Embeddings()
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_Video_Embeddings()

    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_brain_difference_Text_Embeddings_RDM()
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_brain_difference_Bridge_Embeddings_RDM()
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_brain_difference_Video_Embeddings_RDM()

    ##############################################################################
    rsa_con = RSAConfig()
    rsa_con.strategy = Strategy.mice.name

    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.Searchlight_Text_Embeddings()
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.Searchlight_Bridge_Embeddings()
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.Searchlight_Video_Embeddings()

    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.Searchlight_brain_difference_Text_Embeddings_RDM()
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.Searchlight_brain_difference_Bridge_Embeddings_RDM()
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.Searchlight_brain_difference_Video_Embeddings_RDM()

    #######################################
    # Search Light abstract/Concrete section
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_Abstract_Concrete_RDM()
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_Related_Unrelated_RDM()
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_brain_difference_Related_Unrelated_RDM()
    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.Searchlight_brain_difference_Abstract_Concrete_RDM()

    rsa_con = RSAConfig()
    rsa_con.strategy = Strategy.mice.name
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.Searchlight_Abstract_Concrete_RDM()
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.Searchlight_Related_Unrelated_RDM()
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.Searchlight_brain_difference_Related_Unrelated_RDM()
    analyser = FMRIAnalyser(Lobe.ALL, rsa_config=rsa_con)
    analyser.Searchlight_brain_difference_Abstract_Concrete_RDM()
    ################################################


def main():
    # analyser = FMRIAnalyser(Lobe.IFG)
    # analyser.hyperparameter_tuning()
    analyser = FMRIAnalyser(Lobe.IFG)
    analyser.training_config.use_autoencoder = True
    analyser.subject_and_image_classification()

    ###########################################
    # For todays run
    analyser = FMRIAnalyser(Lobe.IFG)
    analyser.binary_subject_binary_image_classification()

    analyser = FMRIAnalyser(Lobe.STG)
    analyser.binary_subject_binary_image_classification()

    analyser = FMRIAnalyser(Lobe.IFG)
    analyser.unary_subject_binary_image_classification()

    analyser = FMRIAnalyser(Lobe.STG)
    analyser.unary_subject_binary_image_classification()
    ###########################################

    # Classic ML section

    #############################################
    # nans plotting

    analyser = FMRIAnalyser(Lobe.IFG)
    analyser.plot_NaNs()

    analyser = FMRIAnalyser(Lobe.STG)
    analyser.plot_NaNs()

    analyser = FMRIAnalyser(Lobe.ALL)
    analyser.plot_NaNs()

    #############################
    all_SLA()

    ##############################################################################
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

    t_config = TrainingConfig()
    t_config.dimension_reduction = True
    t_config.has_fix_components = True

    # t_config.predefined_split = False
    # stg_classification(classifiers, strategies, t_config)# was using to create the results for stg

    ###############################
    # = opt_autoencoder_config_STG
    t_config.optimal_autoencoder_config = {
        "input_dim": 7238,
        "hidden_dim1": 4096,
        "hidden_dim2": 512,
        "embedding_dim": 2,
        "lr": 0.00024641847887259374,
        "batch_size": 384,
        "epochs": 25,
        "lobe": "STG",
    }
    # t_config.optimal_autoencoder_config["epochs"] = 1
    # t_config.folds = 2
    # stg_classification(classifiers, strategies, t_config)
    #####################################
    # opt_autoencoder_config_IFG
    t_config.optimal_autoencoder_config = {
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
