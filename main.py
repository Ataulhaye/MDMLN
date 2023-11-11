import random

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import datasets, metrics, preprocessing, svm
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import (
    ShuffleSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from FirstPhaseData import PrepareData, normalize_data
from init import (
    evaluate_models,
    k_fold_training_and_validation,
    train_and_test_model_accuracy,
)


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
    svm_scores = k_fold_training_and_validation(svm_clf, X, y)

    print("----------------------------")
    dtree_clf = DecisionTreeClassifier(random_state=0)
    dtree_scores = k_fold_training_and_validation(dtree_clf, X, y)

    print("----------------------------")
    knc = KNeighborsClassifier(n_neighbors=3)
    knc_scores = k_fold_training_and_validation(knc, X, y)

    print("----------------------------------------------------")
    evaluate_models(svm_scores, svm_clf, dtree_scores, dtree_clf)
    print("----------------------------------------------------")
    evaluate_models(svm_scores, svm_clf, knc_scores, knc)
    print("----------------------------------------------------")


def classify_STG_on_image_labels():
    data = PrepareData()
    STG = normalize_data(data.STG_raw)
    # STG_nan = normalize_data(data.STG_raw, "nn")
    IFG = normalize_data(data.IFG_raw)

    print("----------------------------")
    svm_clf = svm.SVC(kernel="linear", C=1)
    svm_scores = k_fold_training_and_validation(svm_clf, STG, data.image_labels)

    print("----------------------------")
    dtree_clf = DecisionTreeClassifier(random_state=0)
    dtree_scores = k_fold_training_and_validation(dtree_clf, STG, data.image_labels)

    print("----------------------------")
    knc = KNeighborsClassifier(n_neighbors=3)
    knc_scores = k_fold_training_and_validation(knc, STG, data.image_labels)

    print("----------------------------------------------------")
    evaluate_models(svm_scores, svm_clf, dtree_scores, dtree_clf)
    print("----------------------------------------------------")
    evaluate_models(svm_scores, svm_clf, knc_scores, knc)
    print("----------------------------------------------------")


def classify_STG(folds=5, test_size=0.3):
    data = PrepareData()
    STG = normalize_data(data.STG_raw)

    print("---------------------------------------------------")

    print("Subject labels STG:")
    result1 = train_and_test_model_accuracy(
        X=STG,
        y=data.subject_labels,
        classifier="svm",
        folds=folds,
        test_size=test_size,
        popmean=0.33,
    )
    print(result1)
    print("---------------------------------------------------")

    print("Subject labels STG:")
    result2 = train_and_test_model_accuracy(
        X=STG,
        y=data.subject_labels,
        classifier="n_neighbors",
        folds=folds,
        test_size=test_size,
        popmean=0.33,
    )
    print(result2)
    print("---------------------------------------------------")

    print("Image labels STG:")
    result3 = train_and_test_model_accuracy(
        X=STG,
        y=data.image_labels,
        classifier="svm",
        folds=folds,
        test_size=test_size,
        popmean=0.25,
    )
    print(result3)
    print("---------------------------------------------------")

    print("Image labels STG:")
    result4 = train_and_test_model_accuracy(
        X=STG,
        y=data.image_labels,
        classifier="n_neighbors",
        folds=folds,
        test_size=test_size,
        popmean=0.25,
    )
    print(result4)
    print("---------------------------------------------------")


def classify_IFG(folds=5, test_size=0.3):
    data = PrepareData()
    IFG = normalize_data(data.IFG_raw)

    print("---------------------------------------------------")

    print("Subject labels IFG:")
    result1 = train_and_test_model_accuracy(
        X=IFG,
        y=data.subject_labels,
        classifier="svm",
        folds=folds,
        test_size=test_size,
        popmean=0.33,
    )
    print(result1)
    print("---------------------------------------------------")

    print("Subject labels IFG:")
    result2 = train_and_test_model_accuracy(
        X=IFG,
        y=data.subject_labels,
        classifier="n_neighbors",
        folds=folds,
        test_size=test_size,
        popmean=0.33,
    )
    print(result2)
    print("---------------------------------------------------")

    print("Image labels IFG:")
    result3 = train_and_test_model_accuracy(
        X=IFG,
        y=data.image_labels,
        classifier="svm",
        folds=folds,
        test_size=test_size,
        popmean=0.25,
    )
    print(result3)
    print("---------------------------------------------------")

    print("Image labels IFG:")
    result4 = train_and_test_model_accuracy(
        X=IFG,
        y=data.image_labels,
        classifier="n_neighbors",
        folds=folds,
        test_size=test_size,
        popmean=0.25,
    )
    print(result4)
    print("---------------------------------------------------")


def classify_IRIS():
    X, y = datasets.load_iris(return_X_y=True)
    result = train_and_test_model_accuracy(
        X=X,
        y=y,
        classifier="svm",
        test_size=0.3,
        popmean=0.33,
    )
    print(result)


def classify_data(
    classifiers, labels, data, strategies, popmean, folds=5, test_size=0.3
):
    for strategy in strategies:
        X = normalize_data(data, strategy=strategy)
        print("---------------------------------------------------")

        for classifier in classifiers:
            print("llllllllll")
            results = train_and_test_model_accuracy(
                X=X,
                y=labels,
                classifier=classifier,
                folds=folds,
                test_size=test_size,
                popmean=popmean,
            )
            print(results)
            # ToDo return a object and save in the dict
            # return the dict
            # export the data in another function to a file
            print("---------------------------------------------------")


if __name__ == "__main__":
    # pass
    classify_STG(folds=1)
    classify_IFG(folds=1)
# classify_IRIS()
