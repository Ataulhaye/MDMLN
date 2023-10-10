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

from init import evaluate_models, k_fold


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
    svm_scores = k_fold(svm_clf, X, y)

    print("----------------------------")
    dtree_clf = DecisionTreeClassifier(random_state=0)
    dtree_scores = k_fold(dtree_clf, X, y)

    print("----------------------------")
    knc = KNeighborsClassifier(n_neighbors=3)
    knc_scores = k_fold(knc, X, y)

    print("----------------------------------------------------")
    evaluate_models(svm_scores, svm_clf, dtree_scores, dtree_clf)
    print("----------------------------------------------------")
    evaluate_models(svm_scores, svm_clf, knc_scores, knc)
    print("----------------------------------------------------")


if __name__ == "__main__":
    run_evaluation()
