import math
import time
from math import ceil, floor
from random import randrange

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from Brain import Brain
from BrainDataConfig import BrainDataConfig
from BrainDataLabel import BrainDataLabel
from EvaluateTrainingModel import EvaluateTrainingModel
from ExportEntity import ExportEntity


class DataTraining:
    nan_classifiers = ["DecisionTree", "HistGradientBoosting", "LGBM", "CatBoost"]

    def training_prediction_using_cross_validation(
        self,
        model,
        x,
        y,
        folds: int = 5,
        test_size: float = 0.2,
        predefined_split: bool = True,
    ):
        score_array = []
        for i in range(folds):
            x_test, x_train, y_test, y_train = None, None, None, None
            if predefined_split:
                (
                    x_test,
                    x_train,
                    y_test,
                    y_train,
                ) = self.premeditate_random_train_test_split(x, y, test_size)
            else:
                x_test, x_train, y_test, y_train = self.random_train_test_split(
                    x, y, test_size
                )

            model.fit(x_train, y_train)
            score_array.append(model.score(x_test, y_test))

        self.explain_model(
            model,
            x,
            y,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        # print(f"scores using {type(model).__name__} with {folds}-fold cross-validation:",score_array,)
        score_array = np.array(score_array)

        # print(f"{type(model).__name__}: %0.2f accuracy with a standard deviation of %0.2f"% (score_array.mean(), score_array.std()))
        return score_array

    def explain_model(
        self,
        model,
        x,
        y,
        x_train,
        x_test,
        y_train,
        y_test,
    ):
        # explain all the predictions in the test set
        explainer = shap.KernelExplainer(model.predict_proba, x_train)
        shap_values = explainer.shap_values(x_test)
        shap.force_plot(explainer.expected_value[0], shap_values[0], x_test)

        plt.savefig("shap_summary.svg", dpi=700)
        plt.close()

    def training_prediction_using_default_cross_validation(
        self,
        model,
        x,
        y,
        folds: int = 5,
        test_size: float = 0.2,
        predefined_split: bool = True,
    ):
        score_array = []

        score_array = cross_val_score(
            model,
            x,
            y,
            cv=folds,
        )

        return score_array

    @staticmethod
    def random_train_test_split(x, y, test_size):
        # split the data set randomly into test and train sets
        # random_state=some number will always output the same sets by every execution
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        return x_test, x_train, y_test, y_train

    def premeditate_random_train_test_split(self, x, y, test_size: float):
        x_test, x_train, y_test, y_train = [], [], [], []
        train_size = 1.0 - test_size
        sample_start = 0
        sample_stop = 0
        config = BrainDataConfig()
        for subset_size in config.patients:
            subset_samples = subset_size * config.conditions
            n_test = ceil(subset_size * test_size)
            n_train = floor(subset_size * train_size)
            sample_stop = sample_stop + subset_samples
            subset_indices = np.arange(
                start=sample_start, stop=sample_stop, step=config.conditions
            )
            sample_start = sample_stop
            rng = np.random.mtrand._rand
            permutation = rng.permutation(subset_indices)
            subset_test_ind = permutation[:n_test]
            subset_train_ind = permutation[n_test : (n_test + n_train)]
            self.extract_subset_chunk(x, x_test, y, y_test, subset_test_ind, config)
            self.extract_subset_chunk(x, x_train, y, y_train, subset_train_ind, config)

        return np.array(x_test), np.array(x_train), np.array(y_test), np.array(y_train)

    @staticmethod
    def extract_subset_chunk(
        x: list,
        x_subset: list,
        y: list,
        y_subset: list,
        subset_indices: list,
        config: BrainDataConfig,
    ):
        for start in subset_indices:
            end = start + config.conditions
            x_subset.extend(x[start:end])
            y_subset.extend(y[start:end])

    def train_and_test_model_accuracy(
        self,
        x,
        y: BrainDataLabel,
        popmean,
        folds,
        test_size,
        strategy,
        predefined_split,
        classifier="SVM",
    ):
        """Performs k-Fold classification, training and testing
        Args:
            x (_type_): numpy.ndarray data
            y (_type_): numpy.ndarray labels
            classifier (str, optional): _description_. Defaults to "SVM" SVC kernal is linear.
            if 'KNearestNeighbors' then KNeighborsClassifier
            if 'DecisionTree' then DecisionTreeClassifier
            if 'LinearDiscriminant' then LinearDiscriminantAnalysis
            if 'GaussianNaiveBayes' then GaussianNB
            folds (int, optional): _description_. Defaults to 5.
            test_size (float, optional): size of test data. Defaults to 0.3.
            strategy: used for data normalization
            predefined_split: if True the split will be according to the BrainDataConfig conditions
            popmean (float, optional): popmean of data. Defaults to 0.3.
        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        model = None
        if classifier == "SVM":
            model = svm.SVC(kernel="linear", C=1, probability=True)
        elif classifier == "KNearestNeighbors":
            model = KNeighborsClassifier(n_neighbors=3)
        elif classifier == "DecisionTree":
            model = DecisionTreeClassifier(random_state=0)
        elif classifier == "GaussianNaiveBayes":
            model = GaussianNB()
        elif classifier == "LinearDiscriminant":
            model = LinearDiscriminantAnalysis()
        elif classifier == "MLP":
            model = MLPClassifier()
        elif classifier == "LogisticRegression":
            model = LogisticRegression()
        elif classifier == "RandomForest":
            model = RandomForestClassifier(max_depth=2, random_state=0)
        # elif classifier == "XGBoost":
        # model = XGBClassifier()
        elif classifier == "LGBM":
            model = LGBMClassifier()
        elif classifier == "CatBoost":
            model = CatBoostClassifier(verbose=0, n_estimators=100)
        elif classifier == "HistGradientBoosting":
            model = HistGradientBoostingClassifier()
        else:
            raise TypeError("Classifier Not Supported")

        if strategy is None and classifier not in self.nan_classifiers:
            return ExportEntity(
                p_value=None,
                row_name=type(model).__name__,
                sub_column_name=strategy,
                column_name=y.name,
                result=tuple(("", "")),
            )
        start = time.time()
        print(
            f"Started training and prediction of model: {type(model).__name__} using strategy as {strategy} on {y.name} with {folds}-fold"
        )

        scores = self.training_prediction_using_cross_validation(
            model=model,
            x=x,
            y=y.labels,
            folds=folds,
            test_size=test_size,
            predefined_split=predefined_split,
        )
        # scores = self.training_prediction_using_default_cross_validation(model=model,x=x,y=y.labels,folds=folds,test_size=test_size,predefined_split=predefined_split,)

        print(
            f"Scores of {type(model).__name__} using strategy as {strategy} on {y.name} with default {folds}-fold cross-validation:",
            scores,
        )
        end = time.time()
        print(
            f"Finished training and prediction of model: {type(model).__name__} using strategy as {strategy} on {y.name} with {folds}-fold in {round(((end - start)/60),2)} minutes."
        )
        return EvaluateTrainingModel().evaluate_training_model_by_ttest(
            model, popmean, scores, y.name, strategy
        )

    def classify_brain_data(
        self,
        classifiers: list[str],
        labels: list[BrainDataLabel],
        data,
        strategies,
        predefined_split,
        folds,
        test_size,
    ):
        # data_dict = dict({})
        data_list = list()
        for strategy in strategies:
            x = Brain().normalize_data(data, strategy=strategy)
            for label in labels:
                for classifier in classifiers:
                    results = self.train_and_test_model_accuracy(
                        x=x,
                        y=label,
                        popmean=label.popmean,
                        folds=folds,
                        test_size=test_size,
                        strategy=strategy,
                        predefined_split=predefined_split,
                        classifier=classifier,
                    )
                    data_list.append(results)
        return data_list


"""
random_y = []
for i in range(y_train.size):
    random_y.append(random.randint(0,2))

random_y = np.array(random_y)

scores_rndm_labels = cross_val_score(clf, X_train, random_y, cv=5)
print("Scores_rndm_labels:", scores_rndm_labels)

print("%0.2f accuracy on random assigned labels with a standard deviation of %0.2f" % (scores_rndm_labels.mean(), scores_rndm_labels.std()))
"""
