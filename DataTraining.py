import time
from datetime import datetime
from math import ceil, floor

import matplotlib.pyplot as plt

plt.switch_backend("agg")
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import TensorDataset

from Brain import Brain
from BrainDataConfig import BrainDataConfig
from BrainDataLabel import BrainDataLabel
from BrainTrainUtils import test_autoencoder_braindata, train_autoencoder_braindata
from EvaluateTrainingModel import EvaluateTrainingModel
from ExportData import ExportData
from ExportEntity import ExportEntity
from TestTrainingSet import TestTrainingSet, TestTrainingTensorDataset
from TrainingConfig import TrainingConfig


class DataTraining:
    nan_classifiers = ["DecisionTree", "HistGradientBoosting", "LGBM", "CatBoost"]

    def training_prediction_using_cross_validation(
        self, model, brain: Brain, train_config: TrainingConfig
    ):
        scores = []
        for i in range(train_config.folds):
            train_test_set = None

            if train_config.predefined_split:
                train_test_set = self.premeditate_random_train_test_split(
                    brain, train_config
                )
            else:
                train_test_set = self.random_train_test_split(
                    brain, train_config.test_size
                )

            train_test_set = brain.normalize_data_safely(
                strategy=train_config.strategy, data_set=train_test_set
            )

            if train_config.dimension_reduction:
                train_test_set = self.apply_PCA(train_test_set)

            if train_config.use_autoencoder:
                train_test_set = self.apply_autoencoder(
                    brain, train_config, train_test_set
                )

            if train_config.use_autoencoder and train_config.tsne:
                self.t_sne(train_test_set, brain, train_config, i, type(model).__name__)

            try:
                print("Starting fitting of the model.")
                model.fit(train_test_set.X_train, train_test_set.y_train)
                print("Finished the fitting of the model.")
                score = model.score(train_test_set.X_test, train_test_set.y_test)
                print("Score", score)
                scores.append(score)
            except (Exception, ValueError, RuntimeError, TypeError, NameError) as err:
                print("Error!", err)

            if train_config.explain and "binary" in brain.current_labels.name:
                self.explain_model(
                    model,
                    brain=brain,
                    tt_set=train_test_set,
                    scores=scores,
                    train_config=train_config,
                )
        # print(f"scores using {type(model).__name__} with {folds}-fold cross-validation:",score_array,)
        scores = np.array(scores)

        # print(f"{type(model).__name__}: %0.2f accuracy with a standard deviation of %0.2f"% (score_array.mean(), score_array.std()))
        return scores

    def apply_autoencoder(
        self,
        brain: Brain,
        train_config: TrainingConfig,
        train_test_set: TestTrainingSet,
    ):
        tensor_datasets = self.to_tensor_datasets(train_test_set)
        # check the input dim config
        # voxel_dim = tensor_datasets.train_set.tensors[0].shape[1]
        # input_dim = train_config.best_autoencoder_config["input_dim"]
        if (
            train_config.best_autoencoder_config["input_dim"]
            != tensor_datasets.train_set.tensors[0].shape[1]
        ):
            print("Model config:", train_config.best_autoencoder_config)
            train_config.best_autoencoder_config["input_dim"] = (
                tensor_datasets.train_set.tensors[0].shape[1]
            )
            print("Changed model config:", train_config.best_autoencoder_config)
        train_config.best_autoencoder_config["brain_area"] = (
            f"{brain.area}_{train_config.strategy}"
        )
        (
            autoencoder_model,
            train_encodings,
            train_labels,
        ) = train_autoencoder_braindata(
            train_config.best_autoencoder_config, tensor_datasets
        )
        loss, test_encoding, test_labels = test_autoencoder_braindata(
            autoencoder_model, tensor_datasets.test_set
        )
        train_test_set = TestTrainingSet(
            X_train=train_encodings,
            X_test=test_encoding,
            y_train=train_labels,
            y_test=test_labels,
        )
        return train_test_set

    def apply_PCA(self, train_test_set: TestTrainingSet):
        x_dim = train_test_set.X_train.shape[1]
        pca = PCA(n_components=0.99, svd_solver="full")
        pca.fit(train_test_set.X_train)
        pca_x_train = pca.transform(train_test_set.X_train)
        pca_x_test = pca.transform(train_test_set.X_test)
        print("explained_variance_ratio: ", pca.explained_variance_ratio_.sum())
        train_test_set.X_train = pca_x_train
        train_test_set.X_test = pca_x_test
        print(
            f"Data reduced from {x_dim} dimensions to {train_test_set.X_train.shape[1]} dimensions"
        )
        return train_test_set

    def t_sne(
        self,
        train_test_set: TestTrainingSet,
        brain: Brain,
        train_config: TrainingConfig,
        fold: int,
        model_name: str,
    ):
        time_start = time.time()
        components = train_test_set.X_train.shape[1]  # it is two dimensional
        tsne = TSNE(n_components=components, perplexity=100)
        # tsne.fit(train_test_set.X_train)
        # tsne_results = tsne.fit_transform(train_test_set.X_test)
        tsne_results = tsne.fit_transform(train_test_set.X_train)
        print("t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))

        df_tsne = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2"])
        # Add labels
        # df_tsne["labels"] = train_test_set.y_train
        df_tsne["labels"] = train_test_set.y_train

        print(df_tsne)
        scatter_plot = sns.scatterplot(
            data=df_tsne,
            x="TSNE1",
            y="TSNE2",
            hue="labels",
            palette="bright",
            legend="full",
        )
        fig = scatter_plot.get_figure()
        name = f"{brain.area}_{brain.current_labels.name[:-4]}_{train_config.strategy}-{fold}th_fold-{model_name}"
        graph_name = ExportData.get_file_name(".png", name)
        plt.title(f"Using t-SNE {name}")
        fig.savefig(graph_name, dpi=1200)
        plt.close()
        print("Finshed")

    def explain_model(
        self,
        model,
        brain: Brain,
        tt_set: TestTrainingSet,
        scores,
        train_config: TrainingConfig,
    ):
        # explain all the predictions in the test set
        # explainer = shap.KernelExplainer(model.predict_proba, x_train)
        explainer = shap.KernelExplainer(model.predict, tt_set.X_train)
        shap_values = explainer.shap_values(tt_set.X_test)
        # shap.force_plot(explainer.expected_value[0], shap_values[0], x_test)

        shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values,
            features=tt_set.X_test,
        )
        name = (
            f"{brain.current_labels.name}_{round(scores[train_config.folds-1],2)}_force"
        )
        graph_name = self.get_graph_file_name(name=name)
        plt.savefig(graph_name, dpi=700)
        plt.close()

        name = f"{brain.current_labels.name}_{round(scores[train_config.folds-1],2)}_decision"
        graph_name = self.get_graph_file_name(name=name)
        shap.decision_plot(explainer.expected_value, shap_values, tt_set.X_test)
        plt.savefig(graph_name, dpi=700)
        plt.close()

        name = f"{brain.current_labels.name}_{round(scores[train_config.folds-1],2)}_summary"
        graph_name = self.get_graph_file_name(name=name)
        shap.summary_plot(shap_values=shap_values, features=tt_set.X_test)
        plt.savefig(graph_name, dpi=700)
        plt.close()

    @staticmethod
    def get_graph_file_name(name, extension=".svg"):
        dt = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        return f"{name}_{dt}{extension}"

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

    def random_train_test_split(self, brain: Brain, test_size):
        # split the data set randomly into test and train sets
        # random_state=some number will always output the same sets by every execution
        x_train, x_test, y_train, y_test = train_test_split(
            brain.voxels, brain.current_labels.labels, test_size=test_size
        )
        set = TestTrainingSet()
        set.X_train = x_train
        set.y_train = y_train
        set.X_test = x_test
        set.y_test = y_test
        return set

    def premeditate_random_train_test_split(
        self, brain: Brain, train_config: TrainingConfig, config: BrainDataConfig = None
    ):
        x_test, x_train, y_test, y_train = [], [], [], []
        train_size = 1.0 - train_config.test_size
        sample_start = 0
        sample_stop = 0
        if config is None:
            config = BrainDataConfig()
        for subset_size in config.patients:
            subset_samples = subset_size * config.conditions
            n_test = ceil(subset_size * train_config.test_size)
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
            self.extract_subset_chunk(
                brain.voxels,
                x_test,
                brain.current_labels.labels,
                y_test,
                subset_test_ind,
                config,
            )
            self.extract_subset_chunk(
                brain.voxels,
                x_train,
                brain.current_labels.labels,
                y_train,
                subset_train_ind,
                config,
            )

        set = TestTrainingSet()
        set.X_test = np.array(x_test)
        set.X_train = np.array(x_train)
        set.y_test = np.array(y_test)
        set.y_train = np.array(y_train)
        return set

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
        brain: Brain,
        train_config: TrainingConfig,
    ):
        """Performs k-Fold classification, training and testing
        Args:
        Raises:
            TypeError: _description_

        Returns:
            ExportEntity: _description_
        """
        model = None

        match train_config.classifier:
            case "SVM":
                model = svm.SVC(kernel="linear", C=1)
            case "KNearestNeighbors":
                model = KNeighborsClassifier(n_neighbors=3)
            case "DecisionTree":
                model = DecisionTreeClassifier(random_state=0)
            case "GaussianNaiveBayes":
                model = GaussianNB()
            case "LinearDiscriminant":
                model = LinearDiscriminantAnalysis()
            case "MLP":
                model = MLPClassifier()
            case "LogisticRegression":
                model = LogisticRegression()
            case "RandomForest":
                model = RandomForestClassifier(max_depth=2, random_state=0)
            case "LGBM":
                model = LGBMClassifier()
            case "CatBoost":
                model = CatBoostClassifier(verbose=0, n_estimators=100)
            case "HistGradientBoosting":
                model = HistGradientBoostingClassifier()
            case _:
                raise TypeError("Classifier Not Supported")

        if (
            train_config.strategy is None
            and train_config.classifier not in train_config.nan_classifiers
        ):
            return ExportEntity(
                p_value=None,
                row_name=type(model).__name__,
                sub_column_name=train_config.strategy,
                column_name=brain.current_labels.name,
                result=tuple(("", "")),
            )
        start = time.time()
        print(
            f"Started training and prediction of model: {type(model).__name__} using strategy as {train_config.strategy} on {brain.current_labels.name} with {train_config.folds}-fold"
        )

        scores = self.training_prediction_using_cross_validation(
            model=model, train_config=train_config, brain=brain
        )
        # scores = self.training_prediction_using_default_cross_validation(model=model,x=x,y=y.labels,folds=folds,test_size=test_size,predefined_split=predefined_split,)

        print(
            f"Scores of {type(model).__name__} using strategy as {train_config.strategy} on {brain.current_labels.name} with default {train_config.folds}-fold cross-validation:",
            scores,
        )
        end = time.time()
        print(
            f"Finished training and prediction of model: {type(model).__name__} using strategy as {train_config.strategy} on {brain.current_labels.name} with {train_config.folds}-fold in {round(((end - start)/60),2)} minutes."
        )
        return EvaluateTrainingModel().evaluate_training_model_by_ttest(
            model,
            brain.current_labels.popmean,
            scores,
            brain.current_labels.name,
            train_config.strategy,
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
        partially=False,
        dimension_reduction=False,
        explain=False,
    ):
        # data_dict = dict({})
        data_list = list()
        config = BrainDataConfig()
        for strategy in strategies:
            b = Brain()
            X = b.normalize_data(data, strategy=strategy)
            for label in labels:
                if partially:
                    subset = b.voxels_labels_subset(X, 25, config, label)
                    X = subset[0]
                    label = subset[1]
                if dimension_reduction:
                    pca = PCA(n_components=50)  # n_components=300
                    X = pca.fit_transform(X)

                for classifier in classifiers:
                    results = self.train_and_test_model_accuracy(
                        x=X,
                        y=label,
                        popmean=label.popmean,
                        folds=folds,
                        test_size=test_size,
                        strategy=strategy,
                        predefined_split=predefined_split,
                        classifier=classifier,
                        explaination=explain,
                    )
                    data_list.append(results)
        return data_list

    def brain_data_classification(
        self,
        brain: Brain,
        train_config: TrainingConfig,
        strategies: list[str],
        classifiers: list[str],
    ):
        data_list = list()
        config = BrainDataConfig()

        if train_config.partially:
            brain.brain_subset(25, config)

        for classifier in classifiers:
            train_config.classifier = classifier
            for strategy in strategies:
                train_config.strategy = strategy
                results = self.train_and_test_model_accuracy(brain, train_config)
                data_list.append(results)

        return data_list

    def get_tensor_datasets(
        self, brain: Brain, train_config: TrainingConfig, tt_set: TestTrainingSet
    ):
        bd_config = BrainDataConfig()

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

        modify_tt_set = self.premeditate_random_train_test_split(
            modify_brain, train_config, modify_bd_config
        )

        XT_train = torch.Tensor(modify_tt_set.X_train)
        XT_val = torch.Tensor(modify_tt_set.X_test)
        # yT_train = torch.Tensor(modify_tt_set.y_train)
        # yT_val = torch.Tensor(modify_tt_set.y_test)
        yT_train = torch.tensor(modify_tt_set.y_train, dtype=torch.int)
        yT_val = torch.tensor(modify_tt_set.y_test, dtype=torch.int)

        tr_set = TensorDataset(XT_train, yT_train)
        vl_set = TensorDataset(XT_val, yT_val)
        # ts_set = TensorDataset(torch.Tensor(tt_set.X_test), torch.Tensor(tt_set.y_test))
        ts_set = TensorDataset(
            torch.Tensor(tt_set.X_test), torch.tensor(tt_set.y_test, dtype=torch.int)
        )
        sets = TestTrainingTensorDataset(
            train_set=tr_set, val_set=vl_set, test_set=ts_set
        )

        # file_name = f"{brain.area}_{train_config.strategy}_static_wholeSet.pickle"
        # with open(file_name, "wb") as output:
        # pickle.dump(sets, output)

        return sets

    def to_tensor_datasets(self, tt_set: TestTrainingSet):

        tr_set = TensorDataset(
            torch.Tensor(tt_set.X_train), torch.tensor(tt_set.y_train, dtype=torch.int)
        )

        ts_set = TensorDataset(
            torch.Tensor(tt_set.X_test), torch.tensor(tt_set.y_test, dtype=torch.int)
        )
        sets = TestTrainingTensorDataset(train_set=tr_set, test_set=ts_set)

        # file_name = f"{brain.area}_{train_config.strategy}_static_wholeSet.pickle"
        # with open(file_name, "wb") as output:
        # pickle.dump(sets, output)

        return sets


"""
random_y = []
for i in range(y_train.size):
    random_y.append(random.randint(0,2))

random_y = np.array(random_y)

scores_rndm_labels = cross_val_score(clf, X_train, random_y, cv=5)
print("Scores_rndm_labels:", scores_rndm_labels)

print("%0.2f accuracy on random assigned labels with a standard deviation of %0.2f" % (scores_rndm_labels.mean(), scores_rndm_labels.std()))
"""
