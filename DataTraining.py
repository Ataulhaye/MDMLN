import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from BrainData import BrainData
from EvaluateTrainingModel import EvaluateTrainingModel


class DataTraining:
    def k_fold_training_and_validation(
        self, model: BaseEstimator, X, y, folds=10, test_size=0.2
    ):
        score_array = []
        for i in range(folds):
            # split the data set randomly into test and train sets
            # random_state=some number will always output the same sets by every execution
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size
            )
            model.fit(X_train, y_train)
            score_array.append(model.score(X_test, y_test))
        print(
            f"scores using {type(model).__name__} with {folds}-fold cross-validation:",
            score_array,
        )
        score_array = np.array(score_array)
        print(
            f"{type(model).__name__}: %0.2f accuracy with a standard deviation of %0.2f"
            % (score_array.mean(), score_array.std())
        )
        # print("-------------------------------------")
        return score_array

    def train_and_test_model_accuracy(
        self,
        X,
        y,
        classifier="SVM",
        folds=5,
        test_size=0.3,
        popmean=0.3,
        significance_level=0.05,
        strategy=None,
    ):
        """Performe k-Fold classification, training and testing
        Args:
            X (_type_): numpy.ndarray data
            y (_type_): numpy.ndarray labels
            classifier (str, optional): _description_. Defaults to "SVM" SVC kernal is linear.
            if 'KNearestNeighbors' then KNeighborsClassifier
            if 'DecisionTree' then DecisionTreeClassifier
            if 'LinearDiscriminant' then LinearDiscriminantAnalysis
            if 'GaussianNaiveBayes' then GaussianNB
            folds (int, optional): _description_. Defaults to 5.
            test_size (float, optional): size of test data. Defaults to 0.3.
            popmean (float, optional): popmean of data. Defaults to 0.3.
            significance_level (float, optional): significance level of 0.05 indicates a 5% risk of concluding that a difference exists when there is no actual difference. Defaults to 0.05.

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        model = None
        if classifier == "SVM":
            model = svm.SVC(kernel="linear", C=1)
        elif classifier == "KNearestNeighbors":
            model = KNeighborsClassifier(n_neighbors=3)
        elif classifier == "DecisionTree":
            model = DecisionTreeClassifier(random_state=0)
        elif classifier == "GaussianNaiveBayes":
            model = GaussianNB()
        elif classifier == "LinearDiscriminant":
            model = LinearDiscriminantAnalysis()
        else:
            raise TypeError("Classifier Not Supported")

        scores = DataTraining().k_fold_training_and_validation(
            model=model, X=X, y=y[1], folds=folds, test_size=test_size
        )
        return EvaluateTrainingModel().evaluate_training_model_by_ttest_list(
            classifier, popmean, scores, significance_level, y[0], strategy
        )

    def classify_brain_data(
        self, classifiers: list[str], labels, data, strategies, folds=5, test_size=0.3
    ):
        # data_dict = dict({})
        data_list = list()
        for strategy in strategies:
            X = BrainData().normalize_data(data, strategy=strategy)
            for tp in labels:
                mean, label = tp
                for classifier in classifiers:
                    results = self.train_and_test_model_accuracy(
                        X=X,
                        y=label,
                        classifier=classifier,
                        folds=folds,
                        test_size=test_size,
                        popmean=mean,
                        strategy=strategy,
                    )
                    data_list.append(results)
                    # key = list(results.keys())[0]
                    # if key in data_dict:
                    # value = next(iter(results.values()))
                    # data_dict.setdefault(key).update(value)
                    # else:
                    # data_dict.update(results)
        # return data_dict
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
