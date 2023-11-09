import random

import numpy as np
import scipy.stats as stats
from sklearn import datasets, metrics, preprocessing, svm
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    ShuffleSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def k_fold_training_and_validation(
    classifier: BaseEstimator, X, y, folds=10, test_size=0.2
):
    score_array = []
    for i in range(folds):
        # split the data set randomly into test and train sets
        # random_state=some number will always output the same sets by every execution
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        classifier.fit(X_train, y_train)
        score_array.append(classifier.score(X_test, y_test))
    print(
        f"scores using {type(classifier).__name__} with {folds}-fold cross-validation:",
        score_array,
    )
    score_array = np.array(score_array)
    print(
        f"{type(classifier).__name__}: %0.2f accuracy with a standard deviation of %0.2f"
        % (score_array.mean(), score_array.std())
    )
    # print("-------------------------------------")
    return score_array


def train_and_test_model_accuracy(
    X, y, classifier="svm", folds=5, test_size=0.3, popmean=0.3, significance_level=0.05
):
    """Performe k-Fold classification, training and testing
    Args:
        X (_type_): numpy.ndarray data
        y (_type_): numpy.ndarray labels
        classifier (str, optional): _description_. Defaults to "svm" SVC kernal is linear.
        if 'n_neighbors' then KNeighborsClassifier
        if 'decisiontree' then DecisionTreeClassifier
        folds (int, optional): _description_. Defaults to 5.
        test_size (float, optional): size of test data. Defaults to 0.3.
        popmean (float, optional): popmean of data. Defaults to 0.3.
        significance_level (float, optional): significance level of 0.05 indicates a 5% risk of concluding that a difference exists when there is no actual difference. Defaults to 0.05.

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    if classifier == "svm":
        classifier = svm.SVC(kernel="linear", C=1)
    elif classifier == "n_neighbors":
        classifier = KNeighborsClassifier(n_neighbors=3)
    elif classifier == "decisiontree":
        classifier = DecisionTreeClassifier(random_state=0)
    elif classifier == "gaussian":
        raise NotImplementedError()
        # classifier = GaussianNB()
    elif classifier == "lineardiscriminant":
        raise NotImplementedError()
        # classifier = LinearDiscriminantAnalysis()
    else:
        raise TypeError("Classifier Not Supported")

    classifier_name = type(classifier).__name__

    scores = k_fold_training_and_validation(
        classifier=classifier, X=X, y=y, folds=folds, test_size=test_size
    )
    t_statistic, p_value = stats.ttest_1samp(a=scores, popmean=popmean)
    # p value less then 0.05 consider to be significant, greater then 0.05 is consider not to be significant
    print(
        f"{type(classifier).__name__} test results are: t_statistic , p_value",
        t_statistic,
        p_value,
    )

    if p_value <= significance_level:
        percent = "{:0.2f}%".format((scores.mean() * 100))
        return f"Performance of the {classifier_name} is significant. {percent}"
        # in this case rejecting null hypothesis: calssifier is performing as good as by chance
    else:
        return f"{classifier_name} classifier performance is not significant. P-value is: {p_value}"  # not significantly different by chance


def evaluate_models(
    est1_scores, estimator1, est2_scores, estimator2, significance_level=0.05
):
    # significance level of 0.05 indicates a 5% risk of concluding that a difference exists when there is no actual difference.
    # Ein Signifikanzniveau von α = 0,05 bedeutet, dass man 5 % Fehlerwahrscheinlichkeit akzeptiert.
    # Ist ein Test mit α = 0,05 signifikant, so ist unser Ergebnis mit nur 5 % Wahrscheinlichkeit zufällig entstanden
    est1_mean = est1_scores.mean()
    est2_mean = est2_scores.mean()
    estimator1_name = {type(estimator1).__name__}
    estimator2_name = {type(estimator2).__name__}

    est1_vs_est2 = stats.ttest_ind(est1_scores, est2_scores)
    print(f"{estimator1_name} vs {estimator2_name}", est1_vs_est2)

    if est1_vs_est2.pvalue > significance_level:
        print(
            f"Performance of the {estimator1_name} and {estimator2_name} is not significantly different. P-value is:",
            est1_vs_est2.pvalue,
        )  # in this case unable to reject the H0
    else:
        if est1_mean > est2_mean:
            print(
                f"{estimator1_name} performance is better than {estimator2_name}",
                est1_mean * 100,
            )
        else:
            print(
                f"{estimator2_name} performance is better than {estimator1_name}",
                est2_mean * 100,
            )


"""
random_y = []
for i in range(y_train.size):
    random_y.append(random.randint(0,2))
    
random_y = np.array(random_y)

scores_rndm_labels = cross_val_score(clf, X_train, random_y, cv=5)
print("Scores_rndm_labels:", scores_rndm_labels)

print("%0.2f accuracy on random assigned labels with a standard deviation of %0.2f" % (scores_rndm_labels.mean(), scores_rndm_labels.std()))
"""
