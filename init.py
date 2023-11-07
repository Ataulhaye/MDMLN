import random

import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, preprocessing, svm
from sklearn.model_selection import (
    ShuffleSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def k_fold_training_and_validation(classifier, X, y, folds=10, test_size=0.2):
    score_array = []
    for i in range(folds):
        # split the data set randomly into test and train sets
        # random_state=some number will always output the same sets by every execution
        f_train, f_test, t_train, t_test = train_test_split(X, y, test_size=test_size)
        classifier.fit(f_train, t_train)
        score_array.append(classifier.score(f_test, t_test))
    print(f"score_array using {type(classifier).__name__}:", score_array)
    score_array = np.array(score_array)
    print(
        "%0.2f accuracy with a standard deviation of %0.2f"
        % (score_array.mean(), score_array.std())
    )
    return score_array


def train_and_test_model_accuracy(X, y, classifier="svm", folds=1, test_size=0.3, popmean=0.3, significance_level=0.05):
    if classifier=="svm":
        classifier = svm.SVC(kernel="linear", C=1)
    elif classifier=="n_neighbors":
        classifier = DecisionTreeClassifier(random_state=0)
    elif classifier=="decisiontree":
        classifier = KNeighborsClassifier(n_neighbors=3)
    else:
        raise TypeError("Classifier Not Supported ")

    scores =  k_fold_training_and_validation(classifier, X, y, folds=folds, test_size=test_size)
    t_statistic, p_value = stats.ttest_1samp(a=scores, popmean=popmean)
    # p value less then 0.05 consider to be significant, greater then 0.05 is consider not to be significant
    print("t_statistic , p_value", t_statistic, p_value)

    classifier_name = {type(classifier).__name__}

    if p_value > significance_level:
        print(
            f"Performance of the {classifier_name} is not significantly different. P-value is:",
            p_value,
        )  # in this case unable to reject the H0
    else:
        percent = "{:0.2f}%".format((scores.mean() * 100))
        print(
                f"{classifier_name} performance is better than by chance", percent
            )

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
