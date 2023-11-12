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


class EvaluateTrainingModel:
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
