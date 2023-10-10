import random

import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split


def k_fold(classifier, X, y, folds=10):
    score_array = []
    for i in range(folds):
        f_train, f_test, t_train, t_test = train_test_split(X, y, test_size=0.2)
        classifier.fit(f_train, t_train)
        score_array.append(classifier.score(f_test, t_test))
    print(f"score_array using {type(classifier).__name__}:", score_array)
    score_array = np.array(score_array)
    print(
        "%0.2f accuracy with a standard deviation of %0.2f"
        % (score_array.mean(), score_array.std())
    )
    return score_array


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
