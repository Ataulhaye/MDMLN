import scipy.stats as stats


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

    def evaluate_training_model_ttest(classifier, popmean, scores, significance_level):
        classifier_name = type(classifier).__name__
        t_statistic, p_value = stats.ttest_1samp(a=scores, popmean=popmean)
        # p value less than 0.05 consider to be significant, greater than 0.05 is considered not to be significant
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

    def evaluate_training_model_by_ttest(
        classifier, popmean, scores, significance_level, data_label, strategy
    ):
        classifier_name = type(classifier).__name__
        t_statistic, p_value = stats.ttest_1samp(a=scores, popmean=popmean)
        # p value less than 0.05 consider to be significant, greater than 0.05 is considered not to be significant

        # results = dict({"tstatistic": t_statistic, "pvalue": p_value})
        nested_dict = dict({})

        percent = "{:0.2f}%".format((scores.mean() * 100))
        if p_value <= significance_level:
            # percent = "{:0.2f}%".format((scores.mean() * 100))
            nested_dict.update({"Significant:": percent})
            # return f"Performance of the {classifier_name} is significant. {percent}"
            # in this case rejecting null hypothesis: calssifier is performing as good as by chance
        else:
            nested_dict.update({"Not significant:": percent})
            # return f"{classifier_name} classifier performance is not significant. P-value is: {p_value}"  # not significantly different by chance
        # a = dict({classifier: results})#return also other info like which kernel...
        nested_dict.update({"p-value:": p_value})
        final_dict = dict(
            {classifier_name: dict({f"{strategy}-{data_label}": nested_dict})}
        )
        return final_dict
