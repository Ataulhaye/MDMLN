import scipy.stats as stats
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from BrainData import BrainData
from DataTraining import DataTraining
from EvaluateTrainingModel import EvaluateTrainingModel
from ExportData import ExportData
from PlotData import VisualizeData


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
    svm_scores = DataTraining.k_fold_training_and_validation(svm_clf, X, y)

    print("----------------------------")
    dtree_clf = DecisionTreeClassifier(random_state=0)
    dtree_scores = DataTraining.k_fold_training_and_validation(dtree_clf, X, y)

    print("----------------------------")
    knc = KNeighborsClassifier(n_neighbors=3)
    knc_scores = DataTraining.k_fold_training_and_validation(knc, X, y)

    print("----------------------------------------------------")
    EvaluateTrainingModel.evaluate_models(svm_scores, svm_clf, dtree_scores, dtree_clf)
    print("----------------------------------------------------")
    EvaluateTrainingModel.evaluate_models(svm_scores, svm_clf, knc_scores, knc)
    print("----------------------------------------------------")


def classify_STG_on_image_labels():
    data = BrainData()
    STG = BrainData.normalize_data(data.STG)
    # STG_nan = normalize_data(data.STG_raw, "nn")
    IFG = BrainData.normalize_data(data.IFG)

    print("----------------------------")
    svm_clf = svm.SVC(kernel="linear", C=1)
    svm_scores = DataTraining.k_fold_training_and_validation(
        svm_clf, STG, data.image_labels
    )

    print("----------------------------")
    dtree_clf = DecisionTreeClassifier(random_state=0)
    dtree_scores = DataTraining.k_fold_training_and_validation(
        dtree_clf, STG, data.image_labels
    )

    print("----------------------------")
    knc = KNeighborsClassifier(n_neighbors=3)
    knc_scores = DataTraining.k_fold_training_and_validation(
        knc, STG, data.image_labels
    )

    print("----------------------------------------------------")
    EvaluateTrainingModel.evaluate_models(svm_scores, svm_clf, dtree_scores, dtree_clf)
    print("----------------------------------------------------")
    EvaluateTrainingModel.evaluate_models(svm_scores, svm_clf, knc_scores, knc)
    print("----------------------------------------------------")


def classify_STG(folds=5, test_size=0.3):
    data = BrainData()
    STG = BrainData.normalize_data(data.STG)

    print("---------------------------------------------------")

    print("Subject labels STG:")
    result1 = DataTraining.train_and_test_model_accuracy(
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
    result2 = DataTraining.train_and_test_model_accuracy(
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
    result3 = DataTraining.train_and_test_model_accuracy(
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
    result4 = DataTraining.train_and_test_model_accuracy(
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
    data = BrainData()
    IFG = BrainData.normalize_data(data.IFG)

    print("---------------------------------------------------")

    print("Subject labels IFG:")
    result1 = DataTraining.train_and_test_model_accuracy(
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
    result2 = DataTraining.train_and_test_model_accuracy(
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
    result3 = DataTraining.train_and_test_model_accuracy(
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
    result4 = DataTraining.train_and_test_model_accuracy(
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
    result = DataTraining.train_and_test_model_accuracy(
        X=X,
        y=y,
        classifier="svm",
        test_size=0.3,
        popmean=0.33,
    )
    print(result)


def run_test():
    brain_data = BrainData(load_data=True)
    strategies = [
        "mean",
        "median",
        "most_frequent",
        "constant",
        "remove-voxels",
        "n_neighbors",
    ]
    classifiers = ["svm", "n_neighbors", "decisiontree"]
    labels = [(0.33, brain_data.image_labels), (0.25, brain_data.subject_labels)]
    IFG = brain_data.IFG[1]

    training = DataTraining()
    export_data = training.classify_brain_data(
        classifiers, labels=labels, data=IFG, strategies=strategies
    )

    export = ExportData()
    # export.create_and_write_CSV(export_data, "IFG-Results", "IFG")
    export.create_and_write_datasheet(
        export_data, "IFG-Results", "IFG-5-Folds-Classification"
    )


def analyse_nans():
    data = BrainData(load_data=True)
    nans_column_wise = data.calculate_nans_voxel_wise(data.STG[1])
    print("nans_column_wise", len(nans_column_wise))
    nans_voxel_wise = data.calculate_nans_trail_wise(data.STG[1])
    print("nans_voxel_wise", len(nans_voxel_wise))
    print("------------")


def visualize_nans():
    bd = BrainData(load_data=True)
    data_list = [bd.STG, bd.IFG]
    for dt in data_list:
        title, data = dt
        nans_column_wise = bd.calculate_nans_voxel_wise(data)
        columns = [i for i in range(data.shape[1])]
        VisualizeData.plot_bar_graph(
            ("Columns", columns),
            ("nans-length-column-wise", nans_column_wise),
            title=title,
        )

        nans_voxel_wise = bd.calculate_nans_trail_wise(data)
        rows = [i for i in range(data.shape[0])]
        VisualizeData.plot_bar_graph(
            ("nans-length-voxel-wise", nans_voxel_wise),
            ("rows", rows),
            bar_color="red",
            title=title,
        )

    # VisualizeData.plot_data_bar(np.array(x), np.array(nans_column_wise))


if __name__ == "__main__":
    # pass
    # analyse_nans()
    # visualize_nans()
    run_test()

# classify_STG(folds=1)
# classify_IFG(folds=1)
# classify_IRIS()
