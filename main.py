import scipy.stats as stats
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from Brain import Brain
from BrainDataConfig import BrainDataConfig
from BrainDataLabel import BrainDataLabel
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
    svm_scores = DataTraining.training_prediction_using_cross_validation(svm_clf, X, y)

    print("----------------------------")
    dtree_clf = DecisionTreeClassifier(random_state=0)
    dtree_scores = DataTraining.training_prediction_using_cross_validation(
        dtree_clf, X, y
    )

    print("----------------------------")
    knc = KNeighborsClassifier(n_neighbors=3)
    knc_scores = DataTraining.training_prediction_using_cross_validation(knc, X, y)

    print("----------------------------------------------------")
    EvaluateTrainingModel.evaluate_models(svm_scores, svm_clf, dtree_scores, dtree_clf)
    print("----------------------------------------------------")
    EvaluateTrainingModel.evaluate_models(svm_scores, svm_clf, knc_scores, knc)
    print("----------------------------------------------------")


def classify_IRIS():
    X, y = datasets.load_iris(return_X_y=True)
    iris_label = BrainDataLabel(name="IRIS", popmean=0.33, labels=y)
    result = DataTraining().train_and_test_model_accuracy(
        X=X,
        y=iris_label,
        classifier="SVM",
        test_size=0.2,
        popmean=iris_label.popmean,
        folds=5,
        predefined_split=True,
        strategy="m",
    )
    print(result)


def analyse_nans():
    config = BrainDataConfig()
    STG = Brain(area=config.STG, data_path=config.STG_path, load_labels=True)
    nans_column_wise = STG.calculate_nans_voxel_wise(STG.voxels)
    print("STG nans_column_wise", len(nans_column_wise))
    nans_voxel_wise = STG.calculate_nans_trail_wise(STG.voxels)
    print("STG nans_voxel_wise", len(nans_voxel_wise))
    print("------------")

    IFG = Brain(area=config.IFG, data_path=config.IFG_path, load_labels=True)
    nans_column_wise_IFG = IFG.calculate_nans_voxel_wise(IFG.voxels)
    print("IFG nans_column_wise", len(nans_column_wise_IFG))
    nans_voxel_wise_IFG = IFG.calculate_nans_trail_wise(IFG.voxels)
    print("IFG nans_voxel_wise", len(nans_voxel_wise_IFG))
    print("------------")


def visualize_nans():
    config = BrainDataConfig()
    STG = Brain(area=config.STG, data_path=config.STG_path, load_labels=True)
    IFG = Brain(area=config.IFG, data_path=config.IFG_path, load_labels=True)
    data_list = [STG, IFG]
    for data in data_list:
        nans_column_wise = STG.calculate_nans_voxel_wise(data.voxels)
        columns = [i for i in range(data.voxels.shape[1])]
        VisualizeData.plot_bar_graph(
            ("Columns", columns),
            ("nans-length-column-wise", nans_column_wise),
            title=data.area,
        )

        nans_voxel_wise = STG.calculate_nans_trail_wise(data.voxels)
        rows = [i for i in range(data.voxels.shape[0])]
        VisualizeData.plot_bar_graph(
            ("nans-length-voxel-wise", nans_voxel_wise),
            ("rows", rows),
            bar_color="red",
            title=data.area,
        )

    # VisualizeData.plot_data_bar(np.array(x), np.array(nans_column_wise))


def classify_STG(folds, test_size, classifiers, strategies):
    config = BrainDataConfig()
    STG = Brain(area=config.STG, data_path=config.STG_path, load_labels=True)
    labels = [STG.subject_labels, STG.image_labels]

    training = DataTraining()
    export_data = training.classify_brain_data(
        classifiers,
        labels=labels,
        data=STG.voxels,
        strategies=strategies,
        predefined_split=True,
        folds=folds,
        test_size=test_size,
    )
    export = ExportData()
    # export.create_and_write_CSV(export_data, "IFG-Results", "IFG")
    export.create_and_write_datasheet(
        export_data,
        f"STG-Results",
        f"STG-{folds}-Folds-Classification",
        transpose=False,
    )
    export.create_and_write_datasheet(
        export_data,
        f"STG-Results",
        f"STG-{folds}-Folds-Classification",
        transpose=True,
    )


def classify_IFG(folds, test_size, classifiers, strategies):
    config = BrainDataConfig()
    IFG = Brain(area=config.IFG, data_path=config.IFG_path, load_labels=True)
    labels = [IFG.subject_labels, IFG.image_labels]

    training = DataTraining()
    export_data = training.classify_brain_data(
        classifiers,
        labels=labels,
        data=IFG.voxels,
        strategies=strategies,
        predefined_split=True,
        folds=folds,
        test_size=test_size,
    )

    export = ExportData()
    # export.create_and_write_CSV(export_data, "IFG-Results", "IFG")
    export.create_and_write_datasheet(
        export_data,
        f"IFG-Results",
        f"IFG-{folds}-Folds-Classification",
        transpose=True,
    )
    export.create_and_write_datasheet(
        export_data,
        f"IFG-Results",
        f"IFG-{folds}-Folds-Classification",
        transpose=False,
    )


def main():
    # analyse_nans()
    # visualize_nans()
    classify_IRIS()
    folds = 5
    test_size = 0.2
    strategies = [
        # None,
        "mean",
        "median",
        "most_frequent",
        # "constant",
        "remove-voxels",
        # "n_neighbors",
    ]
    classifiers = [
        # "DecisionTree",
        # "HistGradientBoosting",
        "SVM",
        "KNearestNeighbors",
        # "GaussianNaiveBayes",
        "LinearDiscriminant",
        # "MLP",
        # "LogisticRegression",
        # "RandomForest",
    ]

    classify_STG(folds, test_size, classifiers, strategies)
    classify_IFG(folds, test_size, classifiers, strategies)


if __name__ == "__main__":
    main()
