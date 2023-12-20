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


def classify_iris():
    x, y = datasets.load_iris(return_X_y=True)
    iris_label = BrainDataLabel(name="IRIS", popmean=0.33, labels=y)
    result = DataTraining().train_and_test_model_accuracy(
        x=x,
        y=iris_label,
        classifier="LGBM",
        test_size=0.2,
        popmean=iris_label.popmean,
        folds=5,
        predefined_split=True,
        strategy="m",
    )
    print(result)


def analyse_nans():
    config = BrainDataConfig()
    stg = Brain(area=config.STG, data_path=config.STG_path, load_labels=True)
    nans_column_wise = stg.calculate_nans_voxel_wise(stg.voxels)
    print("stg nans_column_wise", len(nans_column_wise))
    nans_voxel_wise = stg.calculate_nans_trail_wise(stg.voxels)
    print("stg nans_voxel_wise", len(nans_voxel_wise))
    print("------------")

    ifg = Brain(area=config.IFG, data_path=config.IFG_path, load_labels=True)
    nans_column_wise_ifg = ifg.calculate_nans_voxel_wise(ifg.voxels)
    print("IFG nans_column_wise", len(nans_column_wise_ifg))
    nans_voxel_wise_ifg = ifg.calculate_nans_trail_wise(ifg.voxels)
    print("IFG nans_voxel_wise", len(nans_voxel_wise_ifg))
    print("------------")


def visualize_nans():
    config = BrainDataConfig()
    stg = Brain(area=config.STG, data_path=config.STG_path, load_labels=True)
    ifg = Brain(area=config.IFG, data_path=config.IFG_path, load_labels=True)
    data_list = [stg, ifg]
    for data in data_list:
        nans_column_wise = stg.calculate_nans_voxel_wise(data.voxels)
        columns = [i for i in range(data.voxels.shape[1])]
        VisualizeData.plot_bar_graph(
            ("Columns", columns),
            ("nans-length-column-wise", nans_column_wise),
            title=data.area,
        )

        nans_voxel_wise = stg.calculate_nans_trail_wise(data.voxels)
        rows = [i for i in range(data.voxels.shape[0])]
        VisualizeData.plot_bar_graph(
            ("nans-length-voxel-wise", nans_voxel_wise),
            ("rows", rows),
            bar_color="red",
            title=data.area,
        )

    # VisualizeData.plot_data_bar(np.array(x), np.array(nans_column_wise))


def classify_stg(
    folds, test_size, classifiers, strategies, predefined_split, int_labels=False
):
    config = BrainDataConfig()
    stg = Brain(
        area=config.STG,
        data_path=config.STG_path,
        load_labels=True,
        load_int_labels=int_labels,
    )

    data_labels = [stg.subject_labels, stg.image_labels]
    if int_labels:
        data_labels = [stg.subject_labels_int, stg.image_labels_int]

    training = DataTraining()
    export_data = training.classify_brain_data(
        classifiers,
        labels=data_labels,
        data=stg.voxels,
        strategies=strategies,
        predefined_split=predefined_split,
        folds=folds,
        test_size=test_size,
        partially=False,
        dimension_reduction=True,
    )

    split = None
    if predefined_split:
        split = "cr_split"
    else:
        split = "r_split"

    export = ExportData()
    # export.create_and_write_CSV(export_data, "IFG-Results", "IFG")
    # export.create_and_write_datasheet(export_data,f"STG-Results",f"STG-{folds}-Folds-{split}-Clf",transpose=False,)
    export.create_and_write_datasheet(
        export_data,
        f"STG-Results",
        f"STG-{folds}-Folds-{split}-Clf",
        transpose=True,
    )


def classify_ifg(
    folds, test_size, classifiers, strategies, predefined_split, int_labels=False
):
    config = BrainDataConfig()
    ifg = Brain(
        area=config.IFG,
        data_path=config.IFG_path,
        load_labels=True,
        load_int_labels=int_labels,
    )
    data_labels = [ifg.subject_labels, ifg.image_labels]
    if int_labels:
        data_labels = [ifg.subject_labels_int, ifg.image_labels_int]

    training = DataTraining()
    export_data = training.classify_brain_data(
        classifiers,
        labels=data_labels,
        data=ifg.voxels,
        strategies=strategies,
        predefined_split=predefined_split,
        folds=folds,
        test_size=test_size,
    )
    split = None
    if predefined_split:
        split = "cr_split"
    else:
        split = "r_split"
    export = ExportData()
    # export.create_and_write_CSV(export_data, "IFG-Results", "IFG")
    export.create_and_write_datasheet(
        export_data,
        f"IFG-Results",
        f"IFG-{folds}-Folds-{split}-Clf",
        transpose=True,
    )
    export.create_and_write_datasheet(
        export_data,
        f"IFG-Results",
        f"IFG-{folds}-Folds-{split}-Clf",
        transpose=False,
    )


def stg_binary_classification(test_size, classifiers, strategies, predefined_split):
    config = BrainDataConfig()
    stg = Brain(
        area=config.STG,
        data_path=config.STG_path,
        load_labels=True,
        load_int_labels=True,
    )

    stg_subject_binary_data = stg.binary_data(config, stg.subject_labels_int)

    for bd in stg_subject_binary_data:
        training = DataTraining()
        export_data = training.classify_brain_data(
            classifiers,
            labels=[bd.binary_labels],
            data=bd.voxels,
            strategies=strategies,
            predefined_split=predefined_split,
            folds=1,
            test_size=test_size,
            partially=False,
            dimension_reduction=True,
            explain=True,
        )

    stg_image_binary_data = stg.binary_data(config, stg.image_labels_int)

    for bd in stg_image_binary_data:
        training = DataTraining()
        export_data = training.classify_brain_data(
            classifiers,
            labels=[bd.binary_labels],
            data=bd.voxels,
            strategies=strategies,
            predefined_split=predefined_split,
            folds=1,
            test_size=test_size,
            partially=False,
            dimension_reduction=True,
            explain=True,
        )


def main():
    # test_pca()
    # analyse_nans()
    # visualize_nans()
    # classify_iris()

    folds = 1
    test_size = 0.2
    strategies = [
        None,
        "mean",
        "median",
        "most_frequent",
        "constant",
        "remove-voxels",
        "n_neighbors",
    ]
    classifiers = [
        # "XGBoost" # not working, Invalid classes inferred from unique values of `y`.  Expected: [0 1 2], got ['D' 'N' 'S']
        "CatBoost",
        "LGBM",
        "DecisionTree",
        # "HistGradientBoosting",
        "SVM",
        "KNearestNeighbors",
        "GaussianNaiveBayes",
        "LinearDiscriminant",
        "MLP",
        "LogisticRegression",
        "RandomForest",
    ]
    strategies = ["remove-voxels"]
    classifiers = ["SVM"]
    predefined_split = False
    # classify_ifg(folds, test_size, classifiers, strategies, predefined_split, int_labels=False)
    stg_binary_classification(test_size, classifiers, strategies, predefined_split)
    classify_stg(
        folds, test_size, classifiers, strategies, predefined_split, int_labels=True
    )

    predefined_split = True
    # classify_ifg(folds, test_size, classifiers, strategies, predefined_split)
    # classify_stg(folds, test_size, classifiers, strategies, predefined_split)


def test_pca():
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.tree import DecisionTreeClassifier

    # load data
    iris = load_iris()

    # initiate PCA and classifier
    pca = PCA(n_components=2)
    newdata_transformed = pca.transform(iris.data)
    classifier = DecisionTreeClassifier()

    # transform / fit

    X_transformed = pca.fit_transform(iris.data)
    classifier.fit(X_transformed, iris.target)

    # predict "new" data
    # (I'm faking it here by using the original data)

    newdata = iris.data

    # transform new data using already fitted pca
    # (don't re-fit the pca)
    newdata_transformed = pca.transform(newdata)

    # predict labels using the trained classifier
    score = classifier.score(newdata_transformed, iris.target)

    pred_labels = classifier.predict(newdata_transformed)

    print(pred_labels)


if __name__ == "__main__":
    main()
