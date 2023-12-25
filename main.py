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
from TrainingConfig import TrainingConfig


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


def ifg_classification(classifiers, strategies, t_config: TrainingConfig):
    config = BrainDataConfig()
    brain = Brain(
        area=config.IFG,
        data_path=config.IFG_path,
        load_labels=True,
        load_int_labels=True,
    )

    training = DataTraining()

    brain.current_labels = brain.subject_labels_int
    export_data = training.brain_data_classification(
        brain, t_config, strategies, classifiers
    )

    brain.current_labels = brain.image_labels_int
    e_data = training.brain_data_classification(
        brain, t_config, strategies, classifiers
    )
    export_data.extend(e_data)

    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=export_data,
        sheet_name=f"{brain.area}-Results",
        title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
    )


def stg_binary_classification(classifiers, strategies, t_config: TrainingConfig):
    config = BrainDataConfig()
    brain = Brain(
        area=config.STG,
        data_path=config.STG_path,
        load_labels=True,
        load_int_labels=True,
    )

    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    stg_subject_binary_data = brain.binary_data(config, brain.subject_labels_int)

    t_config.dimension_reduction = True
    t_config.explain = True
    t_config.folds = 1
    t_config.predefined_split = False

    for bd in stg_subject_binary_data:
        training = DataTraining()
        export_data = training.brain_data_classification(
            bd,
            t_config,
            strategies,
            classifiers,
        )
        export = ExportData()
        note = export.create_note(t_config)
        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=f"{brain.area}-Results",
            title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
        )

    stg_image_binary_data = brain.binary_data(config, brain.image_labels_int)

    for bd in stg_image_binary_data:
        training = DataTraining()
        export_data = training.brain_data_classification(
            bd,
            t_config,
            strategies,
            classifiers,
        )
        export = ExportData()
        note = export.create_note(t_config)
        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=f"{brain.area}-Results",
            title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
        )


def stg_classification(classifiers, strategies, t_config: TrainingConfig):
    config = BrainDataConfig()
    brain = Brain(
        area=config.STG,
        data_path=config.STG_path,
        load_labels=True,
        load_int_labels=True,
    )

    brain.current_labels = brain.subject_labels_int

    training = DataTraining()

    export_data = training.brain_data_classification(
        brain, t_config, strategies, classifiers
    )

    brain.current_labels = brain.image_labels_int
    e_data = training.brain_data_classification(
        brain, t_config, strategies, classifiers
    )
    export_data.extend(e_data)

    split = "r_split"
    if t_config.predefined_split:
        split = "cr_split"

    export = ExportData()
    note = export.create_note(t_config)
    export.create_and_write_datasheet(
        data=export_data,
        sheet_name=f"{brain.area}-Results",
        title=f"{brain.area}-{t_config.folds}-Folds-{split}-Clf",
        notes=note,
        transpose=True,
    )


def main():
    # analyse_nans()
    # visualize_nans()
    # classify_iris()
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
    strategies = ["mean", "remove-voxels", "median"]
    strategies = ["mean"]
    classifiers = ["SVM", "MLP", "LinearDiscriminant"]
    t_config = TrainingConfig()
    # t_config.folds = 1
    # t_config.explain = True
    # t_config.dimension_reduction = True
    t_config.predefined_split = True

    # stg_binary_classification(classifiers, strategies, t_config)
    stg_classification(classifiers, strategies, t_config)
    ifg_classification(classifiers, strategies, t_config)

    t_config.predefined_split = False

    stg_classification(classifiers, strategies, t_config)
    ifg_classification(classifiers, strategies, t_config)


if __name__ == "__main__":
    main()
