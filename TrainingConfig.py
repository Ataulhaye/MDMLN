class TrainingConfig:
    """Provides the configuration for training and testing of brain data
    Args:
        classifier (str, optional): _description_. Defaults to "SVM" SVC kernal is linear.
        if 'KNearestNeighbors' then KNeighborsClassifier
        if 'DecisionTree' then DecisionTreeClassifier
        if 'LinearDiscriminant' then LinearDiscriminantAnalysis
        if 'GaussianNaiveBayes' then GaussianNB
        folds (int, optional): _description_. Defaults to 5.
        test_size (float, optional): size of test data. Defaults to 0.3.
        strategy: used for data normalization
        predefined_split: if True the split will be according to the BrainDataConfig conditions
        explaination=False, do the KernalExplanation and draw graphs
    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self,
        strategy: str = None,
        predefined_split: bool = True,
        classifier: str = "SVM",
        folds: int = 5,
        test_size: float = 0.2,
        partially: bool = False,
        dimension_reduction: bool = False,
        explain: bool = False,
        nan_classifiers=["DecisionTree", "HistGradientBoosting", "LGBM", "CatBoost"],
    ):
        self.strategy = strategy
        self.classifier = classifier
        self.predefined_split = predefined_split
        self.folds = folds
        self.test_size = test_size
        self.partially = partially
        self.dimension_reduction = dimension_reduction
        self.explain = explain
        self.nan_classifiers = nan_classifiers

    def __repr__(self) -> str:
        return f"folds:{self.folds}, test_size:{self.test_size}, predefined_split:{self.predefined_split} , dimension_reduction:{self.dimension_reduction}"
