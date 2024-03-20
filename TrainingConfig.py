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
        use_autoencoder=False,
        brain_area="",
        tsne=False,
        best_autoencoder_config_STG={
            "input_dim": 7238,
            "hidden_dim1": 32,
            "hidden_dim2": 512,
            "embedding_dim": 2,
            "lr": 0.07539378759292441,
            "batch_size": 128,
            "epochs": 10,
            "brain_area": "STG",
        },
        # ----------------------------------------
        # -------------IFG-----------------------
        # Best trial config: {'input_dim': 523, 'hidden_dim1': 1024, 'hidden_dim2': 8, 'embedding_dim': 2, 'lr': 0.0016028928095361706, 'batch_size': 384, 'epochs': 25}
        # Best trial final training loss: 0.01012803427875042
        # Best trial epoch: 24
        # Best model path C:/Users/ataul/ray_results/tune_with_parameters_2024-03-16_17-40-53/tune_with_parameters_f4bd1_00011_11_embedding_dim=2,hidden_dim1=1024,hidden_dim2=8,lr=0.0016_2024-03-16_18-09-11
        # ----------------------------------------
        best_autoencoder_config_IFG={
            "input_dim": 523,
            "hidden_dim1": 1024,
            "hidden_dim2": 8,
            "embedding_dim": 2,
            "lr": 0.0016028928095361706,
            "batch_size": 384,
            "epochs": 25,
            "brain_area": "IFG",
        },
        best_autoencoder_config={
            "input_dim": 523,
            "hidden_dim1": 8,
            "hidden_dim2": 1,
            "embedding_dim": 4,
            "lr": 0.023394330747395223,
            "batch_size": 128,
            "epochs": 10,
            "brain_area": "",
        },
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
        self.use_autoencoder = use_autoencoder
        self.best_autoencoder_config = best_autoencoder_config
        self.brain_area = brain_area
        self.tsne = tsne

    def __repr__(self) -> str:
        return f"Folds:{self.folds}, Test size:{self.test_size}, Conditional split:{self.predefined_split} , Dimension reduction:{self.dimension_reduction}, use_autoencoder:{self.use_autoencoder}, brain_area:{self.brain_area}"
