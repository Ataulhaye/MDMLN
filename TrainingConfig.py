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
        analyze_concatenated_trails=False,
        analyze_binary_trails=False,
        lobe="",
        tsne=False,
        has_fix_components=(False, 75),
        # ----------------------------------------
        # Best trial config: {'input_dim': 7238, 'hidden_dim1': 1024, 'hidden_dim2': 4, 'embedding_dim': 16, 'lr': 0.0007659874628602165, 'batch_size': 512, 'epochs': 30}
        # Best trial final training loss: 0.021615224579970043
        # Best trial epoch: 29
        # Best model path C:/Users/ataul/ray_results/tune_with_parameters_2024-08-04_15-04-55/tune_with_parameters_255d5_00012_12_batch_size=512,embedding_dim=16,hidden_dim1=1024,hidden_dim2=4,lr=0.0008_2024-08-04_15-50-28
        # ----------------------------------------
        optimal_autoencoder_config_STG={
            "input_dim": 7238,
            "hidden_dim1": 1024,
            "hidden_dim2": 4,
            "embedding_dim": 16,
            "lr": 0.0007659874628602165,
            "batch_size": 512,
            "epochs": 29,
            "lobe": "STG",
        },
        # -------------IFG-----------------------
        # ----------------------------------------
        # Best trial config: {'input_dim': 523, 'hidden_dim1': 32, 'hidden_dim2': 4, 'embedding_dim': 16, 'lr': 0.007467380652375706, 'batch_size': 512, 'epochs': 30}
        # Best trial final training loss: 0.01083484540383021
        # Best trial epoch: 29
        # Best model path C:/Users/ataul/ray_results/tune_with_parameters_2024-08-04_17-00-53/tune_with_parameters_5888b_00006_6_batch_size=512,embedding_dim=16,hidden_dim1=32,hidden_dim2=4,lr=0.0075_2024-08-04_18-21-40
        # ----------------------------------------
        optimal_autoencoder_config_IFG={
            "input_dim": 523,
            "hidden_dim1": 32,
            "hidden_dim2": 4,
            "embedding_dim": 16,
            "lr": 0.007467380652375706,
            "batch_size": 512,
            "epochs": 29,
            "lobe": "IFG",
        },
        optimal_autoencoder_config={},
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
        self.optimal_autoencoder_config = optimal_autoencoder_config
        self.optimal_autoencoder_config_STG = optimal_autoencoder_config_STG
        self.optimal_autoencoder_config_IFG = optimal_autoencoder_config_IFG
        self.lobe = lobe
        self.tsne = tsne
        self.has_fix_components = has_fix_components
        # self.pca_fix_components = pca_fix_components
        self.analyze_concatenated_trails = analyze_concatenated_trails
        self.analyze_binary_trails = analyze_binary_trails

    def __repr__(self) -> str:
        return f"Folds:{self.folds}, Test size:{self.test_size}, Conditional split:{self.predefined_split} , Dimension reduction:{self.dimension_reduction}, use_autoencoder:{self.use_autoencoder}, lobe:{self.lobe}, concatenated_trails:{self.analyze_concatenated_trails}, binary_trails:{self.analyze_binary_trails}"
