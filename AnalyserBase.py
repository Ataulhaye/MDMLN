from Brain import Brain
from BrainDataConfig import BrainDataConfig
from TrainingConfig import TrainingConfig


class AnalyserBase:

    def __init__(
        self,
        brain: Brain = None,
        classifiers=None,
        strategies=None,
        training_config=None,
        data_config=None,
    ):
        if classifiers is None:
            self.classifiers = ["SVM", "MLP", "LinearDiscriminant", "LGBM"]
        else:
            self.classifiers = classifiers

        if strategies is None:
            self.strategies = ["mean", "remove-voxels", "median"]
        else:
            self.strategies = strategies

        if training_config is None:
            self.training_config = TrainingConfig()
        else:
            self.training_config = training_config

        if data_config is None:
            self.data_config = BrainDataConfig()
        else:
            self.data_config = data_config

        self.brain = brain
