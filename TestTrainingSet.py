from numpy import dtype, ndarray


class TestTrainingSet:
    def __init__(
        self,
        X_train: ndarray,
        X_test: ndarray,
        y_train: ndarray,
        y_test: ndarray,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __repr__(self) -> str:
        return f"X_train:{self.X_train.shape}, X_test:{self.X_test.shap}, y_train:{self.y_train.shape} , y_test:{self.y_test.shape}"
