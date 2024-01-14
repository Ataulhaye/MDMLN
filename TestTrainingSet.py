from numpy import dtype, ndarray
from torch.utils.data import TensorDataset


class TestTrainingSet:
    def __init__(
        self,
        X_train: ndarray = None,
        X_test: ndarray = None,
        y_train: ndarray = None,
        y_test: ndarray = None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __repr__(self) -> str:
        return f"X_train:{self.X_train.shape}, X_test:{self.X_test.shape}, y_train:{self.y_train.shape} , y_test:{self.y_test.shape}"


class TestTrainingTensorDataset:
    def __init__(
        self,
        X_train: TensorDataset = None,
        X_test: TensorDataset = None,
        y_train: TensorDataset = None,
        y_test: TensorDataset = None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __repr__(self) -> str:
        return f"X_train:{self.X_train.tensors[0].shape}, X_test:{self.X_test.tensors[0].shape}, y_train:{self.y_train.tensors[0].shape} , y_test:{self.y_test.tensors[0].shape}"
