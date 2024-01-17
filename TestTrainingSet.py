from numpy import ndarray
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
        train_set: TensorDataset = None,
        test_set: TensorDataset = None,
        val_set: TensorDataset = None,
    ):
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set

    def __repr__(self) -> str:
        return f"X_train:{self.train_set.tensors[0].shape}, X_test:{self.test_set.tensors[0].shape}, X_val:{self.val_set.tensors[0].shape}, y_train:{self.train_set.tensors[1].shape} , y_test:{self.train_set.tensors[1].shape}, y_val:{self.val_set.tensors[1].shape}"
