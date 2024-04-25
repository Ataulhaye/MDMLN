from numpy import dtype, ndarray


class BrainDataLabel:
    def __init__(self, name, popmean, labels: ndarray, type=None):
        self.name = name
        self.popmean = popmean
        self.labels = labels
        if type is None:
            self.type = self.name
        else:
            self.type = type

    def __repr__(self) -> str:
        return f"Label:{self.name}, Popmean:{self.popmean}, Shape:{self.labels.shape}, Type:{self.type}"
