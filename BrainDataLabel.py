from numpy import dtype, ndarray


class BrainDataLabel:
    def __init__(self, name, popmean, labels: ndarray):
        self.name = name
        self.popmean = popmean
        self.labels = labels

    def __repr__(self) -> str:
        return f"Label:{self.name}, Popmean:{self.popmean}, Shape:{self.labels.shape}"
