from numpy import dtype


class BrainDataLabel:
    def __init__(self, name, popmean, labels: list[dtype]):
        self.name = name
        self.popmean = popmean
        self.labels = labels

    def __repr__(self) -> str:
        return f"Label:{self.name}, Popmean:{self.popmean}, Shape:{self.labels.shape}"
