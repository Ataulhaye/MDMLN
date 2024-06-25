import numpy as np

from Enums import Strategy


class RSAConfig:

    def __init__(
        self,
        radius=7,
        normalize=True,
        strategy=Strategy.mean.name,
        radius_adjustment=1.5,
        # abstract_concrete_RDM == audio
        abstract_concrete_RDM=np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ]
        ),
        related_unrelated_RDM=np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ]
        ),
    ):
        self.radius = radius
        self.radius_adjustment = radius_adjustment
        self.abstract_concrete_RDM = abstract_concrete_RDM
        self.related_unrelated_RDM = related_unrelated_RDM
        self.normalize = normalize
        self.strategy = strategy

    def __repr__(self) -> str:
        return f"Radius:{self.radius},Strategy:{self.strategy} Abstract/Concrete RDM:{self.abstract_concrete_RDM}, Related/Unrelated RDM:{self.related_unrelated_RDM}"
