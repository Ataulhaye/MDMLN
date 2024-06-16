import numpy as np


class RSAConfig:

    def __init__(
        self,
        radius=7,
        normalize=False,
        # abstract_concrete_RDM == audio
        audio_RDM=np.array(
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
        self.audio_RDM = audio_RDM
        self.related_unrelated_RDM = related_unrelated_RDM
        self.normalize = normalize

    def __repr__(self) -> str:
        return f"Radius:{self.radius}, Audio RDM:{self.audio_RDM}, Related/Unrelated RDM:{self.related_unrelated_RDM}"
