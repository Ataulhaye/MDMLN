import torch


class Codings:
    def __init__(self, encoding: torch.Tensor, decoding: torch.Tensor):
        self.encoding = encoding
        self.decoding = decoding

    def __repr__(self) -> str:
        return f"Encoding:{self.encoding.shape}, Decoding:{self.decoding.shape}"
