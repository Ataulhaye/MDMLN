from enum import Enum


class Lobe(Enum):
    STG = 1
    IFG = 2
    ALL = 3
    Other = 4


class LabelType(Enum):
    Subject = 1
    Image = 2
    Subject_int = 3
    Image_int = 4


class Strategy(Enum):
    mean = 1
    median = 2
    most_frequent = 3
    constant = 4
    remove_voxels = 5
    n_neighbors = 6
    mice = 7
