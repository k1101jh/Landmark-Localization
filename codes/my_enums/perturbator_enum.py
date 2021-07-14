from enum import Enum


class PerturbatorEnum(Enum):
    NO_PERTURBATION = 0
    BLACKOUT = 1
    WHITEOUT = 2
    SMOOTHING = 3
    BINARIZATION = 4
    EDGE_DETECTION = 5
    HYBRID = 6
    CUSTOM = 7
