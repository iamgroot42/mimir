from enum import Enum

# Attack definitions
class BlackBoxAttacks(str, Enum):
    LOSS = "loss"
    REFERENCE_BASED = "ref"
    ZLIB = "zlib"
    MIN_K = "min_k"
    NEIGHBOR = "ne"

# TODO: Move attacks in models into this file as functions