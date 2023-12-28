"""
    Enum class for attacks. Also contains the base attack class.
"""

from enum import Enum
from mimir.models import Model


# Attack definitions
class BlackBoxAttacks(str, Enum):
    LOSS = "loss"
    REFERENCE_BASED = "ref"
    ZLIB = "zlib"
    MIN_K = "min_k"
    NEIGHBOR = "ne"
    QUANTILE = "quantile"


# TODO: Move attacks in models into this file as functions
# TODO Use decorators to link attack implementations with enum above

# Base attack class
class Attack:
    def __init__(self, config, target_model: Model, ref_model: Model = None):
        self.config = config
        self.target_model = target_model
        self.ref_model = ref_model

    def prepare(self, **kwargs):
        """
        Any attack-specific steps (one-time) preparation
        """
        pass

    def attack(self, document, **kwargs):
        """
        Score a document using the attack's scoring function
        """
        raise NotImplementedError("Attack must implement attack()")
