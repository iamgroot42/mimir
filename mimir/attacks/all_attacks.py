"""
    Enum class for attacks. Also contains the base attack class.
"""

from enum import Enum
from mimir.models import Model


# Attack definitions
class AllAttacks(str, Enum):
    LOSS = "loss" # Done
    REFERENCE_BASED = "ref" # Done
    ZLIB = "zlib" # Done
    MIN_K = "min_k" # Done
    NEIGHBOR = "ne" # Done
    GRADNORM = "gradnorm" # Done
    # QUANTILE = "quantile" # Uncomment when tested implementation is available


# Base attack class
class Attack:
    def __init__(self, config, target_model: Model, ref_model: Model = None, is_blackbox: bool = True):
        self.config = config
        self.target_model = target_model
        self.ref_model = ref_model
        self.is_loaded = False
        self.is_blackbox = is_blackbox

    def load(self):
        """
        Any attack-specific steps (one-time) preparation
        """
        if self.ref_model is not None:
            self.ref_model.load()
            self.is_loaded = True

    def unload(self):
        if self.ref_model is not None:
            self.ref_model.unload()
            self.is_loaded = False

    def _attack(self, document, probs, tokens=None, **kwargs):
        """
        Actual logic for attack. 
        """
        raise NotImplementedError("Attack must implement attack()")

    def attack(self, document, probs, **kwargs):
        """
        Score a document using the attack's scoring function. Calls self._attack
        """
        # Load attack if not loaded yet
        if not self.is_loaded:
            self.load()
            self.is_loaded = True

        detokenized_sample = kwargs.get("detokenized_sample", None)
        if self.config.pretokenized and detokenized_sample is None:
            raise ValueError("detokenized_sample must be provided")

        score = (
            self._attack(document, probs=probs, **kwargs)
            if not self.config.pretokenized
            else self._attack(
                detokenized_sample, tokens=document, probs=probs, **kwargs
            )
        )

        return score
