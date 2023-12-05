"""
    Base class for all black-box attack implementations.
"""
from mimir.models import Model


class Attack:
    def __init__(self, config, target_model: Model, ref_model: Model = None, **kwargs):
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
