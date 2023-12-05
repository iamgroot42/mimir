"""
    Reference-based attacks.
"""
from mimir.attacks.base import Attack


class ReferenceAttack(Attack):
    def __init__(self, config, model, reference_model):
        super().__init__(config, model, reference_model)
    
    def prepare(self, **kwargs):
        self.reference_model.load()
    
    def attack(self, document, **kwargs):
        # TODO: Implement
        pass

