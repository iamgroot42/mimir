"""
    Reference-based attacks.
"""
from mimir.attacks.blackbox_attacks import Attack


class ReferenceAttack(Attack):
    def __init__(self, config, model, reference_model):
        super().__init__(config, model, reference_model)

    def load(self):
        self.ref_model.load()

    def _attack(self, document, probs, tokens=None, **kwargs):
        loss = kwargs.get('loss', None)
        if loss is None:
            loss = self.model.get_ll(document, probs=probs, tokens=tokens)
        ref_loss = self.ref_model.get_ll(document, probs=probs, tokens=tokens)
        return ref_loss - loss
