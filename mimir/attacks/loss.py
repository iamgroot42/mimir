"""
    Straight-forward LOSS attack
"""
from mimir.attacks.blackbox_attacks import Attack


class LOSSAttack(Attack):
    def __init__(self, config, model):
        super().__init__(config, model, ref_model=None)

    def _attack(self, document, probs, tokens=None, **kwargs):
        return self.model.get_ll(document, probs=probs, tokens=tokens)
