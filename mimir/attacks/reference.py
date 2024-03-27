"""
    Reference-based attacks.
"""
from mimir.attacks.all_attacks import Attack
from mimir.models import Model, ReferenceModel
from mimir.config import ExperimentConfig


class ReferenceAttack(Attack):

    def __init__(
        self, config: ExperimentConfig,
        model: Model,
        reference_model: ReferenceModel
    ):
        super().__init__(config, model, reference_model)

    def _attack(self, document, probs, tokens=None, **kwargs):
        """
        Reference-based attack score. Performs difficulty calibration in model likelihood using a reference model.
        """
        loss = kwargs.get('loss', None)
        if loss is None:
            loss = self.target_model.get_ll(document, probs=probs, tokens=tokens)
        ref_loss = self.ref_model.get_ll(document, probs=probs, tokens=tokens)
        return loss - ref_loss
