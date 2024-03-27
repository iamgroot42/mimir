"""
    zlib-normalization Attack: https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
"""

import torch as ch
import zlib

from mimir.attacks.all_attacks import Attack
from mimir.models import Model
from mimir.config import ExperimentConfig


class ZLIBAttack(Attack):

    def __init__(self,
                 config: ExperimentConfig,
                 model: Model):
        super().__init__(config, model, ref_model=None)

    @ch.no_grad()
    def _attack(
        self,
        document,
        probs,
        tokens=None,
        **kwargs
    ):
        """
        zlib-based attack score. Performs difficulty calibration in model likelihood by normalizing with zlib entropy.
        """
        loss = kwargs.get("loss", None)
        if loss is None:
            loss = self.target_model.get_ll(document, probs=probs, tokens=tokens)
        zlib_entropy = len(zlib.compress(bytes(document, "utf-8")))
        return loss / zlib_entropy
