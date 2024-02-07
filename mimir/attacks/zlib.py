"""
    zlib-normalization Attack: https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
"""

import torch as ch
import zlib

from mimir.attacks.blackbox_attacks import Attack


class ZLIBAttack(Attack):
    def __init__(self, config, model):
        super().__init__(config, model, ref_model=None)

    @ch.no_grad()
    def _attack(
        self,
        document,
        probs,
        tokens=None,
        **kwargs
    ):
        loss = kwargs.get("loss", None)
        if loss is None:
            loss = self.model.get_ll(document, probs=probs, tokens=tokens)
        zlib_entropy = len(zlib.compress(bytes(document, "utf-8")))
        return loss / zlib_entropy
