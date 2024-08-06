"""
    ReCaLL Attack: https://github.com/ruoyuxie/recall/
"""
import torch 
import numpy as np
from mimir.attacks.all_attacks import Attack
from mimir.models import Model
from mimir.config import ExperimentConfig

class ReCaLLAttack(Attack):

    def __init__(self, config: ExperimentConfig, target_model: Model):
        super().__init__(config, target_model, ref_model = None)

    @torch.no_grad()
    def _attack(self, document, probs, tokens = None, **kwargs):
        # TODO implement ReCaLL Attack
        raise NotImplementedError("Need to do")
