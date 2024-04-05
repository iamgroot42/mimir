"""
    Min-K%++ Attack: https://github.com/zjysteven/mink-plus-plus
"""
import torch as ch
import numpy as np
from mimir.attacks.all_attacks import Attack
from mimir.models import Model
from mimir.config import ExperimentConfig


class MinKPlusPlusAttack(Attack):

    def __init__(self, config: ExperimentConfig, model: Model):
        super().__init__(config, model, ref_model=None)

    @ch.no_grad()
    def _attack(self, document, probs, tokens=None, **kwargs):
        """
        Min-K%++ Attack. 
        Gets token probabilties, normalize with the mean and std over the whole categorical distribution,
        and returns normalized likelihood when computed over top k% of ngrams.
        """
        # Hyper-params specific to min-k attack
        k: float = kwargs.get("k", 0.2)
        all_probs = kwargs.get("all_probs", None)

        # these are all log probabilites
        target_prob, all_probs = (
            (probs, all_probs)
            if (probs is not None and all_probs is not None)
            else self.model.get_probabilities(document, tokens=tokens, return_all_probs=True)
        )
        
        mu = (ch.exp(all_probs) * all_probs).sum(-1)
        sigma = (ch.exp(all_probs) * ch.square(all_probs)).sum(-1) - ch.square(mu)
        scores = (np.array(target_prob) - mu.numpy()) / sigma.sqrt().numpy()
        
        return -np.mean(sorted(scores)[:int(len(scores) * k)])