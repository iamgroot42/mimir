"""
    Min-k % Prob Attack: https://arxiv.org/pdf/2310.16789.pdf
"""
import torch as ch
import numpy as np
from mimir.attacks.all_attacks import Attack
from mimir.models import Model
from mimir.config import ExperimentConfig


class MinKProbAttack(Attack):

    def __init__(self, config: ExperimentConfig, model: Model):
        super().__init__(config, model, ref_model=None)

    @ch.no_grad()
    def _attack(self, document, probs, tokens=None, **kwargs):
        """
        Min-k % Prob Attack. Gets model probabilities and returns likelihood when computed over top k% of ngrams.
        """
        # Hyper-params specific to min-k attack
        k: float = kwargs.get("k", 0.2)
        window: int = kwargs.get("window", 1)
        stride: int = kwargs.get("stride", 1)

        all_prob = (
            probs
            if probs is not None
            else self.target_model.get_probabilities(document, tokens=tokens)
        )
        # iterate through probabilities by ngram defined by window size at given stride
        ngram_probs = []
        for i in range(0, len(all_prob) - window + 1, stride):
            ngram_prob = all_prob[i : i + window]
            ngram_probs.append(np.mean(ngram_prob))
        min_k_probs = sorted(ngram_probs)[: int(len(ngram_probs) * k)]

        return -np.mean(min_k_probs)
