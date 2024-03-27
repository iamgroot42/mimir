"""
    Gradient-norm attack. Proposed for MIA in multiple settings, and particularly experimented for pre-training data and LLMs in https://arxiv.org/abs/2402.17012
"""

import torch as ch
import numpy as np
from mimir.attacks.all_attacks import Attack
from mimir.models import Model
from mimir.config import ExperimentConfig


class GradNormAttack(Attack):
    def __init__(self, config: ExperimentConfig, model: Model):
        super().__init__(config, model, ref_model=None, is_blackbox=False)

    def _attack(self, document, probs, tokens=None, **kwargs):
        """
        Gradient Norm Attack. Computes p-norm of gradients w.r.t. input tokens.
        """
        # We ignore probs here since they are computed in the general case without gradient-tracking (to save memory)

        # Hyper-params specific to min-k attack
        p: float = kwargs.get("p", np.inf)
        if p not in [1, 2, np.inf]:
            raise ValueError(f"Invalid p-norm value: {p}.")

        # Make sure model params require gradients
        # for name, param in self.target_model.model.named_parameters():
        #    param.requires_grad = True

        # Get gradients for model parameters
        self.target_model.model.zero_grad()
        all_prob = self.target_model.get_probabilities(document, tokens=tokens, no_grads=False)
        loss = - ch.mean(all_prob)
        loss.backward()

        # Compute p-norm of gradients (for all model params where grad exists)
        grad_norms = []
        for param in self.target_model.model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.detach().norm(p))
        grad_norm = ch.stack(grad_norms).mean()

        # Zero out gradients again
        self.target_model.model.zero_grad()

        return -grad_norm.cpu().numpy()
