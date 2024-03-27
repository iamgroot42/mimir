"""
    Test attack implementations. Consists of basic execution tests to make sure attack works as expected and returns values as expected.
"""
import torch
import pytest
import numpy as np
import torch.nn as nn

from mimir.attacks.all_attacks import AllAttacks
from mimir.attacks.utils import get_attacker


class TestAttack:
    def test_attacks_exist(self):
        """
            Check if all known attacks can be loaded.
        """
        # Enumerate all "available" attacks and make sure they are available
        for attack in AllAttacks:
            attacker = get_attacker(attack)
            assert attacker is not None, f"Attack {attack} not found"
            # TODO: Use a 'testing' config and model to check if the attack can be loaded
            # attacker_obj = attacker(None, None)

    def test_attack_shape(self):
        # Check 1 - attack accepts inputs in given shape, and works for both text-based and tokenized inputs
        pass

    def test_attack_scores_shape(self):
        # Check 2 - scores returned match exepected shape
        pass

    def test_attack_score_range(self):
        # Check 3 - scores match expected range
        pass

    def test_attack_auc(self):
        # Check 4 (TODO) - Attack AUC is not horribly bad
        pass
