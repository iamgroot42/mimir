from mimir.attacks.all_attacks import AllAttacks

from mimir.attacks.loss import LOSSAttack
from mimir.attacks.reference import ReferenceAttack
from mimir.attacks.zlib import ZLIBAttack
from mimir.attacks.min_k import MinKProbAttack
from mimir.attacks.min_k_plus_plus import MinKPlusPlusAttack
from mimir.attacks.neighborhood import NeighborhoodAttack
from mimir.attacks.gradnorm import GradNormAttack
from mimir.attacks.recall import ReCaLLAttack


# TODO Use decorators to link attack implementations with enum above
def get_attacker(attack: str):
    mapping = {
        AllAttacks.LOSS: LOSSAttack,
        AllAttacks.REFERENCE_BASED: ReferenceAttack,
        AllAttacks.ZLIB: ZLIBAttack,
        AllAttacks.MIN_K: MinKProbAttack,
        AllAttacks.MIN_K_PLUS_PLUS: MinKPlusPlusAttack,
        AllAttacks.NEIGHBOR: NeighborhoodAttack,
        AllAttacks.GRADNORM: GradNormAttack,
        AllAttacks.RECALL: ReCaLLAttack
    }
    attack_cls = mapping.get(attack, None)
    if attack_cls is None:
        raise ValueError(f"Attack {attack} not found")
    return attack_cls
