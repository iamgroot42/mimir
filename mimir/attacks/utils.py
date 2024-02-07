from mimir.attacks.blackbox_attacks import BlackBoxAttacks

from mimir.attacks.loss import LOSSAttack
from mimir.attacks.reference import ReferenceAttack
from mimir.attacks.zlib import ZLIBAttack
from mimir.attacks.min_k import MinKProbAttack
from mimir.attacks.neighborhood import NeighborhoodAttack


# TODO Use decorators to link attack implementations with enum above
def get_attacker(attack: str):
    mapping = {
        BlackBoxAttacks.LOSS: LOSSAttack,
        BlackBoxAttacks.REFERENCE_BASED: ReferenceAttack,
        BlackBoxAttacks.ZLIB: ZLIBAttack,
        BlackBoxAttacks.MIN_K: MinKProbAttack,
        BlackBoxAttacks.NEIGHBOR: NeighborhoodAttack,
    }
    attack_cls = mapping.get(attack, None)
    if attack_cls is None:
        raise ValueError(f"Attack {attack} not found")
    return attack_cls
