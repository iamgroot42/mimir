"""
    Straight-forward LOSS attack
"""
from mimir.attacks.blackbox_attacks import Attack


class LOSSAttack(Attack):
    def __init__(self, config, model):
        super().__init__(config, model, ref_model=None)

    def attack(self, document, **kwargs):
        """
        Score a document using the attack's scoring function
        """
        substr = kwargs.get("substr", None)
        if substr is None:
            raise ValueError("substr (tokenized version of document) must be provided")

        if self.config.pretokenized:
            detokenized_sample = kwargs.get("detokenized_sample", None)
            if detokenized_sample is None:
                raise ValueError("detokenized_sample must be provided when self.config.pretokenized is True")

            s_tk_probs = self.model.get_probabilities(detokenized_sample, tokens=substr)
            loss = self.get_ll(detokenized_sample, tokens=substr, probs=s_tk_probs)
        else:
            s_tk_probs = self.model.get_probabilities(substr)
            loss = self.get_ll(document, probs=s_tk_probs)
        return loss
