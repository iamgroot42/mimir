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
        nonmember_prefix = kwargs.get("nonmember_prefix", None)
        assert nonmember_prefix, "nonmember_prefix should not be None or empty"

        lls = self.target_model.get_ll(document, probs = probs, tokens = tokens)
        ll_nonmember = self.get_conditional_ll(nonmember_prefix = nonmember_prefix, text = document,
                                                model = self.target_model, tokenizer=self.target_model.tokenizer, tokens = tokens)
        recall = ll_nonmember / lls

        return recall
    
    def get_conditional_ll(self, nonmember_prefix, text, model, tokenizer, tokens = None):
        assert nonmember_prefix, "nonmember_prefix should not be None or empty"
        
        input_encodings = tokenizer(text = nonmember_prefix, return_tensors="pt")
        if tokens is None:
            target_encodings = tokenizer(text = text, return_tensors="pt")
        else:
            target_encodings = tokens

        max_length = model.max_length
        input_ids = input_encodings.input_ids.to(model.device)
        target_ids = target_encodings.input_ids.to(model.device)
        
        total_length = input_ids.size(1) + target_ids.size(1)
        
        if total_length > max_length:
            excess_length = total_length - max_length
            target_ids = target_ids[:, :-excess_length] 
        concat_ids = torch.cat((input_ids, target_ids), dim=1)
        labels = concat_ids.clone()
        labels[:, :input_ids.size(1)] = -100

        with torch.no_grad():
            outputs = model.model(concat_ids, labels=labels)
        
        loss, logits = outputs[:2]
        ll = -loss.item()
        return ll
    

