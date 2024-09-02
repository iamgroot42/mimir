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
        self.prefix = None

    @torch.no_grad()
    def _attack(self, document, probs, tokens = None, **kwargs):        
        recall_dict: dict = kwargs.get("recall_dict", None)

        nonmember_prefix = recall_dict.get("prefix")
        num_shots = recall_dict.get("num_shots")
        avg_length = recall_dict.get("avg_length")

        assert nonmember_prefix, "nonmember_prefix should not be None or empty"

        lls = self.target_model.get_ll(document, probs = probs, tokens = tokens)
        ll_nonmember = self.get_conditional_ll(nonmember_prefix = nonmember_prefix, text = document,
                                                num_shots = num_shots, avg_length = avg_length,
                                                  tokens = tokens)
        recall = ll_nonmember / lls

        assert not np.isnan(recall)
        return recall
    
    def process_prefix(self, prefix, avg_length, total_shots):
        model = self.target_model
        tokenizer = self.target_model.tokenizer

        if self.prefix is not None:
            # We only need to process the prefix once, after that we can just return
            return self.prefix

        max_length = model.max_length
        token_counts = [len(tokenizer.encode(shot)) for shot in prefix]

        target_token_count = avg_length
        total_tokens = sum(token_counts) + target_token_count
        if total_tokens<=max_length:
            self.prefix = prefix
            return self.prefix
        # Determine the maximum number of shots that can fit within the max_length
        max_shots = 0
        cumulative_tokens = target_token_count
        for count in token_counts:
            if cumulative_tokens + count <= max_length:
                max_shots += 1
                cumulative_tokens += count
            else:
                break
        # Truncate the prefix to include only the maximum number of shots
        truncated_prefix = prefix[-max_shots:]
        print(f"""\nToo many shots used. Initial ReCaLL number of shots was {total_shots}. Maximum number of shots is {max_shots}. Defaulting to maximum number of shots.""")
        self.prefix = truncated_prefix
        return self.prefix
    
    def get_conditional_ll(self, nonmember_prefix, text, num_shots, avg_length, tokens=None):
        assert nonmember_prefix, "nonmember_prefix should not be None or empty"

        model = self.target_model
        tokenizer = self.target_model.tokenizer

        if tokens is None:
            target_encodings = tokenizer(text=text, return_tensors="pt")
        else:
            target_encodings = tokens

        processed_prefix = self.process_prefix(nonmember_prefix, avg_length, total_shots=num_shots)
        input_encodings = tokenizer(text="".join(processed_prefix), return_tensors="pt")

        prefix_ids = input_encodings.input_ids.to(model.device)
        text_ids = target_encodings.input_ids.to(model.device)

        max_length = model.max_length
        total_length = prefix_ids.size(1) + text_ids.size(1)

        if prefix_ids.size(1) >= max_length:
            raise ValueError("Prefix length exceeds or equals the model's maximum context window.")
        stride = model.stride

        labels = torch.cat((prefix_ids, text_ids), dim=1)

        total_loss = 0
        with torch.no_grad():
            for i in range(0, labels.size(1), stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, labels.size(1))
                trg_len = end_loc - i
                input_ids = labels[:, begin_loc:end_loc].to(model.device)
                target_ids = input_ids.clone()
                
                target_ids[:, :max(0, prefix_ids.size(1) - begin_loc)] = -100
                target_ids[:, :-trg_len] = -100

                if torch.all(target_ids == -100):
                    continue #this prevents the model from outputting nan as loss value
                
                outputs = model.model(input_ids, labels=target_ids)
                loss = outputs.loss
                total_loss += -loss.item()

        return total_loss

    

