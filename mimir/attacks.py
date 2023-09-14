"""
    Main attack implementations
"""
from heapq import nlargest
import torch
import re
import numpy as np
from tqdm import tqdm
import random
import transformers
from typing import List

from mimir.config import ExperimentConfig
from mimir.attack_utils import count_masks, apply_extracted_fills
from mimir.models import Model


class MaskFillingModel(Model):
    def __init__(self, config: ExperimentConfig, **kwargs):
        super(MaskFillingModel, self).__init__(config, **kwargs)
        self.device = self.config.env_config.device_aux
        self.name = self.config.neighborhood_config.model

    def generate_neighbors(self, texts, **kwargs) -> List[str]:
        raise NotImplementedError("generate_neighbors not implemented")


class T5Model(MaskFillingModel):
    def __init__(self, config: ExperimentConfig, **kwargs):
        super().__init__(config, **kwargs)
        model_kwargs = self.kwargs.get('model_kwargs', {})
        tokenizer_kwargs = self.kwargs.get('tokenizer_kwargs', {})

        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self.name, **model_kwargs, cache_dir=self.cache_dir)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.name, **tokenizer_kwargs, cache_dir=self.cache_dir)

        # define regex to match all <extra_id_*> tokens, where * is an integer
        self.pattern = re.compile(r"<extra_id_\d+>")

    def create_fill_dictionary(self, data):
        self.FILL_DICTIONARY = set()
        for texts in data.values():
            for text in texts:
                self.FILL_DICTIONARY.update(text.split())
        self.FILL_DICTIONARY = sorted(list(self.FILL_DICTIONARY))

    def tokenize_and_mask(self, text: str, span_length: int, pct: float, ceil_pct: bool = False):
        buffer_size = self.config.neighborhood_config.buffer_size

        tokens = text.split(' ')
        mask_string = '<<<mask>>>'

        span_length = min(int(pct*len(tokens)), span_length)
        # avoid div zero:

        span_length = max(1, span_length)

        n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, max(1, len(tokens) - span_length))
            end = start + span_length
            search_start = max(0, start - buffer_size)
            search_end = min(len(tokens), end + buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1

        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text
    
    def extract_fills(self, texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [self.pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills

    def replace_masks(self, texts: List[str]):
        """
            Replace each masked span with a sample from T5 mask_model
        """
        mask_top_p = self.config.neighborhood_config.top_p
        n_expected = count_masks(texts)
        stop_id = self.tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**tokens, max_length=150, do_sample=True, top_p=mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def generate_neighbors_(self, texts: List[str], **kwargs):
        span_length: int = kwargs.get('span_length')
        pct: float = kwargs.get('pct')
        ceil_pct: bool = kwargs.get('ceil_pct', False)
        base_tokenizer = kwargs.get('base_tokenizer', None)
        neigh_config = self.config.neighborhood_config

        if not neigh_config.random_fills:
            masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']

            # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
            attempts = 1
            break_out_of_loop: bool = False
            while '' in perturbed_texts:
                if attempts > neigh_config.max_tries:
                    for idx in idxs :
                        perturbed_texts[idx] = texts[idx]
                    break_out_of_loop = True
                    break
                if break_out_of_loop:
                    break
                idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
                print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
                masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
                raw_fills = self.replace_masks(masked_texts)
                extracted_fills = self.extract_fills(raw_fills)
                new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
                for idx, x in zip(idxs, new_perturbed_texts):
                    perturbed_texts[idx] = x
                attempts += 1
        else:
            if neigh_config.random_fills_tokens:
                if base_tokenizer is None:
                    raise ValueError("base_tokenizer must be provided if random_fills and random_fills_tokens are True")

                # tokenize base_tokenizer
                tokens = base_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
                valid_tokens = tokens.input_ids != base_tokenizer.pad_token_id
                replace_pct = neigh_config.pct_words_masked * (neigh_config.span_length / (neigh_config.span_length + 2 * neigh_config.buffer_size))

                # replace replace_pct of input_ids with random tokens
                random_mask = torch.rand(tokens.input_ids.shape, device=self.device) < replace_pct
                random_mask &= valid_tokens
                random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=self.device)
                # while any of the random tokens are special tokens, replace them with random non-special tokens
                while any(base_tokenizer.decode(x) in base_tokenizer.all_special_tokens for x in random_tokens):
                    random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=self.device)
                tokens.input_ids[random_mask] = random_tokens
                perturbed_texts = base_tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)
            else:
                masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
                perturbed_texts = masked_texts
                # replace each <extra_id_*> with neigh_config.span_length random words from FILL_DICTIONARY
                for idx, text in enumerate(perturbed_texts):
                    filled_text = text
                    for fill_idx in range(count_masks([text])[0]):
                        fill = random.sample(self.FILL_DICTIONARY, span_length)
                        filled_text = filled_text.replace(f"<extra_id_{fill_idx}>", " ".join(fill))
                    assert count_masks([filled_text])[0] == 0, "Failed to replace all masks"
                    perturbed_texts[idx] = filled_text

        return perturbed_texts

    def generate_neighbors(self, texts, **kwargs) -> List[str]:
        n_neighbors = kwargs.get('n_perturbations', 25)
        # Repeat text if T-5 model
        texts_use = [x for x in texts for _ in range(n_neighbors)]

        chunk_size = self.config.chunk_size
        if '11b' in self.config.neighborhood_config.model:
            chunk_size //= 2

        outputs = []
        for i in tqdm(range(0, len(texts_use), chunk_size), desc="Applying perturbations"):
            outputs.extend(self.generate_neighbors_(texts_use[i:i + chunk_size], **kwargs))
        return outputs


class BertModel(MaskFillingModel):
    def __init__(self, config: ExperimentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.token_dropout = torch.nn.Dropout(p=0.7)
        if self.name == 'bert':
            self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=self.cache_dir)
            self.model = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir=self.cache_dir)
        elif self.name == 'distilbert':
            self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=self.cache_dir)
            self.model = transformers.DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased', cache_dir=self.cache_dir)
        elif self.name == 'roberta':
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', cache_dir=self.cache_dir)
            self.model = transformers.RobertaForMaskedLM.from_pretrained('roberta-base', cache_dir=self.cache_dir)
        else:
            raise ValueError(f"Invalid model name {self.name}")

    def generate_neighbors(self, texts, **kwargs) -> List[str]:
        neighbors = []
        for text in tqdm(texts, desc="Generating neighbors"):
            neighbors.extend(self.generate_neighbors_(text, **kwargs))
        return neighbors

    def generate_neighbors_(self, text: str, **kwargs):
        text_tokenized = self.tokenizer(text, padding=True, truncation=True,
                                        max_length=self.config.max_tokens, return_tensors='pt').input_ids.to(self.device)
        original_text = self.tokenizer.batch_decode(text_tokenized)[0]
        n_neighbors = kwargs.get('n_perturbations', 25)

        candidate_scores = dict()
        replacements = dict()

        for target_token_index in list(range(len(text_tokenized[0, :])))[1:]:

            target_token = text_tokenized[0, target_token_index]
            if self.name == 'bert':
                embeds = self.model.bert.embeddings(text_tokenized)
            elif self.name == 'distilbert':
                embeds = self.model.distilbert.embeddings(text_tokenized)
            elif self.name == 'roberta':
                embeds = self.model.roberta.embeddings(text_tokenized)

            embeds = torch.cat((embeds[:, :target_token_index, :], self.token_dropout(
                embeds[:, target_token_index, :]).unsqueeze(dim=0), embeds[:, target_token_index+1:, :]), dim=1)

            token_probs = torch.softmax(self.model(inputs_embeds=embeds).logits, dim=2)

            original_prob = token_probs[0, target_token_index, target_token]

            top_probabilities, top_candidates = torch.topk(
                token_probs[:, target_token_index, :], 6, dim=1)

            for cand, prob in zip(top_candidates[0], top_probabilities[0]):
                if not cand == target_token:

                    # alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to(device), text_tokenized[:,target_token_index+1:]), dim=1)
                    # alt_text = search_tokenizer.batch_decode(alt)[0]
                    if original_prob.item() == 1:
                        replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
                    else:
                        replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())

        replacement_keys = nlargest(n_neighbors, replacements, key=replacements.get)
        replacements_new = dict()
        for rk in replacement_keys:
            replacements_new[rk] = replacements[rk]
    
        replacements = replacements_new

        highest_scored = nlargest(100, replacements, key=replacements.get)

        neighbors, texts = [], []
        for single in highest_scored:
            alt = text_tokenized
            target_token_index, cand = single
            alt = torch.cat((alt[:, :target_token_index], torch.LongTensor([cand]).unsqueeze(0).to(self.device), alt[:, target_token_index+1:]), dim=1)
            alt_text = self.tokenizer.batch_decode(alt)[0]
            # Remove [CLS] and [SEP] tokens
            alt_text = alt_text.replace('[CLS]', '').replace('[SEP]', '')
            texts.append((alt_text, replacements[single]))
            neighbors.append(alt_text)

        # return texts
        return neighbors
