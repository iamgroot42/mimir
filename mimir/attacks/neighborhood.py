"""
    Neighborhood-MIA attack https://arxiv.org/pdf/2305.18462.pdf
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
from mimir.attacks.attack_utils import count_masks, apply_extracted_fills
from mimir.models import Model, ReferenceModel
from mimir.attacks.all_attacks import Attack


class NeighborhoodAttack(Attack):

    def __init__(
        self,
        config: ExperimentConfig,
        target_model: Model,
        ref_model: ReferenceModel = None,
        **kwargs,
    ):
        super().__init__(config, target_model, ref_model=None)
        self.ref_model = self._pick_neighbor_model()
        assert issubclass(type(self.ref_model), MaskFillingModel), "ref_model must be MaskFillingModel for neighborhood attack"

    def get_mask_model(self):
        """
            Return the mask filling model.
        """
        return self.ref_model

    def create_fill_dictionary(self, data):
        """
            (Only valid for T5 model) Create fill-fictionary used for random_fills
        """
        neigh_config = self.config.neighborhood_config
        if "t5" in neigh_config.model and neigh_config.random_fills:
            if not self.config.pretokenized:
                # TODO: maybe can be done if detokenized, but currently not supported
                self.ref_model.create_fill_dictionary(data)

    def _pick_neighbor_model(self):
        """
            Select and load the mask filling model requested in the config.
        """
        # mask filling t5 model
        mask_model = None
        neigh_config = self.config.neighborhood_config
        env_config = self.config.env_config

        model_kwargs = dict()
        if not neigh_config.random_fills:
            if env_config.int8:
                model_kwargs = dict(
                    load_in_8bit=True, device_map="auto", torch_dtype=torch.bfloat16
                )
            elif env_config.half:
                model_kwargs = dict(torch_dtype=torch.bfloat16)
            try:
                n_positions = (
                    512  # Should fix later, but for T-5 this is 512 indeed
                )
                # mask_model.config.n_positions
            except AttributeError:
                n_positions = self.config.max_tokens
        else:
            n_positions = self.config.max_tokens
        tokenizer_kwargs = {
            "model_max_length": n_positions,
        }

        print(f"Loading mask filling model {neigh_config.model}...")
        if "t5" in neigh_config.model:
            mask_model = T5Model(
                self.config,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
            )
        elif "bert" in neigh_config.model:
            mask_model = BertModel(self.config)
        else:
            raise ValueError(f"Unknown model {neigh_config.model}")
        # if config.dataset_member in ['english', 'german']:
        #     preproc_tokenizer = mask_tokenizer
        return mask_model

    def load(self):
        """
        Any attack-specific steps (one-time) preparation
        """
        print("MOVING MASK MODEL TO GPU...", end="", flush=True)
        self.ref_model.load()

    def get_neighbors(self, documents, **kwargs):
        """
            Generate neighbors for given documents.
        """
        n_perturbations = kwargs.get("n_perturbations", 1)
        span_length = kwargs.get("span_length", 10)
        neigh_config = self.config.neighborhood_config
        ceil_pct = neigh_config.ceil_pct
        kwargs = {}
        if type(self.ref_model) == T5Model:
            kwargs = {
                "span_length": span_length,
                "pct": neigh_config.pct_words_masked,
                "chunk_size": self.config.chunk_size,
                "ceil_pct": ceil_pct,
            }
        kwargs["n_perturbations"] = n_perturbations

        # Generate neighbors
        neighbors = self.ref_model.generate_neighbors(documents, **kwargs)
        return neighbors

    def _attack(self, document, probs, tokens=None, **kwargs):
        """
        Neighborhood attack score. Looks at difference in likelihood for given document and average likelihood of its neighbors
        """
        # documents here are actually neighbors
        batch_size = kwargs.get("batch_size", 4)
        substr_neighbors = kwargs.get("substr_neighbors", None)
        loss = kwargs.get("loss", None)
        if loss is None:
            loss = self.target_model.get_ll(document, probs=probs, tokens=tokens)

        # Only evaluate neighborhood attack when not caching neighbors
        mean_substr_score = self.target_model.get_lls(
            substr_neighbors, batch_size=batch_size
        )
        d_based_score = loss - mean_substr_score
        return d_based_score


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
        model_kwargs = self.kwargs.get("model_kwargs", {})
        tokenizer_kwargs = self.kwargs.get("tokenizer_kwargs", {})

        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self.name, **model_kwargs, cache_dir=self.cache_dir
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.name, **tokenizer_kwargs, cache_dir=self.cache_dir
        )

        # define regex to match all <extra_id_*> tokens, where * is an integer
        self.pattern = re.compile(r"<extra_id_\d+>")

    def create_fill_dictionary(self, data):
        self.FILL_DICTIONARY = set()
        for texts in data.values():
            for text in texts:
                self.FILL_DICTIONARY.update(text.split())
        self.FILL_DICTIONARY = sorted(list(self.FILL_DICTIONARY))

    def tokenize_and_mask(
        self, text: str, span_length: int, pct: float, ceil_pct: bool = False
    ):
        buffer_size = self.config.neighborhood_config.buffer_size

        tokens = text.split(" ")
        mask_string = "<<<mask>>>"

        span_length = min(int(pct * len(tokens)), span_length)
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
                tokens[idx] = f"<extra_id_{num_filled}>"
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = " ".join(tokens)
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
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True).to(
            self.device
        )
        outputs = self.model.generate(
            **tokens,
            max_length=150,
            do_sample=True,
            top_p=mask_top_p,
            num_return_sequences=1,
            eos_token_id=stop_id,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def generate_neighbors_(self, texts: List[str], **kwargs):
        span_length: int = kwargs.get("span_length")
        pct: float = kwargs.get("pct")
        ceil_pct: bool = kwargs.get("ceil_pct", False)
        base_tokenizer = kwargs.get("base_tokenizer", None)
        neigh_config = self.config.neighborhood_config

        if not neigh_config.random_fills:
            masked_texts = [
                self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts
            ]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == ""]

            # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
            attempts = 1
            break_out_of_loop: bool = False
            while "" in perturbed_texts:
                if attempts > neigh_config.max_tries:
                    for idx in idxs:
                        perturbed_texts[idx] = texts[idx]
                    break_out_of_loop = True
                    break
                if break_out_of_loop:
                    break
                idxs = [idx for idx, x in enumerate(perturbed_texts) if x == ""]
                print(
                    f"WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}]."
                )
                masked_texts = [
                    self.tokenize_and_mask(x, span_length, pct, ceil_pct)
                    for idx, x in enumerate(texts)
                    if idx in idxs
                ]
                raw_fills = self.replace_masks(masked_texts)
                extracted_fills = self.extract_fills(raw_fills)
                new_perturbed_texts = apply_extracted_fills(
                    masked_texts, extracted_fills
                )
                for idx, x in zip(idxs, new_perturbed_texts):
                    perturbed_texts[idx] = x
                attempts += 1
        else:
            if neigh_config.random_fills_tokens:
                if base_tokenizer is None:
                    raise ValueError(
                        "base_tokenizer must be provided if random_fills and random_fills_tokens are True"
                    )

                # tokenize base_tokenizer
                tokens = base_tokenizer(texts, return_tensors="pt", padding=True).to(
                    self.device
                )
                valid_tokens = tokens.input_ids != base_tokenizer.pad_token_id
                replace_pct = neigh_config.pct_words_masked * (
                    neigh_config.span_length
                    / (neigh_config.span_length + 2 * neigh_config.buffer_size)
                )

                # replace replace_pct of input_ids with random tokens
                random_mask = (
                    torch.rand(tokens.input_ids.shape, device=self.device) < replace_pct
                )
                random_mask &= valid_tokens
                random_tokens = torch.randint(
                    0,
                    base_tokenizer.vocab_size,
                    (random_mask.sum(),),
                    device=self.device,
                )
                # while any of the random tokens are special tokens, replace them with random non-special tokens
                while any(
                    base_tokenizer.decode(x) in base_tokenizer.all_special_tokens
                    for x in random_tokens
                ):
                    random_tokens = torch.randint(
                        0,
                        base_tokenizer.vocab_size,
                        (random_mask.sum(),),
                        device=self.device,
                    )
                tokens.input_ids[random_mask] = random_tokens
                perturbed_texts = base_tokenizer.batch_decode(
                    tokens.input_ids, skip_special_tokens=True
                )
            else:
                masked_texts = [
                    self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts
                ]
                perturbed_texts = masked_texts
                # replace each <extra_id_*> with neigh_config.span_length random words from FILL_DICTIONARY
                for idx, text in enumerate(perturbed_texts):
                    filled_text = text
                    for fill_idx in range(count_masks([text])[0]):
                        fill = random.sample(self.FILL_DICTIONARY, span_length)
                        filled_text = filled_text.replace(
                            f"<extra_id_{fill_idx}>", " ".join(fill)
                        )
                    assert (
                        count_masks([filled_text])[0] == 0
                    ), "Failed to replace all masks"
                    perturbed_texts[idx] = filled_text

        return perturbed_texts

    def generate_neighbors(self, texts, **kwargs) -> List[str]:
        n_neighbors = kwargs.get("n_perturbations", 25)
        # Repeat text if T-5 model
        texts_use = [x for x in texts for _ in range(n_neighbors)]

        chunk_size = self.config.chunk_size
        if "11b" in self.config.neighborhood_config.model:
            chunk_size //= 2

        outputs = []
        for i in tqdm(
            range(0, len(texts_use), chunk_size), desc="Applying perturbations"
        ):
            outputs.extend(
                self.generate_neighbors_(texts_use[i : i + chunk_size], **kwargs)
            )
        return outputs


class BertModel(MaskFillingModel):
    def __init__(self, config: ExperimentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.token_dropout = torch.nn.Dropout(p=0.7)
        if self.name == "bert":
            self.tokenizer = transformers.BertTokenizerFast.from_pretrained(
                "bert-base-uncased", cache_dir=self.cache_dir
            )
            self.model = transformers.BertForMaskedLM.from_pretrained(
                "bert-base-uncased", cache_dir=self.cache_dir
            )
        elif self.name == "distilbert":
            self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased", cache_dir=self.cache_dir
            )
            self.model = transformers.DistilBertForMaskedLM.from_pretrained(
                "distilbert-base-uncased", cache_dir=self.cache_dir
            )
        elif self.name == "roberta":
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(
                "roberta-base", cache_dir=self.cache_dir
            )
            self.model = transformers.RobertaForMaskedLM.from_pretrained(
                "roberta-base", cache_dir=self.cache_dir
            )
        else:
            raise ValueError(f"Invalid model name {self.name}")

    def generate_neighbors(self, texts, **kwargs) -> List[str]:
        neighbors = []
        for text in tqdm(texts, desc="Generating neighbors"):
            neighbors.extend(self.generate_neighbors_(text, **kwargs))
        return neighbors

    def generate_neighbors_(self, text: str, **kwargs):
        in_place_swap = self.config.neighborhood_config.original_tokenization_swap

        tokenizer_output = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_offsets_mapping=in_place_swap,
            max_length=self.config.max_tokens,
            return_tensors="pt",
        )
        text_tokenized = tokenizer_output.input_ids.to(self.device)
        n_neighbors = kwargs.get("n_perturbations", 25)
        num_tokens = len(text_tokenized[0, :])
        n_swap = int(num_tokens * self.config.neighborhood_config.pct_swap_bert)

        if in_place_swap:
            token_positions = tokenizer_output.offset_mapping[0]

        replacements = dict()

        target_token_indices = range(1, num_tokens)
        for target_token_index in target_token_indices:
            target_token = text_tokenized[0, target_token_index]
            if self.name == "bert":
                embeds = self.model.bert.embeddings(text_tokenized)
            elif self.name == "distilbert":
                embeds = self.model.distilbert.embeddings(text_tokenized)
            elif self.name == "roberta":
                embeds = self.model.roberta.embeddings(text_tokenized)

            embeds = torch.cat(
                (
                    embeds[:, :target_token_index, :],
                    self.token_dropout(embeds[:, target_token_index, :]).unsqueeze(
                        dim=0
                    ),
                    embeds[:, target_token_index + 1 :, :],
                ),
                dim=1,
            )

            token_probs = torch.softmax(self.model(inputs_embeds=embeds).logits, dim=2)

            original_prob = token_probs[0, target_token_index, target_token]

            top_probabilities, top_candidates = torch.topk(
                token_probs[:, target_token_index, :], 6, dim=1
            )

            for cand, prob in zip(top_candidates[0], top_probabilities[0]):
                if not cand == target_token:
                    # alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to(device), text_tokenized[:,target_token_index+1:]), dim=1)
                    # alt_text = search_tokenizer.batch_decode(alt)[0]
                    if original_prob.item() == 1:
                        replacements[(target_token_index, cand)] = prob.item() / (
                            1 - 0.9
                        )
                    else:
                        replacements[(target_token_index, cand)] = prob.item() / (
                            1 - original_prob.item()
                        )

        if self.config.neighborhood_config.neighbor_strategy == "deterministic":
            replacement_keys = nlargest(n_neighbors, replacements, key=replacements.get)
            replacements_new = dict()
            for rk in replacement_keys:
                replacements_new[rk] = replacements[rk]

            replacements = replacements_new

            # TODO: Not sure if this is needed (perhaps making sure we never take >= 100)? Consider removing later
            highest_scored = nlargest(100, replacements, key=replacements.get)

            neighbors = []
            for single in highest_scored:
                target_token_index, cand = single

                if in_place_swap:
                    # Get indices of original text that we want to swap out
                    start, end = token_positions[target_token_index]
                    # Get text corresponding to cand token
                    fill_in_text = self.tokenizer.decode(cand)
                    # Remove any '##' from prefix (since we're doing a plug back into text)
                    fill_in_text = fill_in_text.replace("##", "")
                    alt_text = text[:start] + fill_in_text + text[end:]
                else:
                    alt = text_tokenized
                    alt = torch.cat(
                        (
                            alt[:, :target_token_index],
                            torch.LongTensor([cand]).unsqueeze(0).to(self.device),
                            alt[:, target_token_index + 1 :],
                        ),
                        dim=1,
                    )
                    alt_text = self.tokenizer.batch_decode(alt)[0]
                    # Remove [CLS] and [SEP] tokens
                    alt_text = alt_text.replace("[CLS]", "").replace("[SEP]", "")
                    # texts.append((alt_text, replacements[single]))
                neighbors.append(alt_text)

        elif self.config.neighborhood_config.neighbor_strategy == "random":
            if not in_place_swap:
                raise ValueError(
                    "Random neighbor strategy only works with in_place_swap=True right now"
                )

            # Make new dict replacements_new with structure [key[0]]: (key[1], value) for all keys in replacements
            replacements_new = dict()
            for k, v in replacements.items():
                if k[0] not in replacements_new:
                    replacements_new[k[0]] = []
                replacements_new[k[0]].append((k[1].item(), v))
            # Sort each entry by score
            for k, v in replacements_new.items():
                replacements_new[k] = sorted(v, key=lambda x: x[1], reverse=True)

            num_trials = int(1e3)
            replacements, scores = [], []
            for _ in range(num_trials):
                # Pick n_swap random positions
                swap_positions = np.random.choice(
                    list(replacements_new.keys()), n_swap, replace=False
                )
                # Out of all replacements, pick keys where target_token_index is in swap_positions
                picked = [replacements_new[x][0] for x in swap_positions]
                # Compute score (sum)
                score = sum([x[1] for x in picked])
                scores.append(score)
                # Also keep track of replacements (position, candidate)
                replacements.append(
                    [(i, replacements_new[i][0][0]) for i in swap_positions]
                )

            # Out of all trials, pick n_neighbors combinations (highest scores)
            highest_scored = nlargest(
                n_neighbors, zip(scores, replacements), key=lambda x: x[0]
            )

            neighbors = []
            for _, single in highest_scored:
                # Sort according to target_token_index
                single = sorted(single, key=lambda x: x[0])
                # Get corresponding positions in text
                single = [
                    (token_positions[target_token_index], cand)
                    for target_token_index, cand in single
                ]
                # Add start of text (before first swap)
                end_prev = 0
                alt_text = ""
                for (start, end), cand in single:
                    # Get text corresponding to cand token
                    fill_in_text = self.tokenizer.decode(cand)
                    # Remove any '##' from prefix (since we're doing a plug back into text)
                    fill_in_text = fill_in_text.replace("##", "")
                    alt_text += text[end_prev:start] + fill_in_text
                    end_prev = end
                # Add remainder text (after last swap)
                start, end = single[-1][0]
                alt_text += text[end:]
                neighbors.append(alt_text)

        else:
            raise NotImplementedError(
                f"Invalid neighbor strategy {self.config.neighborhood_config.neighbor_strategy}"
            )

        # return texts
        return neighbors
