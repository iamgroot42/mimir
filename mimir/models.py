import torch
import torch.nn as nn
import openai
from typing import List
import numpy as np
import transformers
import time
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import zlib

from mimir.config import ExperimentConfig
from mimir.custom_datasets import SEPARATOR
from mimir.data_utils import drop_last_word

from utils.transformers.model import OpenLMforCausalLM


class Model(nn.Module):
    """
        Base class (for LLMs)
    """
    def __init__(self, config: ExperimentConfig, **kwargs):
        super().__init__()
        self.model = None # Set by child class
        self.tokenizer = None # Set by child class
        self.config = config
        self.device = None
        self.device_map = None
        self.name = None
        self.kwargs = kwargs
        self.cache_dir = self.config.env_config.cache_dir
    
    def to(self, device):
        self.model.to(device)
    
    def load(self):
        """
            Load model onto GPU (and compile, if requested) if not already loaded with device map
        """
        if not self.device_map:
            start = time.time()
            try:
                self.model.cpu()
            except NameError:
                pass
            if self.config.openai_config is None:
                self.model.to(self.device)
            if self.config.env_config.compile:
                torch.compile(self.model)
            print(f'DONE ({time.time() - start:.2f}s)')
    
    def unload(self):
        """
            Unload model from GPU
        """
        start = time.time()
        try:
            self.model.cpu()
        except NameError:
            pass
        print(f'DONE ({time.time() - start:.2f}s)')

    @torch.no_grad()
    def get_probabilities(self, text: str, tokens=None):
        """
            Get the probabilities or log-softmaxed logits for a text under the current model
        """
        if self.device is None or self.name is None:
            raise ValueError("Please set self.device and self.name in child class")

        if tokens is not None:
            labels = torch.from_numpy(tokens.astype(np.int64)).type(torch.LongTensor)
            if labels.shape[0] != 1:
                # expand first dimension
                labels = labels.unsqueeze(0)
            labels = labels.to(self.device)
        else:
            tokenized = self.tokenizer(
                text, return_tensors="pt").to(self.device)
            labels = tokenized.input_ids

        all_prob = []
        for i in range(0, labels.size(1), self.stride):
            begin_loc = max(i + self.stride - self.max_length, 0)
            end_loc = min(i + self.stride, labels.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = labels[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = self.model(input_ids, labels=target_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            probabilities = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            shift_labels = target_ids[..., 1:].contiguous()
            labels_processed = shift_labels[0]

            for i, token_id in enumerate(labels_processed):
                if token_id != -100:
                    probability = probabilities[0, i, token_id].item()
                    all_prob.append(probability)
        # Should be equal to # of tokens - 1 to account for shift
        assert len(all_prob) == labels.size(1) - 1

        return all_prob
    
    @torch.no_grad()
    def get_ll(self, text: str, tokens=None, probs=None):
        """
            Get the log likelihood of each text under the base_model
        """
        all_prob = probs if probs is not None else self.get_probabilities(text, tokens=tokens)
        return -np.mean(all_prob)
    
    def load_base_model_and_tokenizer(self, model_kwargs):
        if self.device is None or self.name is None:
            raise ValueError("Please set self.device and self.name in child class")

        if self.config.openai_config is None:
            print(f'Loading BASE model {self.name}...')
            device_map = self.device_map if self.device_map else 'cpu'
            if "silo" in self.name or "balanced" in self.name:
                model = OpenLMforCausalLM.from_pretrained(
                    self.name, **model_kwargs, device_map=self.device, cache_dir=self.cache_dir)
                # Extract the model from the model wrapper so we dont need to call model.model
            elif "llama" in self.name or "alpaca" in self.name:
                # llama is too big, gotta use device map
                model = transformers.AutoModelForCausalLM.from_pretrained(self.name, **model_kwargs, device_map="balanced_low_0", cache_dir=self.cache_dir)
                self.device = 'cuda:1'
            else:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.name, **model_kwargs, device_map=device_map, cache_dir=self.cache_dir)
        else:
            model = None

        optional_tok_kwargs = {}
        if "facebook/opt-" in self.name:
            print("Using non-fast tokenizer for OPT")
            optional_tok_kwargs['fast'] = False
        if self.config.dataset_member in ['pubmed'] or self.config.dataset_nonmember in ['pubmed']:
            optional_tok_kwargs['padding_side'] = 'left'
            self.pad_token = self.tokenizer.eos_token_id
        if "silo" in self.name or "balanced" in self.name:
            tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(
                "EleutherAI/gpt-neox-20b", **optional_tok_kwargs, cache_dir=self.cache_dir)
        elif "datablations" in self.name:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                "gpt2", **optional_tok_kwargs, cache_dir=self.cache_dir)
        elif "llama" in self.name or "alpaca" in self.name:
            tokenizer = transformers.LlamaTokenizer.from_pretrained(
                self.name, **optional_tok_kwargs, cache_dir=self.cache_dir)
        elif "pubmedgpt" in self.name:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                "stanford-crfm/BioMedLM", **optional_tok_kwargs, cache_dir=self.cache_dir)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.name, **optional_tok_kwargs, cache_dir=self.cache_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        return model, tokenizer
    
    def load_model_properties(self):
         # TODO: getting max_length of input could be more generic
        if "silo" in self.name or "balanced" in self.name:
            self.max_length = self.model.model.seq_len
        elif hasattr(self.model.config, 'max_position_embeddings'):
            self.max_length = self.model.config.max_position_embeddings
        elif hasattr(self.model.config, 'n_positions'):
            self.max_length = self.model.config.n_positions
        else:
            # Default window size
            self.max_length = 1024
        self.stride = self.max_length // 2


class ReferenceModel(Model):
    """
        Wrapper for reference model
    """
    def __init__(self, config: ExperimentConfig, name: str):
        super().__init__(config)
        self.device = self.config.env_config.device_aux
        self.name = name
        base_model_kwargs = {'revision': 'main'}
        if 'gpt-j' in self.name or 'neox' in self.name or 'llama' in self.name or 'alpaca' in self.name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in self.name:
            base_model_kwargs.update(dict(revision='float16'))
        if ':' in self.name:
            print("Applying ref model revision")
            # Allow them to provide revisions as part of model name, then parse accordingly
            split = self.name.split(':')
            self.name = split[0]
            base_model_kwargs.update(dict(revision=split[-1]))
        self.model, self.tokenizer = self.load_base_model_and_tokenizer(
            model_kwargs=base_model_kwargs)
        self.load_model_properties()


class QuantileReferenceModel(Model):
    """
        Wrapper for referenc model, specifically used for quantile regression
    """
    def __init__(self, config: ExperimentConfig, name: str):
        super().__init__(config)
        self.device = self.config.env_config.device_aux
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name,
            num_labels=2,
            max_position_embeddings=1024)
        # Modify model's last linear layer to have only 1 output
        self.model.classifier.linear_out = nn.Linear(self.model.classifier.linear_out.in_features, 1)
        self.load_model_properties()


class EvalModel(Model):
    """
        GPT-based detector that can distinguish between machine-generated and human-written text
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.device = self.config.env_config.device_aux
        self.name = 'roberta-base-openai-detector'
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(self.name, cache_dir=self.cache_dir).to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.name, cache_dir=self.cache_dir)
 
    @torch.no_grad()
    def get_preds(self, data):
        batch_size = self.config.batch_size
        preds = []
        for batch in tqdm(range(len(data) // batch_size), desc="Evaluating fake"):
            batch_fake = data[batch * batch_size:(batch + 1) * batch_size]
            batch_fake = self.tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            preds.extend(self.model(**batch_fake).logits.softmax(-1)[:,0].tolist())
        return preds


class LanguageModel(Model):
    """
        Generic LM- used most often for target model
    """
    def __init__(self, config: ExperimentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.device = self.config.env_config.device
        self.device_map = self.config.env_config.device_map        
        # Use provided name (if provided)
        # Relevant for scoring-model scenario
        self.name = self.kwargs.get('name', self.config.base_model)

        base_model_kwargs = {}
        if config.revision:
            base_model_kwargs.update(dict(revision=config.revision))
        if 'gpt-j' in self.name or 'neox' in self.name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in self.name:
            base_model_kwargs.update(dict(revision='float16'))
        self.model, self.tokenizer = self.load_base_model_and_tokenizer(
            model_kwargs=base_model_kwargs)
        self.load_model_properties()
        
    
    @torch.no_grad()
    def get_ref(self, text: str, ref_model: ReferenceModel, tokens=None, probs=None):
        """
            Compute the loss of a given text calibrated against the text's loss under a reference model -- MIA baseline
        """
        lls = self.get_ll(text, tokens=tokens, probs=probs)
        lls_ref = ref_model.get_ll(text)

        return lls - lls_ref

    @torch.no_grad()
    def get_rank(self, text: str, log: bool=False):
        """
            Get the average rank of each observed token sorted by model likelihood
        """
        openai_config = self.config.openai_config
        assert openai_config is None, "get_rank not implemented for OpenAI models"

        tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
        logits = self.model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()

    # TODO extend for longer sequences
    @torch.no_grad()
    def get_lls(self, texts: str):
        # return [self.get_ll(text) for text in texts]
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True)
        labels = tokenized.input_ids
        batch_size = 25
        losses = []
        for i in range(0, labels.shape[0], batch_size):
            label_batch = labels[i:i+batch_size].to(self.device)
            output = self.model(label_batch, labels=label_batch)
            logits = output.logits
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_logits = torch.transpose(shift_logits, 1, 2)
            shift_labels = label_batch[..., 1:].contiguous()
            loss = F.cross_entropy(input=shift_logits, target=shift_labels, reduction='none').mean(dim=1)
            losses.extend(loss.tolist())
        return losses
    
    @torch.no_grad()
    def get_min_k_prob(self, text: str, tokens=None, probs=None, k=.2, window=1, stride=1):
        all_prob = probs if probs is not None else self.get_probabilities(text, tokens=tokens)
        # iterate through probabilities by ngram defined by window size at given stride
        ngram_probs = []
        for i in range(0, len(all_prob) - window + 1, stride):
            ngram_prob = all_prob[i:i+window]
            ngram_probs.append(np.mean(ngram_prob))
        min_k_probs = sorted(ngram_probs)[:int(len(ngram_probs) * k)]

        return -np.mean(min_k_probs)
    

    def sample_from_model(self, texts: List[str], **kwargs):
        """
            Sample from base_model using ****only**** the first 30 tokens in each example as context
        """
        min_words = kwargs.get('min_words', 55)
        max_words = kwargs.get('max_words', 200)
        prompt_tokens = kwargs.get('prompt_tokens', 30)

        # encode each text as a list of token ids
        if self.config.dataset_member == 'pubmed':
            texts = [t[:t.index(SEPARATOR)] for t in texts]
            all_encoded = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        else:
            all_encoded = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < min_words and tries <  self.config.neighborhood_config.top_p:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {}
            if self.config.do_top_p:
                sampling_kwargs['top_p'] = self.config.top_p
            elif self.config.do_top_k:
                sampling_kwargs['top_k'] = self.config.top_k
            #min_length = 50 if config.dataset_member in ['pubmed'] else 150

            #outputs = base_model.generate(**all_encoded, min_length=min_length, max_length=max_length, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
            #removed minlen and attention mask min_length=min_length, max_length=200, do_sample=True,pad_token_id=base_tokenizer.eos_token_id,
            outputs = self.model.generate(**all_encoded, min_length=min_words*2, max_length=max_words*3,  **sampling_kwargs,  eos_token_id=self.tokenizer.eos_token_id)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

        return decoded

    @torch.no_grad()
    def get_entropy(self, text: str):
        """
            Get average entropy of each token in the text
        """
        # raise NotImplementedError("get_entropy not implemented for OpenAI models")
        
        tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
        logits = self.model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()

    @torch.no_grad()
    def get_zlib_entropy(self, text: str, tokens=None, probs=None):
        zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
        return self.get_ll(text, tokens=tokens, probs=probs) / zlib_entropy
    
    @torch.no_grad()
    def get_max_norm(self, text: str, context_len=None, tk_freq_map=None):
        # TODO: update like oher attacks
        tokenized = self.tokenizer(
            text, return_tensors="pt").to(self.device)
        labels = tokenized.input_ids

        max_length = context_len if context_len is not None else self.max_length
        stride = max_length // 2 #self.stride
        all_prob = []
        for i in range(0, labels.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, labels.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = labels[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = self.model(input_ids, labels=target_ids)
            logits = outputs.logits
            # Shift so that tokens < n predict n
            # print(logits.shape)
            shift_logits = logits[..., :-1, :].contiguous()
            # shift_logits = torch.transpose(shift_logits, 1, 2)
            probabilities = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            shift_labels = target_ids[..., 1:].contiguous()
            labels_processed = shift_labels[0]

            for i, token_id in enumerate(labels_processed):
                if token_id != -100:
                    probability = probabilities[0, i, token_id].item()
                    max_tk_prob = torch.max(probabilities[0, i]).item()
                    tk_weight = max(tk_freq_map[token_id.item()], 1) / sum(tk_freq_map.values()) if tk_freq_map is not None else 1
                    if tk_weight == 0:
                        print("0 count token", token_id.item())
                    tk_norm = tk_weight
                    all_prob.append((1 - (max_tk_prob - probability)) / tk_norm)

        # Should be equal to # of tokens - 1 to account for shift
        assert len(all_prob) == labels.size(1) - 1
        return -np.mean(all_prob)


class OpenAI_APIModel(LanguageModel):
    """
        Wrapper for OpenAI API calls
    """
    def __init__(self, config: ExperimentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = None
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=self.cache_dir)
        self.API_TOKEN_COUNTER = 0
    
    @property
    def api_calls(self):
        return self.API_TOKEN_COUNTER

    @torch.no_grad()
    def get_ll(self, text: str):
        """
            Get the log likelihood of each text under the base_model
        """
        openai_config = self.config.openai_config

        kwargs = {"engine": openai_config.model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)

    @torch.no_grad()
    def get_ref(self, text: str, ref_model: ReferenceModel):
        """
            Get the  likelihood ratio of each text under the base_model -- MIA baseline
        """
        raise NotImplementedError("OpenAI model not implemented for LIRA")
        openai_config = self.config.openai_config
        kwargs = {"engine": openai_config.model, "temperature": 0,
                    "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)

    def get_lls(self, texts: str):

        # use GPT2_TOKENIZER to get total number of tokens
        total_tokens = sum(len(self.tokenizer.encode(text)) for text in texts)
        self.API_TOKEN_COUNTER += total_tokens * 2  # multiply by two because OpenAI double-counts echo_prompt tokens

        pool = ThreadPool(self.config.batch_size)
        return pool.map(self.get_ll, texts)

    def _openai_sample(self, p: str):
        openai_config = self.config.openai_config
        if self.config.dataset_member != 'pubmed':  # keep Answer: prefix for pubmed
            p = drop_last_word(p)

        # sample from the openai model
        kwargs = { "engine": openai_config.model, "max_tokens": 200 }
        if self.config.do_top_p:
            kwargs['top_p'] = self.config.top_p
    
        r = openai.Completion.create(prompt=f"{p}", **kwargs)
        return p + r['choices'][0].text


    def sample_from_model(self, texts: List[str], **kwargs):
        """
            Sample from base_model using ****only**** the first 30 tokens in each example as context
        """
        prompt_tokens = kwargs.get('prompt_tokens', 30)
        base_tokenizer = kwargs.get('base_tokenizer', None)
        if base_tokenizer is None:
            raise ValueError("Please provide base_tokenizer")

        # encode each text as a list of token ids
        if self.config.dataset_member == 'pubmed':
            texts = [t[:t.index(SEPARATOR)] for t in texts]
            all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        else:
            all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

        # decode the prefixes back into text
        prefixes = base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
        pool = ThreadPool(self.config.batch_size)

        decoded = pool.map(self._openai_sample, prefixes)

        # count total number of tokens with GPT2_TOKENIZER
        total_tokens = sum(len(self.tokenizer.encode(x)) for x in decoded)
        self.API_TOKEN_COUNTER += total_tokens

        return decoded
    
    @torch.no_grad()
    def get_entropy(self, text: str):
        """
            Get average entropy of each token in the text
        """
        raise NotImplementedError("get_entropy not implemented for OpenAI models")
