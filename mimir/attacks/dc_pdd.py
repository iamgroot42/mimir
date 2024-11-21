"""
    DC-PDD Attack: https://aclanthology.org/2024.emnlp-main.300/
    Based on the official implementation: https://github.com/zhang-wei-chao/DC-PDD
"""
import torch as ch
from tqdm import tqdm
import numpy as np
import requests
import io
import gzip
import os
import json
from mimir.attacks.all_attacks import Attack
from mimir.models import Model
from mimir.config import ExperimentConfig
from mimir.utils import get_cache_path


def ensure_parent_directory_exists(filename):
    # Get the parent directory from the given filename
    parent_dir = os.path.dirname(filename)
    
    # Create the parent directory if it does not exist
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)


class DC_PDDAttack(Attack):

    def __init__(self, config: ExperimentConfig, model: Model):
        super().__init__(config, model, ref_model=None)
        # Use subset of C-4
        self.fre_dis = ch.zeros(len(model.tokenizer))
        # Account for model name
        model_name = model.name

        # Load from cache if available, save otherwise
        cached_file_path = os.path.join(get_cache_path(), "DC_PDD_freq_dis", "C4", f"{model_name}.pt")

        if os.path.exists(cached_file_path):
            self.fre_dis = ch.load(cached_file_path)
            print(f"Loaded frequency distribution from cache for {model_name}")
        else:
            # Make sure the directory exists
            ensure_parent_directory_exists(cached_file_path)
            # Collect frequency data
            self._collect_frequency_data()
            ch.save(self.fre_dis, cached_file_path)
            print(f"Saved frequency distribution to cache for {model_name}")

        # Laplace smoothing
        self.fre_dis = (1 + self.fre_dis) / (ch.sum(self.fre_dis) + len(self.fre_dis))

    def _fre_dis(self, ref_data, max_tok: int = 1024):
        """
        token frequency distribution
        ref_data: reference dataset
        tok: tokenizer
        """
        # Tokenize all the text in the reference dataset
        # input_ids = self.target_model.tokenizer(ref_data, truncation=True, max_length=max_tok).input_ids
        for text in tqdm(ref_data):
            input_ids = self.target_model.tokenizer(text, truncation=True, max_length=max_tok).input_ids
            self.fre_dis[input_ids] += 1

    def _collect_frequency_data(self, fil_num: int = 15):
        for i in tqdm(range(fil_num), desc="Downloading and processing dataset"):
            # Download the dataset split
            url = f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{"{:05}".format(i)}-of-01024.json.gz"
            # Download the file
            response = requests.get(url)
            response.raise_for_status()  # Check for download errors

            # Open and parse the .json.gz file - the file is a .json file with one json object per line
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_file:
                sub_dataset = gz_file.readlines()
                examples = []
                # for example in tqdm(sub_dataset):
                for example in sub_dataset:
                    example = json.loads(example)
                    examples.append(example['text'])

                # Compute the frequency distribution
                self._fre_dis(examples)

    @ch.no_grad()
    def _attack(self, document, probs, tokens=None, **kwargs):
        """
        DC-PDD Attack: Use frequency distribution of some large corpus to "calibrate" token probabilities
        and compute a membership score.
        """
        # Hyper-params specific to DC-PDD
        a: float = kwargs.get("a", 0.01)

        # Tokenize text (we process things slightly differently)
        tokens_og = self.target_model.tokenizer(document, return_tensors="pt").input_ids
        # Inject EOS token at beginning
        tokens = ch.cat([ch.tensor([[self.target_model.tokenizer.eos_token_id]]), tokens_og], dim=1).numpy()

        # these are all log probabilites
        probs_with_start_token = self.target_model.get_probabilities(document, tokens=tokens)
        x_pro = np.exp(probs_with_start_token)

        indexes = []
        current_ids = []
        input_ids = tokens_og[0]
        for i, input_id in enumerate(input_ids):
            if input_id not in current_ids:
                indexes.append(i)
                current_ids.append(input_id)

        x_pro = x_pro[indexes]
        x_fre = self.fre_dis[input_ids[indexes]].numpy()

        # Compute alpha values
        alpha = x_pro * np.log(1 / x_fre)

        # Compute membership score
        alpha[alpha > a] = a

        beta = - np.mean(alpha)

        return beta
