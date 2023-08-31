"""
    Datasets and data-processing utilities
"""
import datasets
import random
import numpy as np
from typing import List
import os
import mimir.custom_datasets as custom_datasets
from mimir.config import ExperimentConfig


NAME_KEY_MAPPING = {
    'the_pile': 'text',
    'xsum': 'document'
}


class Data:
    def __init__(self, name, config: ExperimentConfig):
        self.config = config
        self.name = name
        self.key = NAME_KEY_MAPPING.get(name, None)
        if self.key is None:
            raise ValueError(f'Key for dataset {name} not found in NAME_KEY_MAPPING')
        self.cache_dir = self.config.env_config.cache_dir

    def load(self, tokenizer, train: bool):
        data_split = 'train' if train else 'test'
        n_samples = self.config.n_samples

        # Load from cache, if requested
        if self.config.load_from_cache:
            if self.config.specific_sources and self.name == 'the_pile':
                sorted_sources = "_".join(sorted(self.config.specific_sources))
                filename = f'{self.name}_{sorted_sources}'
            else:
                filename = self.name
            data = custom_datasets.load_cached(self.cache_dir, data_split, filename,
                                               min_length=self.config.min_words, max_length=self.config.max_words,
                                               n_samples=self.config.n_samples, max_tokens=self.config.max_tokens)
        else:
            if self.name in custom_datasets.DATASETS:
                data = custom_datasets.load(self.name)
            elif self.name == 'the_pile':
                min_load = max(10000, self.config.max_data)
                data = datasets.load_dataset("json", data_files=os.path.join(self.config.env_config.data_source, "pile/00.jsonl.zst" if train else "pile/test.jsonl.zst"), cache_dir=self.cache_dir, split=f"train[:{min_load}]")
                data = pile_selection_utility(data, self.key, wanted_sources=self.config.specific_sources)
            elif 'human' in self.name :
                data = datasets.load_dataset(self.name, split=f'train[:100]', cache_dir=self.cache_dir)[self.key]
            elif 'nthngdy' in self.name:
                data = datasets.load_dataset(self.name, split='test', cache_dir=self.cache_dir)[self.key]
            else:
                data = datasets.load_dataset(self.name, split=f'train', cache_dir=self.cache_dir)[self.key]
    
        # get unique examples, strip whitespace, and remove newlines
        # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
        # then take just the examples that are <= 512 tokens (for the mask model)
        # then generate n_samples samples

        # remove duplicates from the data
        data = list(dict.fromkeys(data))  # deterministic, as opposed to set()
    
        # strip whitespace around each example
        data = [x.strip() for x in data]

        # remove newlines from each example
        data = [strip_newlines(x) for x in data]

        long_data = [x for x in data if len(x.split()) > self.config.min_words]
        if len(long_data) > 0:
            data = long_data

        not_too_long_data = [x for x in data if len(x.split()) < self.config.max_words]
        if len(not_too_long_data) > 0:
            data = not_too_long_data

        random.seed(0)
        random.shuffle(data)

        data = data[:self.config.max_data]

        # keep only examples with <= 512 tokens according to mask_tokenizer
        # this step has the extra effect of removing examples with low-quality/garbage content
        tokenized_data = tokenizer(data)
        data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= self.config.max_tokens]

        # print stats about remainining data
        print(f"Total number of samples: {len(data)}")
        print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

        if n_samples > len(data):
            print(f'WARNING: n_samples ({n_samples}) > len(data) ({len(data)})')

        # Sample 'n_samples' examples
        data = data[:n_samples]

        # Save to cache (if requested)
        if self.config.dump_cache:
            self.dump_to_cache(data, data_split)

        return data

    def dump_to_cache(self, data, data_split):
        sorted_sources = "_".join(sorted(self.config.specific_sources))
        if self.config.specific_sources and self.name == 'the_pile':
            filename = f'{self.name}_{sorted_sources}'
        else:
            filename = self.name
        custom_datasets.dump_to_cache(data, self.cache_dir, data_split, filename,
                                      min_length=self.config.min_words, max_length=self.config.max_words,
                                      n_samples=self.config.n_samples, max_tokens=self.config.max_tokens)

def strip_newlines(text):
    """
        Strip newlines from each example; replace one or more newlines with a single space
    """
    return ' '.join(text.split())


def trim_to_shorter_length(text_a: str, text_b: str, max_length: int = None):
    """
        Truncate to shorter of o and s
    """
    shorter_length = min(len(text_a.split(' ')), len(text_b.split(' ')))
    if max_length is not None:
        shorter_length = min(shorter_length, max_length)
    text_a = ' '.join(text_a.split(' ')[:shorter_length])
    text_b = ' '.join(text_b.split(' ')[:shorter_length])
    return text_a, text_b


def truncate_to_substring(text: str, substring: str, idx_occurrence: int):
    """
        Truncate everything after the idx_occurrence occurrence of substring
    """
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


def pile_selection_utility(data, key: str, wanted_sources: List[str] = None):
    """
        Filter and select data corresponding to source, if requested.
    """
    if wanted_sources is None or len(wanted_sources) == 0:
        return data[key]
    wanted_data = []
    # Pick sources that match requested source
    for datum in data:
        # print(datum['meta']['pile_set_name'], wanted_sources)
        if datum['meta']['pile_set_name'] in wanted_sources:
            wanted_data.append(datum[key])
    return wanted_data


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])
