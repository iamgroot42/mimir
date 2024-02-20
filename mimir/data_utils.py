"""
    Datasets and data-processing utilities
"""
import datasets
import numpy as np
import os
import mimir.custom_datasets as custom_datasets
from mimir.config import ExperimentConfig
from nltk.tokenize import WhitespaceTokenizer


class Data:
    """
    Data class to load and cache datasets.
    """
    def __init__(self, name,
                 config: ExperimentConfig,
                 presampled: str = None,
                 name_key_mapping: dict = {"the_pile": "text", "xsum": "document"}):
        self.name_key_mapping = name_key_mapping
        self.config = config
        self.name = name
        self.presampled = presampled
        self.key = (
            config.dataset_key
            if config.dataset_key
            else self.name_key_mapping.get(name, None)
        )
        if self.key is None:
            raise ValueError(
                f"Key for dataset {name} not provided or found inname_key_mapping"
            )
        self.cache_dir = self.config.env_config.cache_dir

    def load_neighbors(
        self,
        train: bool,
        num_neighbors: int,
        model: str = "bert",
        in_place_swap: bool = False,
    ):
        """
        Load neighbors from cache (local or from HF)
        """
        data_split = "train" if train else "test"
        data_split += "_neighbors"
        filename = self._get_name_to_save() + "_neighbors_{}_{}".format(
            num_neighbors, model
        )
        if in_place_swap:
            filename += "_in_place_swap"
        data = custom_datasets.load_cached(
            self.cache_dir,
            data_split,
            filename,
            min_length=self.config.min_words,
            max_length=self.config.max_words,
            n_samples=self.config.n_samples,
            max_tokens=self.config.max_tokens,
            load_from_hf=self.config.load_from_hf
        )
        return data

    def dump_neighbors(
        self,
        data,
        train: bool,
        num_neighbors: int,
        model: str = "bert",
        in_place_swap: bool = False,
    ):
        """
        Dump neighbors to cache local cache.
        """
        data_split = "train" if train else "test"
        data_split += "_neighbors"
        filename = self._get_name_to_save() + "_neighbors_{}_{}".format(
            num_neighbors, model
        )
        if in_place_swap:
            filename += "_in_place_swap"
        custom_datasets.dump_to_cache(
            data,
            self.cache_dir,
            data_split,
            filename,
            min_length=self.config.min_words,
            max_length=self.config.max_words,
            n_samples=self.config.n_samples,
            max_tokens=self.config.max_tokens,
        )

    def load(self, train: bool, mask_tokenizer=None, specific_source: str = None):
        data_split = "train" if train else "test"
        n_samples = self.config.n_samples

        # Load from numpy file storing pretokenized sample in a 2d array of shape (num_samples, num_tokens_per_sample)
        if self.config.pretokenized:
            assert self.presampled
            # TODO: Pretokenized full documents (split into substrs) is not currently supported
            assert not self.config.full_doc
            data = np.load(self.presampled)
            return data
        elif (self.config.load_from_cache or self.config.load_from_hf):
            # Load from cache, if requested
            filename = self._get_name_to_save()
            data = custom_datasets.load_cached(
                self.cache_dir,
                data_split,
                filename,
                min_length=self.config.min_words,
                max_length=self.config.max_words,
                n_samples=self.config.n_samples,
                max_tokens=self.config.max_tokens,
                load_from_hf=self.config.load_from_hf
            )
            return data
        else:
            if self.presampled or self.config.full_doc:
                print("using presampled data")
                data = datasets.load_dataset(
                    "json",
                    data_files=self.presampled,
                    split=f"train",
                    cache_dir=self.cache_dir,
                )[self.key]
            elif self.name in custom_datasets.DATASETS:
                data = custom_datasets.load(self.name)
            elif self.name == "the_pile":
                min_load = max(10000, self.config.max_data)
                data = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(
                        self.config.env_config.data_source,
                        "pile/00.jsonl.zst" if train else "pile/test.jsonl.zst",
                    ),
                    cache_dir=self.cache_dir,
                    split=f"train[:{min_load}]",
                )
                specific_source_use = (
                    self.config.specific_source
                    if specific_source is None
                    else specific_source
                )
                data = pile_selection_utility(
                    data, self.key, wanted_source=specific_source_use
                )
            elif "human" in self.name:
                data = datasets.load_dataset(
                    self.name, split=f"train[:100]", cache_dir=self.cache_dir
                )[self.key]
            elif "nthngdy" in self.name:
                data = datasets.load_dataset(
                    self.name, split="test", cache_dir=self.cache_dir
                )[self.key]
            else:
                data = datasets.load_dataset(
                    self.name, split=f"train", cache_dir=self.cache_dir
                )[self.key]

        if not self.config.full_doc:
            # get unique examples
            # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
            # then take just the examples that are <= 512 tokens (for the mask model)
            # then generate n_samples samples
            wsp_tokenizer = WhitespaceTokenizer()

            # remove duplicates from the data
            data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

            whitespace_tokenized_spans = [
                (x, list(wsp_tokenizer.span_tokenize(x))) for x in data
            ]

            # Pick samples with at least self.config.min_words words
            whitespace_tokenized_spans = [
                x
                for x in whitespace_tokenized_spans
                if len(x[1]) >= self.config.min_words
            ]
            if len(whitespace_tokenized_spans) == 0:
                raise ValueError("No examples with length >= min_words")

            if self.config.max_words_cutoff:
                last_spans = [
                    x[1][min(self.config.max_words, len(x[1])) - 1][1]
                    for x in whitespace_tokenized_spans
                ]
                data = [
                    x[0][:y] for x, y in zip(whitespace_tokenized_spans, last_spans)
                ]
            else:
                data = [
                    x[0]
                    for x in whitespace_tokenized_spans
                    if len(x[1]) < self.config.max_words
                ]
                if len(data) == 0:
                    raise ValueError("No examples with length < max_words")

            # TODO: why shuffle
            # random.seed(0)
            # random.shuffle(data)

            data = data[: self.config.max_data]

            # If there is mask tokenizer, keep only examples with <= 512 tokens according to mask_tokenizer
            # this step has the extra effect of removing examples with low-quality/garbage content
            if mask_tokenizer:
                tokenized_data = mask_tokenizer(data)
                new_data = []
                for i, (x, y) in enumerate(zip(data, tokenized_data["input_ids"])):
                    if len(y) <= self.config.max_tokens:
                        new_data.append(x)
                    else:
                        print(
                            "Trimming text to nearest word that fits within mask tokenizer window"
                        )
                        max_token_char_span = tokenized_data.token_to_chars(
                            i, self.config.max_tokens - 1
                        )
                        x = x[: max_token_char_span.end]
                        token_truncated_word_spans = list(
                            wsp_tokenizer.span_tokenize(x)
                        )

                        # Pop off the last "word" since it may be a word piece
                        second_last_span = token_truncated_word_spans[-2]
                        x = x[: second_last_span[1]]

                        new_len = len(mask_tokenizer(x)["input_ids"])
                        assert new_len <= self.config.max_tokens
                        new_data.append(x)
                data = new_data

            # print stats about remainining data
            print(f"Total number of samples: {len(data)}")
            print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

            if n_samples > len(data):
                print(f"WARNING: n_samples ({n_samples}) > len(data) ({len(data)})")

        # Sample 'n_samples' examples
        data = data[:n_samples]

        # Save to cache (if requested)
        if self.config.dump_cache:
            self.dump_to_cache(data, data_split)

        return data

    def dump_to_cache(self, data, data_split):
        filename = self._get_name_to_save()
        custom_datasets.dump_to_cache(
            data,
            self.cache_dir,
            data_split,
            filename,
            min_length=self.config.min_words,
            max_length=self.config.max_words,
            n_samples=self.config.n_samples,
            max_tokens=self.config.max_tokens,
        )

    def _get_name_to_save(self):
        if self.config.specific_source and self.name == "the_pile":
            processed_source = sourcename_process(self.config.specific_source)
            filename = f"{self.name}_{processed_source}"
        else:
            filename = self.name
        return filename


def strip_newlines(text):
    """
    Strip newlines from each example; replace one or more newlines with a single space
    """
    return " ".join(text.split())


def trim_to_shorter_length(text_a: str, text_b: str, max_length: int = None):
    """
    Truncate to shorter of o and s
    """
    shorter_length = min(len(text_a.split(" ")), len(text_b.split(" ")))
    if max_length is not None:
        shorter_length = min(shorter_length, max_length)
    text_a = " ".join(text_a.split(" ")[:shorter_length])
    text_b = " ".join(text_b.split(" ")[:shorter_length])
    return text_a, text_b


def truncate_to_substring(text: str, substring: str, idx_occurrence: int):
    """
    Truncate everything after the idx_occurrence occurrence of substring
    """
    assert idx_occurrence > 0, "idx_occurrence must be > 0"
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


def pile_selection_utility(data, key: str, wanted_source: str = None):
    """
    Filter and select data corresponding to source, if requested.
    """
    if wanted_source is None:
        return data[key]
    wanted_data = []
    # Pick sources that match requested source
    for datum in data:
        if datum["meta"]["pile_set_name"] == wanted_source:
            wanted_data.append(datum[key])
    return wanted_data


def sourcename_process(x: str):
    """
        Helper function to process source name.
    """
    return x.replace(" ", "_").replace("-", "_").lower()


def drop_last_word(text):
    """
        Drop the last word from a given text.
    """
    return " ".join(text.split(" ")[:-1])
