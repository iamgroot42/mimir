"""
    Definitions for configurations.
"""

from dataclasses import dataclass
from typing import Optional, List
from simple_parsing.helpers import Serializable, field
from mimir.utils import get_cache_path, get_data_source


@dataclass
class ReferenceConfig(Serializable):
    """
    Config for attacks that use reference models.
    """
    models: List[str]
    """Reference model names"""


@dataclass
class NeighborhoodConfig(Serializable):
    """
    Config for neighborhood attack
    """
    model: str
    """Mask-filling model"""
    n_perturbation_list: List[int] = field(default_factory=lambda: [1, 10])
    """List of n_neighbors to try."""
    dump_cache: Optional[bool] = False
    "Dump neighbors data to cache? Exits program after dumping"
    load_from_cache: Optional[bool] = False
    """Load neighbors data from cache?"""
    # BERT-specific param
    original_tokenization_swap: Optional[bool] = True
    """Swap out token in original text with neighbor token, instead of re-generating text"""
    pct_swap_bert: Optional[float] = 0.05
    """Percentage of tokens per neighbor that are different from the original text"""
    neighbor_strategy: Optional[str] = "deterministic"
    """Strategy for generating neighbors. One of ['deterministic', 'random']. Deterministic uses only one-word neighbors"""
    # T-5 specific hyper-parameters
    span_length: Optional[int] = 2
    """Span length for neighborhood attack"""
    random_fills_tokens: Optional[bool] = False
    """Randomly fill tokens?"""
    random_fills: Optional[bool] = False
    """Randomly fill?"""
    pct_words_masked: Optional[float] = 0.3
    """Percentage masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))"""
    buffer_size: Optional[int] = 1
    """Buffer size"""
    top_p: Optional[float] = 1.0
    """Use tokens (minimal set) with cumulative probability of <=top_p"""
    max_tries: Optional[int] = 100
    """Maximum number of trials in finding replacements for masked tokens"""
    ceil_pct: Optional[bool] = False
    """Apply ceil operation on span length calculation?"""

    def __post_init__(self):
        if self.dump_cache and self.load_from_cache:
            raise ValueError("Cannot dump and load cache at the same time")


@dataclass
class EnvironmentConfig(Serializable):
    """
    Config for environment-specific parameters
    """
    cache_dir: Optional[str] = None
    """Path to cache directory"""
    data_source: Optional[str] = None
    """Path where data is stored"""
    device: Optional[str] = 'cuda:0'
    """Device (GPU) to load main model on"""
    device_map: Optional[str] = None
    """Configuration for device map if needing to split model across gpus"""
    device_aux: Optional[str] = "cuda:1"
    """Device (GPU) to load any auxiliary model(s) on"""
    compile: Optional[bool] = True
    """Compile models?"""
    int8: Optional[bool] = False
    """Use int8 quantization?"""
    half: Optional[bool] = False
    """Use half precision?"""
    results: Optional[str] = "results"
    """Path for saving final results"""
    tmp_results: Optional[str] = "tmp_results"

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = get_cache_path()
        if self.data_source is None:
            self.data_source = get_data_source()


@dataclass
class OpenAIConfig(Serializable):
    """
    Config for OpenAI calls
    """
    key: str
    """OpenAI API key"""
    model: str
    """Model name"""


@dataclass
class ExperimentConfig(Serializable):
    """
    Config for attacks
    """
    experiment_name: str
    """Name for the experiment"""
    base_model: str
    """Base model name"""
    dataset_member: str
    """Dataset source for members"""
    dataset_nonmember: str
    """Dataset source for nonmembers"""
    output_name: str = None
    """Output name for sub-directory."""
    dataset_nonmember_other_sources: Optional[List[str]] = field(
        default_factory=lambda: None
    )
    """Dataset sources for nonmembers for which metrics will be computed, using the thresholds derived from the main member/nonmember datasets"""
    pretokenized: Optional[bool] = False
    """Is the data already pretokenized"""
    revision: Optional[str] = None
    """Model revision to use"""
    presampled_dataset_member: Optional[str] = None
    """Path to presampled dataset source for members"""
    presampled_dataset_nonmember: Optional[str] = None
    """Path to presampled dataset source for non-members"""
    token_frequency_map: Optional[
        str
    ] = None  # TODO: Handling auxiliary data structures
    """Path to a pre-computed token frequency map"""
    dataset_key: Optional[str] = None
    """Dataset key"""
    specific_source: Optional[str] = None
    """Specific sub-source to focus on. Only valid for the_pile"""
    full_doc: Optional[bool] = False  # TODO: refactor full_doc design?
    """Determines whether MIA will be performed over entire doc or not"""
    max_substrs: Optional[int] = 20
    """If full_doc, determines the maximum number of sample substrs to evaluate on"""
    dump_cache: Optional[bool] = False
    "Dump data to cache? Exits program after dumping"
    load_from_cache: Optional[bool] = False
    """Load data from cache?"""
    load_from_hf: Optional[bool] = True
    """Load data from HuggingFace?"""
    blackbox_attacks: Optional[List[str]] = field(
        default_factory=lambda: None
    )  # Can replace with "default" attacks if we want
    """List of attacks to evaluate"""
    tokenization_attack: Optional[bool] = False
    """Run tokenization attack?"""
    quantile_attack: Optional[bool] = False
    """Run quantile attack?"""
    n_samples: Optional[int] = 200
    """Number of records (member and non-member each) to run the attack(s) for"""
    max_tokens: Optional[int] = 512
    """Consider samples with at most these many tokens"""
    max_data: Optional[int] = 5_000
    """Maximum samples to load from data before processing. Helps with efficiency"""
    min_words: Optional[int] = 100
    """Consider documents with at least these many words"""
    max_words: Optional[int] = 200
    """Consider documents with at most these many words"""
    max_words_cutoff: Optional[bool] = True
    """Is max_words a selection criteria (False), or a cutoff added on text (True)?"""
    batch_size: Optional[int] = 50
    """Batch size"""
    chunk_size: Optional[int] = 20
    """Chunk size"""
    scoring_model_name: Optional[str] = None
    """Scoring model (if different from base model)"""
    top_k: Optional[int] = 40
    """Consider only top-k tokens"""
    do_top_k: Optional[bool] = False
    """Use top-k sampling?"""
    top_p: Optional[float] = 0.96
    """Use tokens (minimal set) with cumulative probability of <=top_p"""
    do_top_p: Optional[bool] = False
    """Use top-p sampling?"""
    pre_perturb_pct: Optional[float] = 0.0
    """Percentage of tokens to perturb before attack"""
    pre_perturb_span_length: Optional[int] = 5
    """Span length for pre-perturbation"""
    tok_by_tok: Optional[bool] = False
    """FPRs at which to compute TPR"""
    fpr_list: Optional[List[float]] = field(default_factory=lambda: [0.001, 0.01])
    """Process data token-wise?"""
    random_seed: Optional[int] = 0
    """Random seed"""
    ref_config: Optional[ReferenceConfig] = None
    """Reference model config"""
    neighborhood_config: Optional[NeighborhoodConfig] = None
    """Neighborhood attack config"""
    env_config: Optional[EnvironmentConfig] = None
    """Environment config"""
    openai_config: Optional[OpenAIConfig] = None
    """OpenAI config"""

    def __post_init__(self):
        if self.dump_cache and (self.load_from_cache or self.load_from_hf):
            raise ValueError("Cannot dump and load cache at the same time")

        if self.neighborhood_config:
            if (
                self.neighborhood_config.dump_cache
                or self.neighborhood_config.load_from_cache
            ) and not (self.load_from_cache or self.dump_cache or self.load_from_hf):
                raise ValueError(
                    "Using dump/load for neighborhood cache without dumping/loading main cache does not make sense"
                )

            if self.neighborhood_config.dump_cache and (self.neighborhood_config.load_from_cache or self.load_from_hf):
                raise ValueError("Cannot dump and load neighborhood cache at the same time")    
