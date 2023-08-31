"""
    Definitions for configurations.
"""

from dataclasses import dataclass
from typing import Optional, List
from simple_parsing.helpers import Serializable, field


@dataclass
class ReferenceConfig(Serializable):
    """
        Config for attacks that use reference models.
    """
    model: str
    """Reference model name"""


@dataclass
class NeighborhoodConfig(Serializable):
    """
        Config for neighborhood attack
    """
    model: str
    """Mask-filling model"""
    n_perturbation_list: List[int] = field(default_factory=lambda: [1, 10])
    """List of n_neighbors to try."""
    # T-5 specific hyper-parameters
    span_length: Optional[int] = 2
    """Span length for neighborhood attack"""
    random_fills_tokens: Optional[bool] = False
    """Randomly fill tokens?"""
    random_fills: Optional[bool] = False
    """Randomly fill?"""
    pct_words_masked: Optional[float] = 0.3
    """Percentage masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))"""
    n_perturbation_rounds: Optional[int] = 1
    """Number of perturbation-round trials"""
    buffer_size: Optional[int] = 1
    """Buffer size"""
    top_p: Optional[float] = 1.0
    """Use tokens (minimal set) with cumulative probability of <=top_p"""
    max_tries: Optional[int] = 100
    """Maximum number of trials in finding replacements for masked tokens"""
    ceil_pct: Optional[bool] = False
    """Apply ceil operation on span length calculation?"""



@dataclass
class EnvironmentConfig(Serializable):
    """
        Config for environment-specific parameters
    """
    cache_dir: Optional[str] = "/trunk/model-hub"
    """Path to cache directory"""
    data_source: Optional[str] = "/trunk/datasets/niloofar"
    """Path where data is stored"""
    device: Optional[str] = 'cuda:0'
    """Device (GPU) to load main model on"""
    device_aux: Optional[str] = 'cuda:1'
    """Device (GPU) to load any auxiliary model(s) on"""
    compile: Optional[bool] = True
    """Compile models?"""
    int8: Optional[bool] = False
    """Use int8 quantization?"""
    half: Optional[bool] = False
    """Use half precision?"""
    results: Optional[str] = 'results'
    """Path for saving final results"""
    tmp_results: Optional[str] = 'tmp_results'


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
class ExtractionConfig(Serializable):
    """
        Config for model-extraction
    """
    prompt_len: Optional[int] = 30
    """Prompt length"""


@dataclass
class ExperimentConfig(Serializable):
    """
        Config for attacks
    """
    base_model: str
    """Base model name"""
    dataset_member: str
    """Dataset source for members"""
    dataset_nonmember: str
    """Dataset source for nonmembers"""
    output_name: Optional[str] = None
    """Output name for sub-directory. Defaults to nothing"""
    specific_sources: Optional[List[str]] = None
    """List of specific sub-sources to focus on. Only valid for the_pile"""
    dump_cache: Optional[bool] = False
    "Dump data to cache? Exits program after dumping"
    load_from_cache: Optional[bool] = False
    """Load data from cache?"""
    baselines_only: Optional[bool] = False
    """Evaluate only baselines?"""
    skip_baselines: Optional[bool] = False
    """Skip baselines?"""
    n_samples: Optional[int] = 200
    """Number of records (member and non-member each) to run the attack(s) for"""
    max_tokens: Optional[int] = 512
    """Consider samples with at most these many tokens"""
    max_data: Optional[int] = 5_000
    """Maximum samples to load from data before processing. Helps with efficiency"""
    min_words: Optional[int] = 100
    """Consider documents with at least these many words"""
    max_words: Optional[int] = 150
    """Consider documents with at most these many words"""
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
    """Process data token-wise?"""
    ref_config: Optional[ReferenceConfig] = None
    """Reference model config"""
    neighborhood_config: Optional[NeighborhoodConfig] = None
    """Neighborhood attack config"""
    env_config: Optional[EnvironmentConfig] = None
    """Environment config"""
    openai_config: Optional[OpenAIConfig] = None
    """OpenAI config"""
    extraction_config: Optional[ExtractionConfig] = None
    """Extraction config"""
    