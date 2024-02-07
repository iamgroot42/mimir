"""
    Implementation of tokenization-based attack (unpublished)
"""
from llm_di.model_utils import AlternativeTokenizationsGenerator
from llm_di.config import TokenizerConfig
import numpy as np


def token_attack(model, doc):
    # Create config
    config = TokenizerConfig(stochastic=True, num_tokenizations=100, max_tokens=2048)
    # Wrap our model and tokenizer into object)
    attacker = AlternativeTokenizationsGenerator(model.tokenizer, config)

    # Get alt tokenizations
    alt_toks = attacker.get_tokenizations(doc)
    # Covert to strings
    alt_strings = model.tokenizer.batch_decode(alt_toks)
    # Get scores
    scores = model.get_lls(alt_strings)
    # return np.mean(scores)
    return np.array(scores)
