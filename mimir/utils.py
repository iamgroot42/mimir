"""
utils.py
This module provides utility functions.

Environment Variables:
    MIMIR_CACHE_PATH: The path to the cache directory. This should be set in the environment.
    MIMIR_DATA_SOURCE: The data source for the MIMIR project. This should be set in the environment.
"""

import os
import random
import torch as ch
import numpy as np

# Read environment variables
CACHE_PATH = os.environ.get('MIMIR_CACHE_PATH', None)
DATA_SOURCE = os.environ.get('MIMIR_DATA_SOURCE', None)


def fix_seed(seed: int = 0):
    """
    Fix seed for reproducibility.

    Parameters:
        seed (int): The seed to set. Default is 0.
    """
    ch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_cache_path():
    """
    Get path to cache directory.
    Returns:
        str: path to cache directory

    Raises:
        ValueError: If the MIMIR_CACHE_PATH environment variable is not set.
    """
    if CACHE_PATH is None:
        raise ValueError('MIMIR_CACHE_PATH environment variable not set')
    return CACHE_PATH


def get_data_source():
    """
    Get path to data source directory.
    Returns:
        str: path to data source directory

    Raises:
        ValueError: If the MIMIR_DATA_SOURCE environment variable is not set.
    """
    if DATA_SOURCE is None:
        raise ValueError('MIMIR_DATA_SOURCE environment variable not set')
    return DATA_SOURCE
