"""
    Misc utils
"""
import os


CACHE_PATH = os.environ.get('MIMIR_CACHE_PATH', None)
if CACHE_PATH is None:
    raise ValueError('MIMIR_CACHE_PATH environment variable not set')

DATA_SOURCE = os.environ.get('MIMIR_DATA_SOURCE', None)
if DATA_SOURCE is None:
    raise ValueError('MIMIR_DATA_SOURCE environment variable not set')
