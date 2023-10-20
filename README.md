# MIMIR

MIMIR- Python package for measuring memorization in LLMs. 

## Instructions

First install the python dependencies
```
pip install -r requirements.txt
```

Then, install our package

```
pip install -e .
```

To use, run the scripts in `scripts/bash`

**Note: Intermediate results are saved in `tmp_results/` and `tmp_results_cross/` for bash scripts. If your experiment completes successfully, the results will be moved into the `results/` and `results_cross/` directory.**

# Setting environment variables

You can either provide the following environment variables, or pass them via your config/CLI:

```
MIMIR_CACHE_PATH: Path to cache directory
MIMIR_DATA_SOURCE: Path to data directory
```

# Using cached data

To use the exact same member/non-member records, unzip `cache_100_200_1000_512.zip` (at least 100, at most 200 words, 1000 samples) into your `cache_dir`. For thie scenario, make sure you use the `--load_from_cache` flag.

# MIA experiments how to run

```
python run.py --config configs/mi_readme.json
```
