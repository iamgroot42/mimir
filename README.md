# MIMIR

![MIMIR logo](assets/logo.png)

MIMIR - Python package for measuring memorization in LLMs. 

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

## Setting environment variables

You can either provide the following environment variables, or pass them via your config/CLI:

```
MIMIR_CACHE_PATH: Path to cache directory
MIMIR_DATA_SOURCE: Path to data directory
```

## Using cached data

To use the exact same member/non-member records, unzip `cache_100_200_1000_512.zip` (at least 100, at most 200 words, 1000 samples) into your `cache_dir`. For this scenario, make sure you use the `--load_from_cache` flag.

## MIA experiments how to run

```
python run.py --config configs/mi_readme.json
```

# Attack implementations

We include and implement the following attacks, as described in our paper.
- [Likelihood](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8429311), available as `loss`. Works by simply using the likelihood of the target datapoint as score.
- [Reference-based](https://arxiv.org/abs/2004.15011), available as `ref`. Normalizes likelihood score with score obtained from a reference model.
- [Zlib Entropy](https://www.usenix.org/system/files/sec21-carlini-extracting.pdf), available as `zlib`. Uses the zlib compression size of a sample to approximate local difficulty of sample.
- [Min-k% Prob](https://swj0419.github.io/detect-pretrain.github.io/), available as `min_k`. Uses k% of tokens with minimum likelihood for score computation.
- [Neighborhood](https://aclanthology.org/2023.findings-acl.719/), available as `ne`. Generates neighbors using auxiliary model and measures change in likelihood.
- [Quantile](https://neurips.cc/virtual/2023/poster/70232), available as `quantile`. Trains meta-classifier for predicting quantile of loss.

## Adding your own attack

To add an attack, create a file for your attack (e.g. `attacks/my_attack.py`) and implement the interface described in `attacks/blackbox_attack.py`.
Then, add a name for your attack to the dictionary in `attacks/blackbox_attack.py`.

If you would like to submit your attack to the repository, please open a pull request describing your attack and the paper it is based on.