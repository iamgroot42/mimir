# MIMIR

![MIMIR logo](assets/logo.png)

MIMIR - Python package for measuring memorization in LLMs.

Documentation is available [here](https://iamgroot42.github.io/mimir.github.io).

[![Tests](https://github.com/iamgroot42/mimir/actions/workflows/test.yml/badge.svg)](https://github.com/iamgroot42/mimir/actions/workflows/test.yml)
[![Documentation](https://github.com/iamgroot42/mimir/actions/workflows/documentation.yml/badge.svg)](https://github.com/iamgroot42/mimir/actions/workflows/documentation.yml)

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

The data we used for our experiments is available on [Hugging Face Datasets](https://huggingface.co/datasets/iamgroot42/mimir). You can either choose to either load the data directly from Hugging Face with the `load_from_hf` flag in the config (preferred), or download the `cache_100_200_....` folders into your `MIMIR_CACHE_PATH` directory.

## MIA experiments how to run

```
python run.py --config configs/mi.json
```

# Attacks

We include and implement the following attacks, as described in our paper.
- [Likelihood](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8429311) (`loss`). Works by simply using the likelihood of the target datapoint as score.
- [Reference-based](https://arxiv.org/abs/2004.15011) (`ref`). Normalizes likelihood score with score obtained from a reference model.
- [Zlib Entropy](https://www.usenix.org/system/files/sec21-carlini-extracting.pdf) (`zlib`). Uses the zlib compression size of a sample to approximate local difficulty of sample.
- [Neighborhood](https://aclanthology.org/2023.findings-acl.719/) (`ne`). Generates neighbors using auxiliary model and measures change in likelihood.
- [Min-K% Prob](https://swj0419.github.io/detect-pretrain.github.io/) (`min_k`). Uses k% of tokens with minimum likelihood for score computation.
- [Min-K%++](https://zjysteven.github.io/mink-plus-plus/) (`min_k++`). Uses k% of tokens with minimum *normalized* likelihood for score computation.
- [Gradient Norm](https://arxiv.org/abs/2402.17012) (`gradnorm`). Uses gradient norm of the target datapoint as score.

## Adding your own dataset

To extend the package for your own dataset, you can directly load your data inside `load_cached()` in `data_utils.py`, or add an additional if-else within `load()` in `data_utils.py` if it cannot be loaded from memory (or some source) easily. We will probably add a more general way to do this in the future.

## Adding your own attack

To add an attack, create a file for your attack (e.g. `attacks/my_attack.py`) and implement the interface described in `attacks/blackbox_attack.py`.
Then, add a name for your attack to the dictionary in `attacks/blackbox_attack.py`.

If you would like to submit your attack to the repository, please open a pull request describing your attack and the paper it is based on.

## Citation

If you use MIMIR in your research, please cite our paper:

```bibtex
@article{duan2024membership,
      title={Do Membership Inference Attacks Work on Large Language Models?}, 
      author={Michael Duan and Anshuman Suri and Niloofar Mireshghallah and Sewon Min and Weijia Shi and Luke Zettlemoyer and Yulia Tsvetkov and Yejin Choi and David Evans and Hannaneh Hajishirzi},
      year={2024},
      journal={arXiv:2402.07841},
}
```