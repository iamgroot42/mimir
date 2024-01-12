#!/bin/bash
# Ideally, run after caching data with cache_data.json
for subset in "full_pile" #"wikimia" "arxiv_2021_01" "arxiv_2021_06" "arxiv_2022_01" "arxiv_2022_06" "arxiv_2023_01" "arxiv_2023_06" "c4"
do
    echo generating neighbors for $subset
    python run.py \
        --config configs/neighbor_gen_new.json \
        --specific_source $subset \
        --max_data 10000 \
        --n_samples 10000
done