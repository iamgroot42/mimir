#!/bin/bash
# Ideally, run after caching data with cache_data.json
for subset in "arxiv_2020-08" "arxiv_2021-01" "arxiv_2021-06" "arxiv_2022-01" "arxiv_2022-06" "arxiv_2023-01" "arxiv_2023-06" #"temporal_wiki_full" #"wikimia" "c4"
do
    echo generating neighbors for $subset
    python run.py \
        --config configs/neighbor_gen_new.json \
        --specific_source $subset
done