#!/bin/bash
# Ideally, run after caching data with cache_data.json
for subset in "wikipedia_(en)" "pile_cc" "pubmed_central" "hackernews" "arxiv" "dm_mathematics" "github"
do
    echo generating neighbors for $subset
    python run.py \
        --config configs/neighbor_gen_new.json \
        --specific_source "${subset}_ngram_13_<0.2_truncated" \
        --max_data 1000 \
        --n_samples 1000

    python run.py \
        --config configs/neighbor_gen_new.json \
        --specific_source "${subset}_ngram_7_<0.2_truncated" \
        --max_data 1000 \
        --n_samples 1000
done

echo generating standard neighbors for pubmed_central
python run.py \
        --config configs/neighbor_gen_new.json \
        --specific_source "pubmed_central_ngram_13_<0.8_truncated" \
        --max_data 1000 \
        --n_samples 1000