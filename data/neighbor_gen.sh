#!/bin/bash
# Ideally, run after caching data with cache_data.json
for subset in "pile_cc" "github" "wikipedia_(en)" "pubmed_central" "arxiv"
do
    echo generating neighbors for $subset
    python run.py \
        --config configs/neighbor_gen.json \
        --specific_source $subset
done