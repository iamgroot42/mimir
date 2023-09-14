#!/bin/bash
for subset in "pile_cc" "github" "wikipedia_(en)" "pubmed_central" "arxiv"
do
    echo caching data for $subset
    python run.py \
        --config configs/cache_data.json \
        --presampled_dataset_member "/gscratch/h2lab/micdun/mimir/data/pile_subsets/$subset/train_raw.jsonl" \
        --presampled_dataset_nonmember "/gscratch/h2lab/micdun/mimir/data/pile_subsets/$subset/test_raw.jsonl" \
        --specific_source $subset
done

