#!/bin/bash
ngram=7
for subset in "full_pile"
do
    echo caching data for $subset
    python run.py \
        --config configs/cache_data.json \
        --presampled_dataset_member "/gscratch/h2lab/micdun/mimir/data/full_pile/full_pile_10000/train_raw.jsonl" \
        --presampled_dataset_nonmember "/gscratch/h2lab/micdun/mimir/data/full_pile/0.0-0.8/full_pile_10000/test_raw.jsonl" \
        --specific_source $subset \
        --max_data 10000 \
        --n_samples 10000
done
 #"/gscratch/h2lab/micdun/mimir/data/ngram_overlap_thresholded_pile_subsets/truncated+ngram_$ngram/0.0-0.2/$subset/test_raw.jsonl"


# python run.py \
#         --config configs/cache_data.json \
#         --presampled_dataset_member "data/c4/sampled/c4/train_raw.jsonl" \
#         --presampled_dataset_nonmember "data/c4/sampled/c4/test_raw.jsonl" \
#         --specific_source c4

# mimir/data/olc/pile_law+pd_law/pd_law/train_raw.jsonl
# python run.py \
#         --config configs/cache_data.json \
#         --presampled_dataset_member "/gscratch/h2lab/micdun/mimir/data/olc/pile_law+pd_law/pd_law/train_raw.jsonl" \
#         --presampled_dataset_nonmember "/gscratch/h2lab/micdun/mimir/data/ngram_overlap_thresholded_pile_subsets/truncated+ngram_13/0.0-0.2/freelaw/test_raw.jsonl" \
#         --specific_source pd_law

# python run.py \
#         --config configs/cache_data.json \
#         --presampled_dataset_member "/gscratch/h2lab/micdun/mimir/data/wikiMIA/sampled/wikiMIA/train_raw.jsonl" \
#         --presampled_dataset_nonmember "/gscratch/h2lab/micdun/mimir/data/wikiMIA/sampled/wikiMIA/test_raw.jsonl" \
#         --specific_source wikiMIA
