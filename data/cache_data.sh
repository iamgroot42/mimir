#!/bin/bash
ngram=7
for subset in "pile_cc" "github" "wikipedia_(en)" "hackernews" "pubmed_central" "arxiv" "dm_mathematics" "books3"
do
    echo caching data for $subset
    python run.py \
        --config configs/cache_data.json \
        --presampled_dataset_member "/gscratch/h2lab/micdun/mimir/data/doc_level_mia_pile_subsets/$subset/train_text.jsonl" \
        --presampled_dataset_nonmember "/gscratch/h2lab/micdun/mimir/data/doc_level_mia_pile_subsets/0.0-0.8/$subset/test_text.jsonl" \
        --dataset_key substr_samples \
        --specific_source ${subset}_sampled_substr \
        --n_samples 100 \
        --full_doc true
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
