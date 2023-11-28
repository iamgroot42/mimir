#!/bin/bash
declare -A split_to_data=( [val]=/data/pile/val.jsonl [test]=/data/pile/test.jsonl [train]="/data/pile/train/00.jsonl /data/pile/train/01.jsonl" )

for split in ${!split_to_data[@]};
do 
    echo "creating $split subsets"
    python create_datasets.py \
        ${split_to_data[$split]} \
        --benchmark_dir doc_level_mia_pile_subsets/ \
        --split $split \
        --n_samples 100 \
        --full_doc

done


# python create_datasets.py \
#     /gscratch/h2lab/micdun/mimir/bff/deduped/full_pile/olc_ngram_13/out_freelaw.jsonl \
#     --benchmark_dir olc/sampled/freelaw \
#     --split test

# python create_datasets.py \
#     olc/sw_github/sampled.jsonl \
#     --benchmark_dir olc/sampled \
#     --provided_subset sw_github \
#     --split train

# python create_datasets.py \
#     /gscratch/h2lab/micdun/mimir/data/pile_subsets/wikipedia_\(en\)/train_raw.jsonl \
#     --benchmark_dir wikiMIA/sampled \
#     --provided_subset wikiMIA \
#     --min_len 128 \
#     --max_len 128 \
#     --split train

# python create_datasets.py \
#     /gscratch/h2lab/micdun/mimir/bff/deduped/ngram_13/olc_ngram_13/out_freelaw.jsonl \
#     --benchmark_dir olc/sampled/freelaw \
#     --split test

# python create_datasets.py \
#     /gscratch/h2lab/micdun/mimir/data/temporal_arxiv/arxiv_by_ts/arxiv_2021-02.jsonl \
#     --benchmark_dir temporal_arxiv/arxiv_2021-02/ \
#     --provided_subset arxiv_2021-02 \
#     --split train

# python create_datasets.py \
#     /gscratch/h2lab/micdun/bff/deduped/full_pile/ngram_13/0/test.jsonl.gz \
#     /gscratch/h2lab/micdun/bff/deduped/full_pile/ngram_13/1/test.jsonl.gz \
#     --benchmark_dir tokenized_test/ \
#     --ngram_metadata \
#     --min_len 200 \
#     --max_ngram_overlap 0.8 \
#     --provided_subset full_pile \
#     --tokenize \
#     --split test