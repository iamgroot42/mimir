#!/bin/bash
declare -A split_to_data=( [test]=/data/pile/test.jsonl [train]="/data/pile/train/00.jsonl /data/pile/train/01.jsonl" ) #[val]=/data/pile/val.jsonl

for split in ${!split_to_data[@]};
do 
    echo "creating $split subsets"
    python create_datasets.py \
        ${split_to_data[$split]} \
        --benchmark_dir full_pile/ \
        --provided_subset full_pile_10000 \
        --split $split \
        --n_samples 10000
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
#     /gscratch/h2lab/micdun/mimir/data/temporal_arxiv/processed_arxiv/arxiv_20210101000000_to_20210201000000.jsonl \
#     --benchmark_dir temporal_arxiv/arxiv_2021-01/ \
#     --provided_subset arxiv_2021-01 \
#     --split test

# python create_datasets.py \
#     /gscratch/h2lab/micdun/bff/deduped/full_pile/ngram_13/0/test.jsonl.gz \
#     /gscratch/h2lab/micdun/bff/deduped/full_pile/ngram_13/1/test.jsonl.gz \
#     --benchmark_dir tokenized_test/ \
#     --ngram_metadata \
#     --n_samples 10000 \
#     --min_len 200 \
#     --max_ngram_overlap 0.8 \
#     --provided_subset full_pile_10000 \
#     --tokenize \
#     --split test


# python create_datasets.py \
#     /gscratch/h2lab/micdun/bff/deduped/temporal_arxiv/ngram_13/arxiv_2020-06/0/test_raw.jsonl.gz \
#     /gscratch/h2lab/micdun/bff/deduped/temporal_arxiv/ngram_13/arxiv_2020-06/1/test_raw.jsonl.gz \
#     --benchmark_dir temporal_arxiv/sampled/ \
#     --ngram_metadata \
#     --n_samples 1000 \
#     --max_ngram_overlap 0.8 \
#     --provided_subset arxiv_2020-06 \
#     --split test

python create_datasets.py \
        /mmfs1/gscratch/h2lab/micdun/mimir/data/temporal_wiki/wikitext_latest_full.json \
        --benchmark_dir temporal_wiki \
        --provided_subset temporal_wiki \
        --split test \
        --n_samples 1000