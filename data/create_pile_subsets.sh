#!/bin/bash
# subsets=(
#     "pubmed_central"
#     "books3"
#     "arxiv"
#     "freelaw"
#     "stackexchange"
#     "uspto_backgrounds"
#     "gutenberg_(pg_19)"
#     "opensubtitles"
#     "dm_mathematics"
#     "ubuntu_irc"
#     "bookcorpus2"
#     "europarl"
#     "hackernews"
#     "youtubesubtitles"
#     "philpapers"
#     "nih_exporter"
#     "enron_emails"
# )
# subsets
declare -A split_to_data=( [val]=/data/pile/val.jsonl [test]=/data/pile/test.jsonl [train]="/data/pile/train/00.jsonl /data/pile/train/01.jsonl" )

for split in ${!split_to_data[@]};
do 
    echo "creating $split subsets"
    python create_datasets.py \
        ${split_to_data[$split]} \
        --benchmark_dir pile_subsets/ \
        --split $split
done
