#!/bin/bash
version=unified_mia_v5_temporal_arxiv

for source in "arxiv_2019_01" "arxiv_2019_06" "arxiv_2020_01" "arxiv_2020_06" #"arxiv_2021_01" "arxiv_2021_06" "arxiv_2022_01" "arxiv_2022_06" "arxiv_2023_01" "arxiv_2023_06"
do
    python run.py \
        --config configs/mi.json \
        --base_model "EleutherAI/pythia-12b-deduped" \
        --revision step99000 \
        --specific_source $source \
        --output_name $version \
        --baselines_only true \
        --blackbox_attacks loss+ref+zlib+min_k
done