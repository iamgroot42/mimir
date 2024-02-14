#!/bin/bash
version=unified_mia_v5_temporal_arxiv_v3_ne

for source in "arxiv_2020-08" "arxiv_2021-01" "arxiv_2021-06" "arxiv_2022-06" "arxiv_2023-06"  "arxiv_2023-01" "arxiv_2022-01" #"arxiv_2019-01" "arxiv_2020-01"
do
    python run.py \
        --experiment_name $version \
        --config configs/single_gpu_mi.json \
        --base_model "EleutherAI/pythia-12b-deduped" \
        --revision step99000 \
        --specific_source $source
done