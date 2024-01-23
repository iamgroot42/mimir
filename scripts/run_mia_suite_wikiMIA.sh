#!/bin/bash
version=unified_mia_v5_temporal_wiki_ne

for subset in "temporal_wiki_full"
do
    for model in "pythia-12b-deduped" # "pythia-1.4b-deduped" "pythia-160m-deduped" "pythia-2.8b-deduped" "pythia-6.9b-deduped"
    do
        python run.py \
            --config configs/single_gpu_mi.json \
            --base_model "EleutherAI/${model}" \
            --revision step99000 \
            --specific_source ${subset} \
            --output_name $version \
            --n_samples 1000 \
            --baselines_only true
    done
done