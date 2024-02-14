#!/bin/bash
version=unified_mia_v5_temporal_wiki_v2_ne

for subset in "temporal_wiki_full"
do
    for model in "pythia-1.4b-deduped" "pythia-160m-deduped" "pythia-2.8b-deduped" "pythia-6.9b-deduped" #"pythia-12b-deduped"  
    do
        python run.py \
            --experiment_name $version \
            --config configs/mi.json \
            --base_model "EleutherAI/${model}" \
            --revision step99000 \
            --specific_source ${subset}
    done
done