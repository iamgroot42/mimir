#!/bin/bash
version=unified_mia_v5_olmo_test_only

for model in "OLMo-1B"
do

    for subset in "dolma_wikipedia" "dolma_s2" # "c4"
    do
        python3.9 run.py \
            --experiment_name $version \
            --config configs/olmo.json \
            --base_model "allenai/$model" \
            --specific_source ${subset} \
            --load_from_cache true \
            --load_from_hf false \
            --n_samples 1000
    done
done
