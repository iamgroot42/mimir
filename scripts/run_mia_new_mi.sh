#!/bin/bash
version=new_mi
ngram=13
for subset in "arxiv" "hackernews" #"wikipedia_(en)" #"github"
    do
        python notebooks/new_mi_experiment.py \
            --experiment_name $version \
            --config configs/new_mi.json \
            --base_model "EleutherAI/pythia-12b-deduped" \
            --revision step99000 \
            --specific_source ${subset} #_ngram_${ngram}_\<0.8_truncated
    done