#!/bin/bash
version=table1
ngram=13

for model in "pythia-160m" "pythia-1.4b" "pythia-2.8b" "pythia-6.9b" "pythia-12b"
do
    for subset in "wikipedia_(en)" "github" "pile_cc" "pubmed_central" "arxiv" "dm_mathematics" "hackernews"
    do
        knocky python run.py \
            --config configs/mi.json \
            --revision step99000 \
            --base_model "EleutherAI/${model}-deduped" \
            --specific_source ${subset}_ngram_${ngram}_\<0.8_truncated \
            --output_name $version
    done
    # full_pile specifically
    python run.py \
            --config configs/mi.json \
            --revision step99000 \
            --base_model "EleutherAI/${model}-deduped" \
            --specific_source "full_pile" \
            --output_name $version
            --n_samples 10000
done

