#!/bin/bash
version=table11
ngram=13

for model in "gpt-neo-125m" "gpt-neo-1.3B" "gpt-neo-2.7B"
do
    for subset in "wikipedia_(en)" "github" "pile_cc" "pubmed_central" "arxiv" "dm_mathematics" "hackernews"
    do
        knocky python run.py \
            --config configs/mi.json \
            --base_model "EleutherAI/${model}" \
            --specific_source ${subset}_ngram_${ngram}_\<0.8_truncated \
            --output_name $version
    done
    # full_pile specifically
    python run.py \
            --config configs/mi.json \
            --base_model "EleutherAI/${model}" \
            --specific_source "full_pile" \
            --output_name $version
            --n_samples 10000
done

