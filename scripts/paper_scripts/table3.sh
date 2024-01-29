#!/bin/bash
version=table3

for ngram in "7" "13"
do
    for subset in "wikipedia_(en)" "github" "pubmed_central" "pile_cc" "arxiv"
    do
        knocky python run.py \
            --config configs/mi.json \
            --revision step99000 \
            --base_model "EleutherAI/pythia-12b-deduped" \
            --specific_source ${subset}_ngram_${ngram}_\<0.2_truncated \
            --output_name $version
    done
done

