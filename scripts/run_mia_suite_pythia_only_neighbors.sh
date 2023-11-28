#!/bin/bash
version=unified_mia_v4_ne
baselines_only=$1
skip_baselines=$2

# non-deduped models
for model in "pythia-160m" "pythia-1.4b" "pythia-2.8b" "pythia-6.9b" "pythia-12b"
do
    for subset in "pile_cc" "github" "wikipedia_(en)" "pubmed_central" "arxiv"
    do
        python run.py \
            --config configs/mi.json \
            --base_model "EleutherAI/${model}" \
            --specific_source "${subset}" \
            --output_name $version \
            --skip_baselines true
    done
done

# deduped models
# TODO: refactor this into one loop
for model in "pythia-160m-deduped" "pythia-1.4b-deduped" "pythia-2.8b-deduped" "pythia-6.9b-deduped" "pythia-12b-deduped"
do
    for subset in "pile_cc" "github" "wikipedia_(en)" "pubmed_central" "arxiv"
    do
        python run.py \
            --config configs/mi.json \
            --base_model "EleutherAI/${model}" \
            --revision step99000 \
            --specific_source "${subset}" \
            --output_name $version \
            --skip_baselines true
    done
done
