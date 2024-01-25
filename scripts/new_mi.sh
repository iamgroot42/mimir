#!/bin/bash
version=new_mi_experiments

for subset in "wikipedia_(en)" "arxiv"
do
    knocky python new_mi_experiment.py \
        --config configs/mi.json \
        --revision step99000 \
        --base_model "EleutherAI/pythia-12b-deduped" \
        --specific_source "${subset}"
done
