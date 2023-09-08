#!/bin/bash
for model in "gpt-neo-125m" "gpt-neo-1.3B" "gpt-neo-2.7B"
do
    for subset in "wikipedia_(en)" "pubmed_central" "arxiv"
    do
        knocky python run.py --config configs/mi.json --base_model "EleutherAI/${model}" --specific_source "${subset}" --output_name "${model}_${subset}" 
    done
done