#!/bin/bash
# for model in "gpt-neo-125m" "gpt-neo-1.3B" "gpt-neo-2.7B"
# do
#     for subset in "wikipedia_(en)" "pubmed_central" "arxiv"
#     do
#         knocky python run.py --config configs/mi.json --base_model "EleutherAI/${model}" --specific_source "${subset}" --output_name "${model}_${subset}" 
#     done
# done

version=unified_mia_v5_gpt_neo
baselines_only=$1
skip_baselines=$2
ngram=13
# CUDA_LAUNCH_BLOCKING=1

# deduped models
# TODO: refactor this into one loop
for model in  "gpt-neo-2.7B" "gpt-neo-1.3B" "gpt-neo-125m"
do
    for subset in "github" "wikipedia_(en)" "dm_mathematics" "hackernews" "pile_cc" "pubmed_central" "arxiv" # "full_pile" "pubmed_central" 
    do
        # CUDA_LAUNCH_BLOCKING=1 python run.py \
        #     --config configs/single_gpu_mi.json \
        #     --base_model "EleutherAI/${model}-deduped" \
        #     --revision step99000 \
        #     --specific_source ${subset}_ngram_${ngram}_\<0.8_truncated \
        #     --output_name $version \
        #     --baselines_only true \
        #     --n_samples 1000
        CUDA_LAUNCH_BLOCKING=1 python run.py \
            --config configs/mi.json \
            --base_model "EleutherAI/${model}" \
            --specific_source ${subset}_ngram_${ngram}_\<0.8_truncated \
            --output_name $version \
            --baselines_only true \
            --n_samples 1000
    done

    for subset in "full_pile" #"wikipedia_(en)" "dm_mathematics" "hackernews" "pile_cc" "pubmed_central" "arxiv" "github" # "full_pile" "pubmed_central" 
    do
        # CUDA_LAUNCH_BLOCKING=1 python run.py \
        #     --config configs/single_gpu_mi.json \
        #     --base_model "EleutherAI/${model}" \
        #     --revision step99000 \
        #     --specific_source ${subset} \
        #     --output_name $version \
        #     --baselines_only true \
        #     --n_samples 10000
        CUDA_LAUNCH_BLOCKING=1 python run.py \
            --config configs/mi.json \
            --base_model "EleutherAI/${model}" \
            --specific_source ${subset} \
            --output_name $version \
            --baselines_only true \
            --n_samples 10000
    done
done