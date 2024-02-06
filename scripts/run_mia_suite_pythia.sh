#!/bin/bash
version=unified_mia_v5_fixed_ne
baselines_only=$1
skip_baselines=$2
ngram=13
# CUDA_LAUNCH_BLOCKING=1

# deduped models
# TODO: refactor this into one loop
for model in "pythia-12b" # "pythia-160m" "pythia-1.4b"  "pythia-6.9b" "pythia-2.8b" 
do
    for subset in "pile_cc" "pubmed_central" "dm_mathematics" "hackernews" # "wikipedia_(en)" "arxiv" "github" # "full_pile" "pubmed_central" 
    do
        # CUDA_LAUNCH_BLOCKING=1 python run.py \
        #     --config configs/single_gpu_mi.json \
        #     --base_model "EleutherAI/${model}-deduped" \
        #     --revision step99000 \
        #     --specific_source ${subset}_ngram_${ngram}_\<0.8_truncated \
        #     --output_name $version \
        #     --baselines_only true \
        #     --n_samples 1000
        python run.py \
            --config configs/single_gpu_mi.json \
            --base_model "EleutherAI/${model}-deduped" \
            --revision step99000 \
            --specific_source ${subset}_ngram_${ngram}_\<0.8_truncated \
            --output_name $version \
            --baselines_only true \
            --n_samples 1000
    done

    # for subset in "full_pile" #"wikipedia_(en)" "dm_mathematics" "hackernews" "pile_cc" "pubmed_central" "arxiv" "github" # "full_pile" "pubmed_central" 
    # do
    #     # CUDA_LAUNCH_BLOCKING=1 python run.py \
    #     #     --config configs/single_gpu_mi.json \
    #     #     --base_model "EleutherAI/${model}" \
    #     #     --revision step99000 \
    #     #     --specific_source ${subset} \
    #     #     --output_name $version \
    #     #     --baselines_only true \
    #     #     --n_samples 10000
    #     CUDA_LAUNCH_BLOCKING=1 python run.py \
    #         --config configs/single_gpu_mi.json \
    #         --base_model "EleutherAI/${model}" \
    #         --specific_source ${subset} \
    #         --output_name $version \
    #         --baselines_only true \
    #         --n_samples 10000
    # done
done

# # non-deduped models
# for model in "pythia-160m" "pythia-1.4b" "pythia-2.8b" "pythia-6.9b" "pythia-12b"
# do
#     for subset in "wikipedia_(en)" "dm_mathematics" "hackernews" "pile_cc" "github" "pubmed_central" "arxiv"
#     do
#         python run.py \
#             --config configs/mi.json \
#             --base_model "EleutherAI/${model}" \
#             --specific_source ${subset}_ngram_${ngram}_\<0.8_truncated \
#             --output_name $version \
#             --baselines_only true \
#             --special_mia true
#     done
# done
