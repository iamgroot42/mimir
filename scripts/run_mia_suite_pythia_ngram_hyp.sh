#!/bin/bash
version=unified_mia_v5_hyp_ngram_overlap_ne_redo
ngram=13

# deduped models
for model in  "pythia-12b-deduped" #"pythia-160m-deduped" "pythia-1.4b-deduped" "pythia-2.8b-deduped" "pythia-6.9b-deduped"
do
    # python run.py \
    #         --config configs/single_gpu_mi.json \
    #         --base_model "EleutherAI/${model}" \
    #         --revision step99000 \
    #         --specific_source "dm_mathematics_ngram_13_<0.2_truncated" \
    #         --output_name unified_mia_v5_hyp_ngram_overlap \
    #         --baselines_only true
    
    # python run.py \
    #         --config configs/single_gpu_mi.json \
    #         --base_model "EleutherAI/${model}" \
    #         --revision step99000 \
    #         --specific_source "dm_mathematics_ngram_7_<0.2_truncated" \
    #         --output_name unified_mia_v5_hyp_ngram_overlap \
    #         --baselines_only true

    for subset in "wikipedia_(en)" "arxiv" #"github" "pile_cc" "pubmed_central"  "arxiv" "hackernews" "dm_mathematics"
    do
        python run.py \
            --config configs/mi.json \
            --base_model "EleutherAI/${model}" \
            --revision step99000 \
            --specific_source "${subset}_ngram_13_<0.2_truncated" \
            --output_name $version \
            --baselines_only true

        python run.py \
            --config configs/mi.json \
            --base_model "EleutherAI/${model}" \
            --revision step99000 \
            --specific_source "${subset}_ngram_7_<0.2_truncated" \
            --output_name $version \
            --baselines_only true
    done
done
