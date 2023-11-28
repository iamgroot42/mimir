#!/bin/bash
version=unified_mia_v5_hyp_ngram_overlap_mink
ngram=13

# deduped models
for model in  "pythia-12b-deduped" "pythia-160m-deduped" "pythia-1.4b-deduped" "pythia-2.8b-deduped" "pythia-6.9b-deduped"
do
    python run.py \
            --config configs/single_gpu_mi.json \
            --base_model "EleutherAI/${model}" \
            --revision step99000 \
            --specific_source "dm_mathematics_ngram_13_<0.2_truncated" \
            --output_name unified_mia_v5_hyp_ngram_overlap \
            --baselines_only true
    
    python run.py \
            --config configs/single_gpu_mi.json \
            --base_model "EleutherAI/${model}" \
            --revision step99000 \
            --specific_source "dm_mathematics_ngram_7_<0.2_truncated" \
            --output_name unified_mia_v5_hyp_ngram_overlap \
            --baselines_only true

    for subset in "github" "wikipedia_(en)" "pile_cc" "pubmed_central" "hackernews" "arxiv" "dm_mathematics"
    do
        python run.py \
            --config configs/single_gpu_mi.json \
            --base_model "EleutherAI/${model}" \
            --revision step99000 \
            --specific_source "${subset}_ngram_13_<0.2_truncated" \
            --output_name $version \
            --skip_baselines true \
            --baselines_only true \
            --special_mia true

        python run.py \
            --config configs/single_gpu_mi.json \
            --base_model "EleutherAI/${model}" \
            --revision step99000 \
            --specific_source "${subset}_ngram_7_<0.2_truncated" \
            --output_name $version \
            --skip_baselines true \
            --baselines_only true \
            --special_mia true
    done
done
