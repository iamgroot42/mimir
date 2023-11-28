#!/bin/bash
version=unified_mia_v5_doc_special_mia_all_substrs_max_80_random
baselines_only=$1
skip_baselines=$2

# deduped models
for model in "pythia-12b-deduped" # "pythia-1.4b-deduped" "pythia-160m-deduped" "pythia-2.8b-deduped" "pythia-6.9b-deduped"
do
    for subset in "books3" # "wikipedia_(en)" "books3" "hackernews" "pile_cc" "pubmed_central" "arxiv" "github" "dm_mathematics"
    do
        python run.py \
            --config configs/single_gpu_mi.json \
            --base_model "EleutherAI/${model}" \
            --revision step99000 \
            --specific_source "${subset}_sampled_substr" \
            --token_frequency_map /gscratch/h2lab/micdun/mimir/data/util/pile_tk_freq.pkl \
            --output_name $version \
            --skip_baselines true \
            --baselines_only true \
            --special_mia true \
            --full_doc true \
            --n_samples 100 \
            --max_substrs 80
    done
done

# # ngram filtered
# ngram=13
# for model in "pythia-1.4b-deduped" "pythia-160m-deduped" "pythia-2.8b-deduped" "pythia-6.9b-deduped" # "pythia-12b-deduped"
# do
#     for subset in "wikipedia_(en)" "pile_cc" "github" "pubmed_central"
#     do
#         python run.py \
#             --config configs/mi.json \
#             --base_model "EleutherAI/${model}" \
#             --revision step99000 \
#             --specific_source "${subset}_ngram_${ngram}_<0.2_truncated" \
#             --output_name $version \
#             --skip_baselines true \
#             --baselines_only true \
#             --special_mia true
#     done
# done
