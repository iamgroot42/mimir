#!/bin/bash
version=unified_mia_v5_ref_exp_app_table_redo
baselines_only=$1
skip_baselines=$2
ngram=13

# deduped models
# TODO: refactor this into one loop
for model in "pythia-2.8b-deduped" #"pythia-12b-deduped" "pythia-160m-deduped" "pythia-1.4b-deduped" "pythia-6.9b-deduped"
do
    for subset in "arxiv" "hackernews" "pile_cc" "pubmed_central" "dm_mathematics"
    do
        python run.py \
            --config configs/ref_exp_mi.json \
            --base_model "EleutherAI/${model}" \
            --revision step99000 \
            --specific_source ${subset}_ngram_${ngram}_\<0.8_truncated \
            --output_name $version \
            --baselines_only true
    done
done

# for model in "pythia-12b-deduped" "pythia-2.8b-deduped" "pythia-6.9b-deduped"
# do
#     for subset in "wikipedia_(en)"
#     do
#         python run.py \
#             --config configs/ref_exp_mi.json \
#             --base_model "EleutherAI/${model}" \
#             --revision step99000 \
#             --specific_source ${subset}_ngram_7_\<0.2_truncated \
#             --output_name $version \
#             --baselines_only true
#     done
# done
