#!/bin/bash
version=unified_mia_v5_test
baselines_only=$1
skip_baselines=$2
ngram=13

# deduped models
# TODO: refactor this into one loop
for model in "pythia-160m-deduped" "pythia-1.4b-deduped" "pythia-2.8b-deduped" "pythia-6.9b-deduped" "pythia-12b-deduped"
do
    for subset in "full_pile" #"wikipedia_(en)" "dm_mathematics" "hackernews" "pile_cc" "github" "pubmed_central" "arxiv"
    do
        # python run.py \
        #     --config configs/single_gpu_mi.json \
        #     --base_model "EleutherAI/${model}" \
        #     --revision step99000 \
        #     --specific_source ${subset}_ngram_${ngram}_\<0.8_truncated \
        #     --output_name $version \
        #     --blackbox_attacks loss+ref+zlib+min_k \
        #     --baselines_only true
        python run.py \
            --config configs/mi.json \
            --base_model "EleutherAI/${model}" \
            --revision step99000 \
            --specific_source $subset\
            --output_name $version \
            --blackbox_attacks loss+ref+zlib+min_k \
            --baselines_only true \
            --n_samples 10000
    done
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
