#!/bin/bash
version=unified_mia_v5_silo_exps_ref

ckpt_dir="/gscratch/h2lab/sewon/nplm-inference/ckpt"
# PDSW semi-balanced
for model in "ours-v1_1.3B_200B_semibalanced" "ours-v1_1.3B_90B_semibalanced" "ours-v1_1.3B_250B_semibalanced" "ours-v1_1.3B_400B_semibalanced"
do
    for subset in "dm_mathematics_ngram_13_<0.8_truncated" "hackernews_ngram_13_<0.8_truncated" "sw_github"
    do
         python run.py \
            --config configs/mi.json \
            --base_model $ckpt_dir/$model \
            --specific_source $subset \
            --output_name $version \
            --baselines_only true
    done
done

# # PDSWBY semibalanced
# for model in "ours-v2_1.3B_200B_semibalanced" "ours-v2_1.3B_250B_semibalanced" "ours-v2_1.3B_300B_semibalanced" "ours-v2_1.3B_350B_semibalanced" "ours-v2_1.3B_400B_semibalanced"
# do
#     for subset in "dm_mathematics_ngram_13_<0.8_truncated" "hackernews_ngram_13_<0.8_truncated" "sw_github"
#     do
#          python run.py \
#             --config configs/mi.json \
#             --base_model $ckpt_dir/$model \
#             --specific_source $subset \
#             --output_name $version \
#             --blackbox_attacks loss+ref+zlib+min_k \
#             --baselines_only true
#     done
# done

# # PDSW unbalanced
# for model in "ours-v1_1.3B_100B_unbalanced" "ours-v1_1.3B_200B_unbalanced" "ours-v1_1.3B_400B_unbalanced"
# do
#     for subset in "dm_mathematics_ngram_13_<0.8_truncated" "hackernews_ngram_13_<0.8_truncated" "sw_github"
#     do
#          python run.py \
#             --config configs/mi.json \
#             --base_model $ckpt_dir/$model \
#             --specific_source $subset \
#             --output_name $version \
#             --blackbox_attacks loss+ref+zlib+min_k \
#             --baselines_only true
#     done
# done

# # PDSWBY unbalanced
# for model in "ours-v2_1.3B_200B_unbalanced" "ours-v2_1.3B_400B_unbalanced"
# do
#     for subset in "dm_mathematics_ngram_13_<0.8_truncated" "hackernews_ngram_13_<0.8_truncated" "sw_github"
#     do
#          python run.py \
#             --config configs/mi.json \
#             --base_model $ckpt_dir/$model \
#             --specific_source $subset \
#             --output_name $version \
#             --blackbox_attacks loss+ref+zlib+min_k \
#             --baselines_only true
#     done
# done

