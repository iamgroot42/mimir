#!/bin/bash
version=unified_mia_v5_hyp_num_epochs_ne_redo

for model in "2b855b55bc4" "2b855b28bc4" "2b855b14bc4" "2b855b9bc4" "2b855b4bc4"
do
    for subset in "c4"
    do
         python run.py \
            --config configs/mi.json \
            --base_model "/gscratch/h2lab/micdun/datablations/lm1-2b8-55b-c4-repetitions/$model/transformers" \
            --specific_source $subset \
            --output_name $version \
            --baselines_only true
    done
done

# #!/bin/bash
# version=unified_mia_v5_hyp_train_data_size

# for model in "lm1-8b7-178b-c4-repetitions/8b7178b178b" "lm1-4b2-84b-c4-repetitions/4b284b84bc4" "lm1-2b8-55b-c4-repetitions/2b855b55bc4" "lm1-1b1-21b-c4seeds/1b121b21bc4seed1"
# do
#     for subset in "c4"
#     do
#         echo running mia for model $model
#         python run.py \
#             --config configs/mi.json \
#             --base_model "/gscratch/h2lab/micdun/datablations/$model/transformers" \
#             --specific_source $subset \
#             --output_name $version \
#             --blackbox_attacks loss+ref+zlib+min_k \
#             --baselines_only true
#     done
# done