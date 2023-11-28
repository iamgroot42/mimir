#!/bin/bash
version=unified_mia_v5_training_data_size_hyp

for i_range in "900 1000 1000" "2750 2850 3000" "5610 5710 6000" "11810 11910 12000" "23730 23830 24000" "47560 47660 48000" "98900 99000 99000"
do
    set -- $i_range
    echo $1 and $2
    echo ckpt $3
    python run.py \
        --config configs/mi.json \
        --base_model "EleutherAI/pythia-12b-deduped" \
        --revision step$3 \
        --presampled_dataset_member /gscratch/h2lab/micdun/pile-domains/pythia/utils/batch_viewing/token_indicies/$1-$2-indicies-n1000-samples.npy \
        --presampled_dataset_nonmember /gscratch/h2lab/micdun/mimir/data/tokenized_test/0.0-0.8/full_pile/test_tk.npy \
        --specific_source $1-$2-pile \
        --output_name $version \
        --blackbox_attacks loss+ref+zlib+min_k \
        --baselines_only true \
        --pretokenized true
done