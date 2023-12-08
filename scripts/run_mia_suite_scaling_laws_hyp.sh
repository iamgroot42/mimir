#!/bin/bash
version=unified_mia_v5_scaling_law_hyp

for i_range in "13900 14000 1.4b" "26900 27000 2.8b" "65900 66000 6.9b" "114900 115000 12b"
do
    set -- $i_range
    echo ckpt $2, model $3
    python run.py \
        --config configs/mi.json \
        --base_model "EleutherAI/pythia-$3-deduped" \
        --revision step$2 \
        --presampled_dataset_member /gscratch/h2lab/micdun/pile-domains/pythia/utils/batch_viewing/token_indicies/$1-$2-indicies-n10000-samples.npy \
        --presampled_dataset_nonmember /gscratch/h2lab/micdun/mimir/data/tokenized_test/0.0-0.8/full_pile_10000/test_tk.npy \
        --specific_source "$1-$2-pile" \
        --output_name $version \
        --blackbox_attacks loss+ref+zlib+min_k \
        --baselines_only true \
        --pretokenized true
done