#!/bin/bash
version=unified_mia_v5_linear_tds_hyp
n_samples=1000

for model in "2.8b" "6.9b" "12b" "1.4b"
do
    for i_range in "4900 5000 5000" "14900 15000 15000" "24900 25000 25000" "34900 35000 35000" "44900 45000 45000" "54900 55000 55000" "64900 65000 65000" "74900 75000 75000" "84900 85000 85000" "94900 95000 95000"
    do
        set -- $i_range
        echo $1 and $2
        echo ckpt$3
        python run.py \
            --config configs/mi.json \
            --base_model "EleutherAI/pythia-$model-deduped" \
            --revision step$3 \
            --presampled_dataset_member /gscratch/h2lab/micdun/pile-domains/pythia/utils/batch_viewing/token_indicies/$1-$2-indicies-n$n_samples-samples.npy \
            --presampled_dataset_nonmember /gscratch/h2lab/micdun/mimir/data/tokenized_test/0.0-0.8/full_pile_$n_samples/test_tk.npy \
            --n_samples $n_samples \
            --specific_source $1-$2-pile \
            --output_name $version \
            --blackbox_attacks loss+ref+zlib+min_k \
            --baselines_only true \
            --pretokenized true
    done
done