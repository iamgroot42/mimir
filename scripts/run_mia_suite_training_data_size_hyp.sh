#!/bin/bash
version=unified_mia_v5_linear_tds_hyp_ref
n_samples=1000



for model in "1.4b" "12b" "2.8b" "6.9b"
do
    for i_range in "900 1000 1000" "9900 10000 10000" "19900 20000 20000" "29900 30000 30000" "39900 40000 40000" "49900 50000 50000" "4900 5000 5000" "14900 15000 15000" "24900 25000 25000" "34900 35000 35000" "44900 45000 45000" "59900 60000 60000" "69900 70000 70000" "79900 80000 80000" "89900 90000 90000" "98900 99000 99000" "54900 55000 55000" "64900 65000 65000" "74900 75000 75000" "84900 85000 85000" "94900 95000 95000"
    do
        set -- $i_range
        # echo $1 and $2
        echo ckpt$3
        # python run.py \
        #     --config configs/mi.json \
        #     --base_model "EleutherAI/pythia-$model-deduped" \
        #     --revision step$3 \
        #     --presampled_dataset_member /gscratch/h2lab/micdun/pile-domains/pythia/utils/batch_viewing/token_indicies/$1-$2-indicies-n$n_samples-samples.npy \
        #     --presampled_dataset_nonmember /gscratch/h2lab/micdun/mimir/data/tokenized_test/0.0-0.8/full_pile_$n_samples/test_tk.npy \
        #     --n_samples $n_samples \
        #     --specific_source $1-$2-pile \
        #     --output_name $version \
        #     --blackbox_attacks loss+ref+zlib+min_k \
        #     --baselines_only true \
        #     --pretokenized true
        python run.py \
            --config configs/mi.json \
            --base_model "EleutherAI/pythia-$model-deduped" \
            --revision step$3 \
            --presampled_dataset_member /gscratch/h2lab/micdun/pile-domains/pythia/utils/batch_viewing/token_indicies/$1-$2-indicies-n$n_samples-samples.npy \
            --presampled_dataset_nonmember /gscratch/h2lab/micdun/mimir/data/tokenized_test/0.0-0.8/full_pile_$n_samples/test_tk.npy \
            --n_samples $n_samples \
            --specific_source "$1-$2-pile" \
            --output_name $version \
            --baselines_only true \
            --pretokenized true
    done
done