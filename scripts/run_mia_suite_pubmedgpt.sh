#!/bin/bash
version=unified_mia_v5_pubmedgpt_exp
ngram=7

# for subset in "pubmed_central" "pubmed_abstracts"
# do
#     for model in "/gscratch/h2lab/micdun/mosaicml-benchmarks/pubmedgpt/converted" "stanford-crfm/BioMedLM"
#     do
#         python run.py \
#             --config configs/mi.json \
#             --base_model $model \
#             --specific_source ${subset}_ngram_${ngram}_\<0.2_truncated \
#             --output_name $version \
#             --baselines_only true 
#     done
# done

for subset in "pubmed_central" "pubmed_abstracts"
do
    for model in "stanford-crfm/BioMedLM" #"/gscratch/h2lab/micdun/mosaicml-benchmarks/pubmedgpt/converted"
    do
        python run.py \
            --config configs/mi.json \
            --base_model $model \
            --specific_source ${subset}_ngram_13_\<0.8_truncated \
            --output_name $version \
            --baselines_only true 
    done
done
