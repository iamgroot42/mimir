#!/bin/bash
for model in "pythia-160m-deduped" "pythia-1.4b-deduped" "pythia-2.8b-deduped" "pythia-6.9b-deduped" "pythia-12b-deduped"
do
    for subset in "freelaw"
    do
        python mia_scores_visualization.py \
            $RESULTS_PATH/mia_unified_mia_v4_testing/EleutherAI_$model--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-${subset}_ngram_13_\<0.2_truncated \
            $RESULTS_PATH/mia_unified_mia_v4/EleutherAI_$model--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-$subset \
            --output_dir score_analysis_v2/$model \
            --subset $subset
    done
done