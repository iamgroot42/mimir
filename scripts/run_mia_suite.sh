#!/bin/bash
for model in "pythia-2.8b" "pythia-2.8b-deduped" "pythia-1.4b" "pythia-1.4b-deduped" "pythia-160m" "pythia-160m-deduped" 
do
    for subset in "wikipedia_(en)" "pubmed_central" "arxiv"
    do
        python run.py \
            --config configs/pythia/${model}/${subset}/mia.json
    done
done
