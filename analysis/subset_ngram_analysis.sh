#!/bin/bash
# "/gscratch/h2lab/micdun/bff/deduped/pile_subsets/ngram_13/github" \
ngram=4
python analyze_ngram_overlap.py \
    /gscratch/h2lab/micdun/bff/deduped/pile_subsets/ngram_$ngram/pubmed_abstracts \
    /gscratch/h2lab/micdun/bff/deduped/pile_subsets/ngram_$ngram/freelaw \
    --subset_overlap_results_dir analysis_results/


    # /gscratch/h2lab/micdun/bff/deduped/pile_subsets/ngram_$ngram/arxiv \
    # /gscratch/h2lab/micdun/bff/deduped/pile_subsets/ngram_$ngram/github \
    # /gscratch/h2lab/micdun/bff/deduped/pile_subsets/ngram_$ngram/pubmed_central \
    # /gscratch/h2lab/micdun/bff/deduped/pile_subsets/ngram_$ngram/wikipedia_\(en\) \
    # /gscratch/h2lab/micdun/bff/deduped/pile_subsets/ngram_$ngram/pile_cc \