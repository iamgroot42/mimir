#!/bin/bash
method_1=zlib #min_k #loss #ref-stablelm-base-alpha-3b-v2
method_2=zlib #min_k #ref-stablelm-base-alpha-3b-v2
# python cross_domain.py \
#"/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5_temporal_wiki/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-temporal_wiki_full/${method_1}_results.json" \
# "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-arxiv_ngram_13_<0.8_truncated/zlib_entropy_threshold_results.json"
python cross_domain.py \
"/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5_temporal_wiki/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-temporal_wiki_full/${method_1}_results.json" \
"/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5_sanitycheck_natural_wiki_arxiv/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-wikipedia_(en)_ngram_13_<0.8_truncated/${method_2}_results.json"

#"/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5_ref_tab1_v2/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-arxiv_ngram_13_<0.8_truncated/${method}_results.json"