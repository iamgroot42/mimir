#!/bin/bash
python agg_ref_mia.py \
    "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5_ref_tab1_v2_remainder_llama/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-arxiv_ngram_13_<0.8_truncated/ref-llama-7b_results.json" \
    "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5_ref_tab1_v2/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-arxiv_ngram_13_<0.8_truncated/ref-stablelm-base-alpha-3b-v2_results.json" \
    "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5_ref_tab1_v2_remainder/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-arxiv_ngram_13_<0.8_truncated/ref-gpt2_results.json" \
    "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5_ref_tab1_v2_remainder/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-arxiv_ngram_13_<0.8_truncated/ref-silo-pdswby-1.3b_results.json" \
    "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5_ref_tab1_v2/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-arxiv_ngram_13_<0.8_truncated/ref-distilgpt2_results.json" \
    "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5_ref_tab1_v2/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-arxiv_ngram_13_<0.8_truncated/ref-opt-1.3B_results.json" \
    "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5_ref_tab1_v2/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-arxiv_ngram_13_<0.8_truncated/ref-gpt-neo-1.3B_results.json" \
    "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v5_ref_tab1_v2/EleutherAI_pythia-12b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-arxiv_ngram_13_<0.8_truncated/ref-pythia-1.4b-deduped_results.json" 
    
    
    
    # "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v4_other_ref/EleutherAI_pythia-6.9b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-wikipedia_(en)/ref_model__gscratch_h2lab_sewon_nplm-inference_ckpt_ours-v2_1.3B_400B_semibalanced_lira_ratio_threshold_results.json" \
    # "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v4_other_ref/EleutherAI_pythia-6.9b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-wikipedia_(en)/ref_model_distilgpt2_lira_ratio_threshold_results.json" \
    # "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v4/EleutherAI_pythia-6.9b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-wikipedia_(en)/ref_model_EleutherAI_pythia-70m_lira_ratio_threshold_results.json" \
    # "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v4/EleutherAI_pythia-6.9b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-wikipedia_(en)/ref_model_gpt2_lira_ratio_threshold_results.json"
