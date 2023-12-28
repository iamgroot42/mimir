#!/bin/bash
python agg_ref_mia.py \
    "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v4_llama_ref/EleutherAI_pythia-2.8b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-wikipedia_(en)/ref_model_decapoda-research_llama-7b-hf_lira_ratio_threshold_results.json" \
    "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v4_pile_refs/EleutherAI_pythia-2.8b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-wikipedia_(en)/ref_model_facebook_opt-125m_lira_ratio_threshold_results.json" \
    
    
    
    # "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v4_other_ref/EleutherAI_pythia-6.9b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-wikipedia_(en)/ref_model__gscratch_h2lab_sewon_nplm-inference_ckpt_ours-v2_1.3B_400B_semibalanced_lira_ratio_threshold_results.json" \
    # "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v4_other_ref/EleutherAI_pythia-6.9b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-wikipedia_(en)/ref_model_distilgpt2_lira_ratio_threshold_results.json" \
    # "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v4/EleutherAI_pythia-6.9b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-wikipedia_(en)/ref_model_EleutherAI_pythia-70m_lira_ratio_threshold_results.json" \
    # "/mmfs1/gscratch/h2lab/micdun/mimir/results_new/mia_unified_mia_v4/EleutherAI_pythia-6.9b-deduped--bert-temp/fp32-0.3-1-the_pile-the_pile-1000200100_plen30_--tok_false-wikipedia_(en)/ref_model_gpt2_lira_ratio_threshold_results.json"
