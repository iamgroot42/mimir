import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

GPT2_REF_FILE = "ref_model_gpt2_lira_ratio_threshold_results.json"
PYTHIA_REF_FILE = "ref_model_EleutherAI_pythia-70m_lira_ratio_threshold_results.json"
LL_FILE = "likelihood_threshold_results.json"

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs="*")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--subset', type=str)
    args = parser.parse_args()
    print(args)

    scatter_results_dir = os.path.join(args.output_dir, "scatter", args.subset)
    if not os.path.exists(scatter_results_dir):
        os.makedirs(scatter_results_dir)

    hist_results_dir = os.path.join(args.output_dir, "hist", args.subset)
    if not os.path.exists(hist_results_dir):
        os.makedirs(hist_results_dir)

#     fig, ax = plt.subplots(1) # Creates figure fig and add an axes, ax.
# fig2, ax2 = plt.subplots(1) # Another figure

    for ref in [GPT2_REF_FILE, PYTHIA_REF_FILE]:
        scatter, scatter_ax = plt.subplots(1)
        scatter_ax.set_xlabel("target score")
        scatter_ax.set_ylabel("ref score")

        # plot target vs ref model scores for each provided setting
        for d in args.dirs:
            f_ll_metrics = open(os.path.join(d, LL_FILE))
            metrics_dict = json.load(f_ll_metrics)
            member_scores = metrics_dict['predictions']["members"]
            nonmember_scores = metrics_dict['predictions']["nonmembers"]
            f_ref_metrics = open(os.path.join(d, ref))
            ref_metrics_dict = json.load(f_ref_metrics)
            ref_member_scores = ref_metrics_dict['predictions']["members"]
            ref_nonmember_scores = ref_metrics_dict['predictions']["nonmembers"]

            ngram_filter_status = "_".join(d.split('-')[-1].split('_')[1:]) if 'ngram' in d else 'original'
            ngram_filter_status = ngram_filter_status.replace('<', '')
            if not os.path.exists(os.path.join(hist_results_dir, ngram_filter_status)):
                os.makedirs(os.path.join(hist_results_dir, ngram_filter_status))

            # Get ref model scores, loss targeet - loss ref = score - > target loss - ref score -> loss ref
            ref_ll_member = np.array(member_scores) - np.array(ref_member_scores)
            ref_ll_nonmember = np.array(nonmember_scores) - np.array(ref_nonmember_scores)

            # Plot the scatter
            scatter_ax.scatter(member_scores, ref_ll_member, label=f"member - {'filtered' if 'ngram' in d else 'original'}", s=5)
            scatter_ax.scatter(nonmember_scores, ref_ll_nonmember, label=f"nonmember - {'filtered' if 'ngram' in d else 'original'}", s=5)
            scatter.legend()

            hist, hist_ax = plt.subplots(1)
            hist_ax.set_xlabel("target score")
            hist_ax.hist(member_scores, bins=100, label=f"member - {'filtered' if 'ngram' in d else 'original'}")
            hist_ax.hist(nonmember_scores, bins=100, label=f"nonmember - {'filtered' if 'ngram' in d else 'original'}")
            hist.legend()
            hist.savefig(os.path.join(hist_results_dir, ngram_filter_status, f"target_ll_hist.png"))

            hist_ref, hist_ref_ax = plt.subplots(1)
            hist_ref_ax.set_xlabel("ref score")
            hist_ref_ax.hist(ref_ll_member, bins=100, label=f"member - {'filtered' if 'ngram' in d else 'original'}")
            hist_ref_ax.hist(ref_ll_nonmember, bins=100, label=f"nonmember - {'filtered' if 'ngram' in d else 'original'}")
            hist_ref.legend()
            hist_ref.savefig(os.path.join(hist_results_dir, ngram_filter_status, f"{'gpt2' if ref == GPT2_REF_FILE else 'pythia'}_ref_hist.png"))

            r, r_ax = plt.subplots(1)
            r_ax.set_xlabel("ref score")
            r_ax.hist(ref_member_scores, bins=100, label=f"member - {'filtered' if 'ngram' in d else 'original'}")
            r_ax.hist(ref_nonmember_scores, bins=100, label=f"nonmember - {'filtered' if 'ngram' in d else 'original'}")
            r.legend()
            r.savefig(os.path.join(hist_results_dir, ngram_filter_status, f"{'gpt2' if ref == GPT2_REF_FILE else 'pythia'}_ref_mia_hist.png"))

            if 'ngram' in d:
                ngram_filter = f"original_vs_{d.split('-')[-1].split('_')[1:]}"
                ngram_filter = ngram_filter.replace('<', '')



        # Save the scatter
        if not os.path.exists(os.path.join(scatter_results_dir, ngram_filter)):
            os.makedirs(os.path.join(scatter_results_dir, ngram_filter))
        scatter.savefig(os.path.join(scatter_results_dir, ngram_filter, f"original_vs_{'gpt2' if ref == GPT2_REF_FILE else 'pythia'}_ref_scatter.png"))
