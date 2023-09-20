import argparse
import json 
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from collections import defaultdict
from tqdm import tqdm

NGRAM_METADATA = "ngram_metadata.json"
LL = "likelihood_threshold_results.json"
LIRAS = [
    "ref_model_EleutherAI_pythia-70m_lira_ratio_threshold_results.json",
    "ref_model_gpt2_lira_ratio_threshold_results.json"
]

N = ""
OUTPUT = "scatterplots"

# Set the figure size
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('dirs', nargs="*")
    parser.add_argument('--scores_dirs', nargs="*")  # TODO: maybe make it support scores_dirs
    parser.add_argument('--subset_overlap_results_dir', type=str)
    parser.add_argument('--ngram', type=int)
    parser.add_argument('--subset', type=str, default=None)
    args = parser.parse_args()
    print(args)
    scores_dirs = args.scores_dirs
    overlap_dir = args.subset_overlap_results_dir
    ngram = args.ngram
    for scores_dir in tqdm(scores_dirs):
        # Use subset from arguments, otherwise get from suffix of scores_dir
        subset = args.subset if args.subset else scores_dir.split('-')[-1].replace('/', '')
        model_result_dir = os.path.split(os.path.split(scores_dir)[0])[-1]
        output_dir = os.path.join(overlap_dir, OUTPUT, model_result_dir, subset, str(ngram))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load overlap metadata
        f_ngram_metadata = open(os.path.join(overlap_dir, subset, f"ngram_{ngram}", NGRAM_METADATA))
        ngram_metadata = json.load(f_ngram_metadata)
        member_overlap = ngram_metadata["train"]["individual_ngram_overlap"]
        nonmember_overlap = ngram_metadata["test"]["individual_ngram_overlap"]
        
        # Load results dictionaries
        f_lls = open(f'{scores_dir}/{LL}')
        lls_dict = json.load(f_lls)
        member_metadata = [(res["member"], pred) for res, pred in zip(lls_dict["raw_results"], lls_dict["predictions"]["members"]) if not math.isnan(pred)]
        members = [metadata[0] for metadata in member_metadata]
        member_lls = [metadata[1] for metadata in member_metadata]

        nonmember_metadata = [(res["nonmember"], pred) for res, pred in zip(lls_dict["raw_results"], lls_dict["predictions"]["nonmembers"]) if not math.isnan(pred)]
        nonmembers = [metadata[0] for metadata in nonmember_metadata]
        nonmember_lls = [metadata[1] for metadata in nonmember_metadata]
        total_lls = member_lls + nonmember_lls

        # overlap to list
        member_overlap_list = [member_overlap[member] for member in members]
        nonmember_overlap_list = [nonmember_overlap[nonmember] for nonmember in nonmembers]
        total_overlap_list = member_overlap_list + nonmember_overlap_list

        # pearson r
        r, p = stats.pearsonr(total_overlap_list, total_lls)

        # Fit a 1st degree polynomial (a line) to the data
        m, c = np.polyfit(total_overlap_list, total_lls, 1)

        # Generate y-values for the line of best fit
        y_fit = m * np.array(total_overlap_list) + c

        # Scatter plot
        plt.figure()
        plt.scatter(member_overlap_list, member_lls, alpha=0.25, c='blue', cmap='hot',label = 'member')
        plt.scatter(nonmember_overlap_list, nonmember_lls, alpha=0.25, c='orange', cmap='hot',label = 'nonmember')

        # Plot the line of best fit
        plt.plot(total_overlap_list, y_fit, 'r')
        plt.xlabel("ngram overlap")
        plt.ylabel("ll score")
        plt.annotate('r = {:.3f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')
        plt.savefig(os.path.join(output_dir, "ll.png"))
        plt.close()

        for ref in LIRAS:
            f_liras = open(f'{scores_dir}/{ref}')
            liras_dict = json.load(f_liras)
            member_metadata = [(res["member"], pred) for res, pred in zip(liras_dict["raw_results"], liras_dict["predictions"]["members"]) if not math.isnan(pred)]
            members = [metadata[0] for metadata in member_metadata]
            member_liras = [metadata[1] for metadata in member_metadata]

            nonmember_metadata = [(res["nonmember"], pred) for res, pred in zip(liras_dict["raw_results"], liras_dict["predictions"]["nonmembers"]) if not math.isnan(pred)]
            nonmembers = [metadata[0] for metadata in nonmember_metadata]
            nonmember_liras = [metadata[1] for metadata in nonmember_metadata]
            total_liras = member_liras + nonmember_liras

            member_overlap_list = [member_overlap[member] for member in members]
            nonmember_overlap_list = [nonmember_overlap[nonmember] for nonmember in nonmembers]
            total_overlap_list = member_overlap_list + nonmember_overlap_list

            # pearson r
            r, p = stats.pearsonr(total_overlap_list, total_liras)

            # Fit a 1st degree polynomial (a line) to the data
            m, c = np.polyfit(total_overlap_list, total_liras, 1)

            # Generate y-values for the line of best fit
            y_fit_lira = m * np.array(total_overlap_list) + c

            plt.figure()
            plt.scatter(member_overlap_list, member_liras, alpha=0.25, c='blue', cmap='hot',label = 'member')
            plt.scatter(nonmember_overlap_list, nonmember_liras, alpha=0.25, c='orange', cmap='hot',label = 'nonmember')
            
            # Plot the line of best fit
            plt.plot(total_overlap_list, y_fit_lira, 'r')
            plt.xlabel("ngram overlap")
            plt.ylabel("lira score")
            plt.annotate('r = {:.3f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')
            plt.savefig(os.path.join(output_dir, f"{ref}_lira.png"))
            plt.close()

    
