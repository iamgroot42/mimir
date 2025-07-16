import argparse
import json 
import os
import datasets
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve, auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_one', type=str)
    parser.add_argument('results_two', type=str)
    parser.add_argument('--attack', type=str)
    args = parser.parse_args()
    print(args)

    results_to_set_threshold = args.results_one
    results_to_apply_threshold = args.results_two
    attack = args.attack

    with open(results_to_set_threshold, 'r') as f:
        metadata = json.load(f)
        member_scores = metadata['predictions']['member']
        nonmember_scores = metadata['predictions']['nonmember']

    # Flip scores
    member_scores = np.array(member_scores) * -1
    nonmember_scores = np.array(nonmember_scores) * -1

    total_labels = [1] * len(member_scores) + [0] * len(nonmember_scores)
    fpr, tpr, thresholds = roc_curve(total_labels, member_scores.tolist() + nonmember_scores.tolist())
    roc_auc = auc(fpr, tpr)
    print("ROC AUC: ", roc_auc)

    threshold_at_low_fpr = {upper_bound: thresholds[np.where(np.array(fpr) < upper_bound)[0][-1]] for upper_bound in [0.001, 0.01, .1, .25, .5]}
    # print(threshold_at_low_fpr)

    with open(results_to_apply_threshold, 'r') as f:
        metadata_apply = json.load(f)
        # structure: attack -> num_edits -> trial
        scores_for_attack = metadata_apply[attack]

    
    for edit, trials in scores_for_attack.items():
        for trial, scores in trials.items():
            edm_scores = np.array(scores) * -1

            fprs = {}
            tnrs = {}
            for upper_bound, threshold in threshold_at_low_fpr.items():
                fp = np.sum(edm_scores >= threshold)
                tn = np.sum(edm_scores < threshold)

                fprs[upper_bound] = fp / (fp + tn)
                tnrs[upper_bound] = tn / (fp + tn)

            

    print("Cross domain FPRs")
    print(fprs)
    print("Cross domain TPRs")
    print(tprs)