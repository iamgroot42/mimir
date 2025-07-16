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
    args = parser.parse_args()
    print(args)

    results_to_set_threshold = args.results_one
    results_to_apply_threshold = args.results_two

    with open(results_to_set_threshold, 'r') as f:
        metadata = json.load(f)
        metadata_preds = metadata['predictions']
        member_scores = metadata_preds['member'] if 'member' in metadata_preds else metadata_preds['members']
        nonmember_scores = metadata_preds['nonmember'] if 'nonmember' in metadata_preds else metadata_preds['nonmembers']

    # Flip scores
    member_scores = np.array(member_scores) * -1
    nonmember_scores = np.array(nonmember_scores) * -1

    total_labels = [1] * len(member_scores) + [0] * len(nonmember_scores)
    fpr, tpr, thresholds = roc_curve(total_labels, member_scores.tolist() + nonmember_scores.tolist())
    roc_auc = auc(fpr, tpr)
    print("ROC AUC: ", roc_auc)

    set_fpr = [.001, .01, .05, .1]
    threshold_at_low_fpr = {upper_bound: thresholds[np.where(np.array(fpr) < upper_bound)[0][-1]] for upper_bound in set_fpr}
    tpr_at_low_fpr = {upper_bound: tpr[np.where(np.array(fpr) < upper_bound)[0][-1]] for upper_bound in set_fpr}
    print(threshold_at_low_fpr)
    # print(tpr_at_low_fpr)

    with open(results_to_apply_threshold, 'r') as f:
        metadata_apply = json.load(f)
        metadata_apply_preds = metadata_apply['predictions']
        apply_nonmember_scores = metadata_apply_preds['nonmember'] if 'nonmember' in metadata_apply_preds else metadata_apply_preds['nonmembers']
        apply_nonmember_scores = np.array(apply_nonmember_scores) * -1
        apply_member_scores = metadata_apply_preds['member'] if 'member' in metadata_apply_preds else metadata_apply_preds['members']
        apply_member_scores = np.array(apply_member_scores) * -1

    fprs = {}
    tprs = {}
    for upper_bound, threshold in threshold_at_low_fpr.items():
        fp = np.sum(apply_nonmember_scores >= threshold)
        tn = np.sum(apply_nonmember_scores < threshold)
        tp = np.sum(apply_member_scores >= threshold)
        fn = np.sum(apply_member_scores < threshold)

        fprs[upper_bound] = fp / (fp + tn)
        tprs[upper_bound] = tp / (tp + fn)

    print("Cross domain FPRs")
    print(fprs)
    # print("Cross domain TPRs")
    # print(tprs)