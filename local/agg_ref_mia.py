import argparse
import json
import numpy as np
from sklearn.metrics import roc_curve, auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ref_files', nargs="*")
    args = parser.parse_args()
    print(args)

    sum_member_scores = None
    sum_nonmember_scores = None
    print(len(args.ref_files))
    for ref in args.ref_files:
        f_ref_metrics = open(ref)
        ref_metrics_dict = json.load(f_ref_metrics)
        ref_member_scores = np.array(ref_metrics_dict['predictions']["member"])
        ref_nonmember_scores = np.array(ref_metrics_dict['predictions']["nonmember"])

        if sum_member_scores is None:
            sum_member_scores = ref_member_scores
            sum_nonmember_scores = ref_nonmember_scores
        else:
            sum_member_scores += ref_member_scores
            sum_nonmember_scores += ref_nonmember_scores

    avg_member_scores = sum_member_scores / len(args.ref_files)
    avg_nonmember_scores = sum_nonmember_scores / len(args.ref_files)

    total_labels = [0] * len(avg_member_scores) + [1] * len(avg_nonmember_scores)
    fpr, tpr, _ = roc_curve(total_labels, avg_member_scores.tolist() + avg_nonmember_scores.tolist())
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

    tpr_at_low_fpr = {upper_bound: tpr[np.where(np.array(fpr) < upper_bound)[0][-1]] for upper_bound in [0.001, 0.01]}
    print(tpr_at_low_fpr)

        