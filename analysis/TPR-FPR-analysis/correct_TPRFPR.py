import argparse
import json 
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', default=[], nargs="*")
    parser.add_argument('--output_dir', default="./")
    args = parser.parse_args()
    print(args)

    results = args.results
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    for results_file in tqdm(results):
        with open(results_file, 'r') as f:
            metadata = json.load(f)
            pred_dict = metadata['predictions']
            member_scores = pred_dict['member'] if 'member' in pred_dict else pred_dict['members']
            nonmember_scores = pred_dict['nonmember'] if 'nonmember' in pred_dict else pred_dict['nonmembers']

        # Flip scores
        member_scores = np.array(member_scores) * -1
        nonmember_scores = np.array(nonmember_scores) * -1

        total_labels = [1] * len(member_scores) + [0] * len(nonmember_scores)
        fpr, tpr, thresholds = roc_curve(total_labels, member_scores.tolist() + nonmember_scores.tolist())
        roc_auc = auc(fpr, tpr)
        print("ROC AUC: ", roc_auc)

        threshold_at_low_fpr = {upper_bound: thresholds[np.where(np.array(fpr) < upper_bound)[0][-1]] for upper_bound in [0.001, 0.01, .1]}
        tpr_at_low_fpr = {upper_bound: tpr[np.where(np.array(fpr) < upper_bound)[0][-1]] for upper_bound in [0.001, 0.01, .1]}
        print(threshold_at_low_fpr)
        print(tpr_at_low_fpr)

        # dump tpr@lowfpr results into file
        attack = os.path.splitext(os.path.basename(results_file))[0]
        domain = os.path.basename(os.path.dirname(results_file))
        model = os.path.basename(os.path.dirname(os.path.dirname(results_file)))
        print(attack, domain, model)
        sub_dir = os.path.join(output_dir, model, domain)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir) 
        with open(os.path.join(sub_dir, f"{attack}_corrected_tpr_lowfpr.json"), 'w') as out:
            json.dump({
                "auc_roc": roc_auc,
                "tpr_lowfpr": tpr_at_low_fpr
            }, out, indent = 4)
