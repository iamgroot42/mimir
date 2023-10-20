"""
    Utility functions for attacks
"""
from typing import List
import torch
from collections import Counter
import math
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.stats import bootstrap


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


def apply_extracted_fills(masked_texts: List[str], extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])[:-1]
    labels = labels.view(-1)[1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean()


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    print(num_same, f1, precision, recall)
    return f1, precision, recall


def get_roc_metrics(preds_member, preds_nonmember, perform_bootstrap: bool=False): # fpr_list,
    preds_member_    = filter_out_nan(preds_member)
    preds_nonmember_ = filter_out_nan(preds_nonmember)
    total_preds = preds_member_ + preds_nonmember_
    # Assign label '0' to members for computation, since sklearn
    # expectes label '0' data to have lower values to get assigned that label
    # which is true for our attacks (lower loss for members, e.g.)
    total_labels = [0] * len(preds_member_) + [1] * len(preds_nonmember_)
    fpr, tpr, _ = roc_curve(total_labels, total_preds)
    roc_auc = auc(fpr, tpr)
    # tpr_at_low_fpr = {upper_bound: tpr[np.where(np.array(fpr) < upper_bound)[0][-1]] for upper_bound in fpr_list}

    if perform_bootstrap:
        def roc_auc_statistic(preds, labels):
            in_preds  = [pred for pred, label in zip(preds, labels) if label == 0]
            out_preds = [pred for pred, label in zip(preds, labels) if label == 1]
            _, _, roc_auc = get_roc_metrics(in_preds, out_preds)
            return roc_auc

        auc_roc_res = bootstrap((total_preds, total_labels), roc_auc_statistic, n_resamples=1000, paired=True)
        
        # tpr_at_low_fpr_res = {}    
        # for ub in fpr_list:
        #     def tpr_at_fpr_statistic(preds, labels):
        #         in_preds = [pred for pred, label in zip(preds, labels) if label == 1]
        #         out_preds = [pred for pred, label in zip(preds, labels) if label == 0]
        #         _, _, _, tpr_at_low_fpr = get_roc_metrics(in_preds, out_preds, [ub])
        #         return tpr_at_low_fpr[ub]
            
        #     tpr_at_low_fpr_res[ub] = bootstrap((total_preds, total_labels), tpr_at_fpr_statistic, n_resamples=1000, paired=True)
        return fpr.tolist(), tpr.tolist(), float(roc_auc), auc_roc_res #tpr_at_low_fpr, tpr_at_low_fpr_res
    
    return fpr.tolist(), tpr.tolist(), float(roc_auc) #, tpr_at_low_fpr


def get_precision_recall_metrics(preds_member, preds_nonmember):
    preds_member_    =  filter_out_nan(preds_member)
    preds_nonmember_ = filter_out_nan(preds_nonmember)
    total_preds = preds_member_ + preds_nonmember_

    total_labels = [0] * len(preds_member_) + [1] * len(preds_nonmember_)

    precision, recall, _ = precision_recall_curve(total_labels, total_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


def filter_out_nan(x):
    return [element for element in x if not math.isnan(element)]
