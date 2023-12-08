"""
    Utilities related to plotting.
"""

import matplotlib.pyplot as plt
# Set high DPI
plt.rcParams['figure.dpi'] = 300


# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


def save_roc_curves(experiments, save_folder, model_name, neighbor_model_name: str = None):
    """
        Save the ROC curve for each experiment, given a list of output dictionaries, one for each experiment, using colorblind-friendly colors.
    """
    # first, clear plt
    plt.clf()

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}", color=color)
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if neighbor_model_name:
        plt.title(f'ROC Curves ({model_name} - {neighbor_model_name})')
    else:
        plt.title(f'ROC Curves ({model_name})')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{save_folder}/roc_curves.png")

    # Also plot ROC curves for low TPR-FPR region
    plt.clf()
    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}", color=color)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.plot([1e-5, 1], [1e-5, 1], color='black', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if neighbor_model_name:
        plt.title(f'ROC Curves ({model_name} - {neighbor_model_name}) : low FPR region')
    else:
        plt.title(f'ROC Curves ({model_name} : low FPR region')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{save_folder}/roc_curves_low_fpr.png")


def save_f1_histogram(f1_scores, save_folder):
    """
        Function for saving F1-score histograms.
    """
    plt.hist(f1_scores, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('F1 Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of F1 Scores')
    plt.savefig(f"{save_folder}/f1_hist.png")
    plt.close()


def save_ll_histograms(experiments, save_folder):
    """
        Save the histogram of log likelihoods in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed.
    """
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.hist([r["sampled_ll"] for r in results], alpha=0.5, bins='auto', label='member')
            plt.hist([r["perturbed_sampled_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed sampled')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist([r["original_ll"] for r in results], alpha=0.5, bins='auto', label='nonmember')
            plt.hist([r["perturbed_original_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed original')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(
                f"{save_folder}/ll_histograms_{experiment['name']}.png")
        except:
            pass


def save_llr_histograms(experiments, save_folder):
    """
        Save the histograms of log likelihood ratios in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed.
    """
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)

            # compute the log likelihood ratio for each result
            for r in results:
                r["sampled_llr"] = r["sampled_ll"] - r["perturbed_sampled_ll"]
                r["original_llr"] = r["original_ll"] - r["perturbed_original_ll"]
            
            plt.hist([r["sampled_llr"] for r in results], alpha=0.5, bins='auto', label='member')
            plt.hist([r["original_llr"] for r in results], alpha=0.5, bins='auto', label='nonmember')
            plt.xlabel("log likelihood ratio")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{save_folder}/llr_histograms_{experiment['name']}.png")
        except:
            pass

