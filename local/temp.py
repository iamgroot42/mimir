import argparse
import json 
import os
import datasets
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

ngrams = ["7-gram", "13-gram"]
ngram_file_map = {
    "7-gram": "ngram_7",
    "13-gram": "ngram_13"
}
base_dir = "/gscratch/h2lab/micdun/bff/deduped/wikiMIA/"

subset_overlaps = defaultdict(dict)
for ngram in ngrams:
    shard_0 = datasets.load_dataset("json", data_files=os.path.join(base_dir, ngram_file_map[ngram], "0", "WikiMIA128_nonmembers.jsonl.gz"), split="train")
    print(shard_0)
    shard_1 = datasets.load_dataset("json", data_files=os.path.join(base_dir, ngram_file_map[ngram], "1", "WikiMIA128_nonmembers.jsonl.gz"), split="train")
    
    assert shard_0["original"][0] == shard_1["original"][0] and shard_0["original"][1] == shard_1["original"][1]
    ngram_inclusion = [np.array(in0) | np.array(in1) for in0, in1 in zip(shard_0["ngram_inclusion"], shard_1["ngram_inclusion"])]
    individual_ngram_overlap = {text: np.mean(d[:200]) for text, d in zip(shard_0["original"], ngram_inclusion)}
    subset_overlaps[ngram] = individual_ngram_overlap

# Set Seaborn style
sns.set(style="whitegrid")
sns.set_context("paper", font_scale = 1.5, rc={'lines.markersize': 5, 'lines.linewidth': 3, 'axes.linewidth': 3})
subset_color_map = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3']
subset_name_map = ['Wikipedia', 'Github', 'Pubmed Central', 'Pile CC']

# Create a figure and axis
fig, axs = plt.subplots(2, 1, figsize=(8,8),gridspec_kw={'wspace':0.05,'hspace':0.05},layout='constrained')

for i, ngram in enumerate(ngrams):

    # Plot the histogram
    ino = subset_overlaps[ngram]
    hplt = sns.histplot(ax=axs[i], data=100 * np.array(list(ino.values())), bins=20, facecolor=subset_color_map[i], stat='probability')

    axs[i].set_xticks(np.array([0, 0.2, 0.4, 0.6, .8, 1]) * 100)
#         hplt.legend_.remove()
    axs[i].set_ylabel(ngram)
    if i == 0:
        axs[i].set_title("WikiMIA")
    # plt.hist(individual_ngram_overlap_values, bins=100, range=(0, 1))

x_ax = fig.supxlabel('% Overlap')
y_ax = fig.supylabel('Proportion of Data')

# Save the histograms
plt.savefig("wikiMIA_ngram_overlap_hist.png", bbox_inches='tight')