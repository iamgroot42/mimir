import argparse
import json 
import os
import datasets
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

OUT = "ngram_metadata.json"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs="*")
    parser.add_argument('--subset_overlap_results_dir', type=str)
    args = parser.parse_args()
    print(args)
    dirs = args.dirs
    results_dir = args.subset_overlap_results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    for dir in dirs:
        subset = os.path.split(dir)[-1]
        ngram = os.path.split(os.path.split(dir)[0])[-1]
        subset_results_dir = os.path.join(results_dir, subset, ngram)
        if not os.path.exists(subset_results_dir):
            os.makedirs(subset_results_dir)
        metadata = defaultdict(dict)
        for split in ["train", "val", "test"]:
            print(f"Loading for split {split}")
            shard_0 = datasets.load_dataset("json", data_files=os.path.join(dir, "0", f"{split}_text.jsonl.gz"), split="train")
            shard_1 = datasets.load_dataset("json", data_files=os.path.join(dir, "1", f"{split}_text.jsonl.gz"), split="train")
            assert shard_0["original"][0] == shard_1["original"][0] and shard_0["original"][1] == shard_1["original"][1]
            ngram_inclusion = [np.array(in0) | np.array(in1) for in0, in1 in zip(shard_0["ngram_inclusion"], shard_1["ngram_inclusion"])]
            individual_ngram_overlap = {text: np.mean(d) for text, d in zip(shard_0["original"], ngram_inclusion)}

            # metadata[split]["ngram_inclusion"] = ngram_inclusion
            metadata[split]["individual_ngram_overlap"] = individual_ngram_overlap
            individual_ngram_overlap_values = list(individual_ngram_overlap.values())
            # Set the figure size
            plt.figure()
            plt.rcParams["figure.figsize"] = [7.00, 3.50]
            plt.rcParams["figure.autolayout"] = True

            # Plot the histogram
            plt.hist(individual_ngram_overlap_values, bins=100, range=(0, 1))

            # Save the histogram
            plt.savefig(os.path.join(subset_results_dir, f"{split}_ngram_overlap_hist.png"))

            print(f"avg ngram overlap for split {split}:", np.mean(individual_ngram_overlap_values))
            print(f"median ngram overlap for split {split}:", np.median(individual_ngram_overlap_values))
            print(f"max ngram overlap for split {split}:", np.max(individual_ngram_overlap_values))
            print(f"min ngram overlap for split {split}:", np.min(individual_ngram_overlap_values))

            metadata[split]["average_overlap"] = np.mean(individual_ngram_overlap_values)
            metadata[split]["median_overlap"] = np.median(individual_ngram_overlap_values)
            metadata[split]["max_overlap"] = np.max(individual_ngram_overlap_values)
            metadata[split]["min_overlap"] = np.min(individual_ngram_overlap_values)
        with open(os.path.join(subset_results_dir, OUT), "w") as f:
            json.dump(metadata, f)