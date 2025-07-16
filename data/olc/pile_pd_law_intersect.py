import argparse
import datasets
import io
import json
import os
import transformers
import re
import numpy as np
from collections import defaultdict, Counter
from nltk.tokenize import WhitespaceTokenizer

from tqdm import tqdm

def write(file_path, data):
    # open file in write mode
    with open(file_path, "w") as fp:
        for d in data:
            fp.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_paths', default=[], nargs="*")
    parser.add_argument('--benchmark_dir', type=str, default="./")
    parser.add_argument('--ngram_overlap_threshold', type=float, default=0.9)

    args = parser.parse_args()
    print(args)

    output_dir = args.benchmark_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    data = defaultdict(list)
    for path in args.data_paths:
        suffix = os.path.basename(path)
        if suffix != "0.jsonl.gz":
            data[suffix].append(path)

    print("filtering intersection data")
    filtered_data = []
    bad_paths = []
    for suffix, paths in tqdm(data.items()):
        try:
            print(f"filtering out data with ngram overlap less than threshold {args.ngram_overlap_threshold}")
            # TODO: just assuming 2 shard files are passed in for now
            assert len(paths) == 2
            print(paths)
            shard_0_path, shard_1_path = paths[0], paths[1]
            shard_0 = datasets.load_dataset("json", data_files=shard_0_path, split="train[:100000]")
            shard_1 = datasets.load_dataset("json", data_files=shard_1_path, split="train[:100000]")
            assert shard_0["original"][0] == shard_1["original"][0] and shard_0["original"][1] == shard_1["original"][1]
            ngram_inclusion = [np.array(in0) | np.array(in1) for in0, in1 in tqdm(zip(shard_0["ngram_inclusion"], shard_1["ngram_inclusion"]))]
            data = [
                {"text": text, "meta": {"pile_set_name": "pd_law"}, "overlap": np.mean(d)} for text, d in tqdm(zip(shard_0["original"], ngram_inclusion)) if np.mean(d) >= args.ngram_overlap_threshold
            ]
            print("candidates:", len(data))
            filtered_data.extend(data)
        except:
            bad_paths.extend(paths)
    print("bad paths:", bad_paths)
    print(len(filtered_data))
    write(os.path.join(output_dir, f"pile_law+pd_law.jsonl"), filtered_data)
        
