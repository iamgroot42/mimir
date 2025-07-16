import argparse
import datasets
import io
import json
import os
import transformers
import re
import numpy as np
from multiprocessing import Manager, Pool, cpu_count

from functools import partial

def process_texts(shard_file, d):
    print(f"processing {shard_file}")
    dataset = datasets.load_dataset("json", data_files=shard_file, split='train', streaming=True)
    for dp in dataset:
        pile_subset = dp["meta"]["pile_set_name"].replace(" ", "_").replace("-", "_").lower()
        if pile_subset in d:
            d[pile_subset].append(dp)

def write(file_path, data):
    # open file in write mode
    with open(file_path, "w") as fp:
        for d in data:
            fp.write(json.dumps(d) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('shard_files', default=[], nargs="*")
    parser.add_argument('--output_dir', type=str, default="./")
    parser.add_argument('--subsets', type=str)

    args = parser.parse_args()
    print(args)

    # don't cache pile to avoid disk usage
    datasets.disable_caching()

    manager = Manager()
    d = manager.dict()

    subsets = args.subsets.split("+")
    for subset in subsets:
        d[subset] = manager.list()
    
    print(d.keys())

    pool = Pool(cpu_count())
    func = partial(process_texts, d=d)
    pool.map(func, args.shard_files)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 

    print("writing data")
    for subset, samples in d.items():
        write(os.path.join(args.output_dir, f"{subset}.jsonl"), samples)

