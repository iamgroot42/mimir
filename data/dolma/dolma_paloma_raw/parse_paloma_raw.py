from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

import argparse
import os
import json
import numpy as np

def write(file_path, data):
    # open file in write mode
    with open(file_path, "w") as fp:
        for d in data:
            fp.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_paths', default=[], nargs="*")
    parser.add_argument('--source', type=str)
    parser.add_argument('--out_dir', type=str)

    args = parser.parse_args()
    print(args)

    data_paths = args.data_paths
    source = args.source
    out_dir = args.out_dir
    split = "val"
    num_proc = 4
    name="v1_5-sample"

    dataset = load_dataset('json', data_files=data_paths, split="train", num_proc=num_proc)
    print(dataset.features)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_data = [{"source": source, "text": sample['text']} for sample in tqdm(dataset)]
    write(os.path.join(out_dir, f"dolma-{name}-{source}-{split}.jsonl"), out_data)

    # data_by_source = defaultdict(list)
    # for sample in tqdm(dataset):
    #     source = sample["source"]
    #     text = sample["text"]
    #     data_by_source[source].append({
    #         "source": source,
    #         "text": text
    #     })

    # out_dir = "nonmember_raw"
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    # # check lengths make sense
    # by_source_len = np.sum([len(v)for v in data_by_source.values()])
    # assert len(dataset) == by_source_len

    # for source, samples in data_by_source.items():
    #     write(os.path.join(out_dir, f"dolma-{name}-{source}-{split}.jsonl"), samples)