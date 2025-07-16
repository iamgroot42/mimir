import argparse
import os
import json
import random

from collections import defaultdict
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('neighbors_path', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    print(args)

    neighbors_path = args.neighbors_path
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load in neighbors
    def load_jsonl(input_path):
        with open(input_path, 'r') as f:
            data = [json.loads(line) for line in tqdm(f)]
        return data

    neighbors = load_jsonl(neighbors_path)

    print(len(neighbors)) # num members
    # print(neighbors[0])
    print(len(neighbors[0])) # 1 perturb round per member
    # print(neighbors[0][0])
    print(len(neighbors[0][0])) # 25 neighbors per member per perturb round

    # for each sample, sample 5 random neighbors as our edited members
    # output in format results_dict[pct_masked][trial] to follow format for new_mi_exp
    out_file = os.path.join(output_dir, "arxiv_ne.json")
    results_dir = defaultdict(lambda: defaultdict(list))
    for member in neighbors:
        ne = member[0]
        ne_sample =random.sample(ne, 5)
        for i, sample in enumerate(ne_sample):
            results_dir["0.3"][str(i)].append(sample)

    with open(out_file, 'w') as out:
        json.dump(results_dir, out)

    print(results_dir.keys())
    print(results_dir["0.3"].keys())
    print(len(results_dir["0.3"]["0"]))



