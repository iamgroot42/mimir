import argparse
import json
import os
from tqdm import tqdm
from collections import defaultdict

def iterate_files(root_dir):
    file_paths = []
    file_names = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            file_names.append(file_path[len(root_dir):])
    return zip(file_paths, file_names)

def custom_open_yield(path, suffix=".jsonl"):
    if suffix == ".jsonl":
        with open(path, 'r') as file:
            for line in file:
                dp = json.loads(line)
                yield dp
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--output_dir', type=str, default="./")

    args = parser.parse_args()
    print(args)
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    samples_by_month = defaultdict(list)
    for data_path, filename in tqdm(iterate_files(args.data_dir)):
        for dp in tqdm(custom_open_yield(data_path)):
            # yymm = dp['meta']['yymm']
            timestamp = dp['meta']['timestamp'].split('T')[0]
            timestamp_yymm = '-'.join(timestamp.split('-')[:-1])
            year = int(timestamp.split('-')[0])
            if year >= 2019:
                samples_by_month[timestamp_yymm].append(dp)

    print(samples_by_month.keys())
    for yymm, data in tqdm(samples_by_month.items(), desc='Samples per month'):
        output_file = f'arxiv_{yymm}.jsonl'
        with open(os.path.join(output_dir, output_file), 'w') as f:
            for line in tqdm(data):
                f.write(json.dumps(line) + "\n")