import argparse
import json
import os
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from Levenshtein import distance


def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in tqdm(f)]

def write(outputs, path):
    with open(path, "w") as f:
        for d in outputs:
            f.write(json.dumps(d) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paraphrase_path', help='path to member samples')
    parser.add_argument('--output_dir', default='./', help='output directory to place generated paraphrases')
    args = parser.parse_args()

    paraphrase_path = args.paraphrase_path
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    analysis_dir = os.path.join(output_dir, "analysis")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    paraphrases = read_jsonl(paraphrase_path)

    # Write a version compatible with edited members script
    em_version = defaultdict(lambda: defaultdict(list))
    for pm in paraphrases:
        for i, p in enumerate(pm['paraphrases']):
            em_version['gpt'][str(i)].append(p)

    assert len(em_version['gpt']['0']) == 1000
    
    with open(os.path.join(output_dir, f"em_version_{os.path.basename(paraphrase_path)}"), 'w') as out:
        json.dump(em_version, out)

    print("outputted em_version")

    # Get average length of paraphrases
    lengths = []
    for pm in paraphrases:
        lengths.append({
            "original_len": len(pm['original'].split()),
            "avg_paraphrase_len": np.mean([len(p.split()) for p in pm['paraphrases']])
        }) 

    # print average delta in paraphrase length
    print("average delta in paraphrase length:", np.mean([l['original_len'] - l['avg_paraphrase_len'] for l in lengths]))
    write(lengths, os.path.join(analysis_dir, f"lengths_{os.path.basename(paraphrase_path)}"))

    # Get average word-based edit distance of paraphrases
    edit_distances = []
    for pm in tqdm(paraphrases):
        original = pm['original'].split()
        edit_distances.append({
            "avg_ed": np.mean([distance(original, p.split()) for p in pm['paraphrases']])
        })

    write(edit_distances, os.path.join(analysis_dir, f"ed_{os.path.basename(paraphrase_path)}"))