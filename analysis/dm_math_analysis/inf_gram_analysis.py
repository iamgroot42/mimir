import argparse
import requests
import transformers
import time
import os
import json

from tqdm import tqdm

INF_GRAM_URL = 'https://api.infini-gram.io/'
        
def load(path):
    with open(path, 'r') as f:
        data = [line for line in f]
    return data

def load_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write(outputs, path):
    with open(path, "w") as f:
        for d in outputs:
            f.write(json.dumps(d) + "\n")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark_path', help='path to member samples')
    parser.add_argument('--domain', default='wikipedia_(en)', help='domain of text to be paraphrased')
    parser.add_argument('--corpus', type=str, help='corpus to count ngrams from')
    parser.add_argument('--n', default=1, type=int, help='n for ngram')
    parser.add_argument('--output_dir', default='./', help='output directory to place generated paraphrases')
    args = parser.parse_args()

    benchmark_path = args.benchmark_path
    domain = args.domain
    corpus = args.corpus
    n = args.n
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load in our member samples
    samples = load(benchmark_path)

    # check if output file already exists
    output_file = os.path.join(output_dir, f"{domain}-{n}-gram-freq.jsonl")
    if os.path.exists(output_file):
        print("using checkpoint")
        ngram_freqs = load_jsonl(output_file)
        existing_len = len(ngram_freqs)
        print(f"{existing_len} samples processed")

        # Make sure there isn't mismatch in checkpoint length
        assert existing_len <= len(samples)

        # Only need to get ngram freqs for remaining samples
        samples = samples[existing_len:]

    else:
        print("generating from scratch")
        ngram_freqs = []

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        "huggyllama/llama-7b", cache_dir="./analysis_cache")
    
    for sample in tqdm(samples, desc='getting ngram freqs'):
        tokens = tokenizer.encode(sample)
        print("Number of tokens: ", len(tokens))

        counts = []
        i = 0
        for i in range(len(tokens) - n + 1):
            payload = {
                'corpus': corpus,
                'query_type': 'count',
                'query_ids': tokens[i:i + n]
            }

            # Allow 5 retries
            retry = True
            retry_count = 0
            while retry:
                x = requests.post(INF_GRAM_URL, json=payload)
                resp = x.json()

                if "error" in resp.keys():
                    print(resp)
                    
                    retry_count += 1

                    if retry_count > 5:
                        print("Quitting since endpoint seems overloaded")
                        quit()
                    # Sleep if error is found
                    time.sleep(1)
                else:
                    counts.append(resp["count"])
                    retry = False

        # Record frequencies for sample
        ngram_freqs.append({
            "sample": sample,
            "counts": counts
        })
    
        # Write every generated paraphrase to output_file
        write(ngram_freqs, output_file)

    assert len(ngram_freqs) >= len(samples)
    print("FINISHED")
    
    #json.dump(em_version, os.path.join(output_dir, f"em_version_{domain}_paraphrases_{n}_samples_{trials}_trials.jsonl"))