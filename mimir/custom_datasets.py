import random
import datasets
import os
import json
from typing import List


SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed']


def load_pubmed(cache_dir):
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def load_cached(cache_dir, path: str, filename: str, min_length: int, max_length: int, n_samples: int, max_tokens: int):
    """"
        Read from cache if available. Used for certain pile sources and xsum
        to ensure fairness in comparison across attacks.runs.
    """
    file_path = os.path.join(cache_dir, f"cache_{min_length}_{max_length}_{n_samples}_{max_tokens}", path, filename + ".jsonl")
    if not os.path.exists(file_path):
        raise ValueError(f"Requested cache file {file_path} does not exist")
    data = load_data(file_path)
    return data


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def dump_to_cache(data: List[str], cache_dir, path, filename: str, min_length: int, max_length: int, n_samples: int, max_tokens: int):
    """
        Cache a file (one text sequence per line)
    """
    # Make sure path directory exists
    subdir = os.path.join(cache_dir, f"cache_{min_length}_{max_length}_{n_samples}_{max_tokens}", path)
    os.makedirs(subdir, exist_ok=True)
    # Dump to file
    # Since each datum has newlines in it potentially, use jsonl format
    save_data(os.path.join(subdir, filename + ".jsonl"), data)


def save_data(file_path, data):
    # Since each datum has newlines in it potentially, use jsonl format
    with open(file_path, 'w') as f:
        for datum in data:
            f.write(json.dumps(datum) + "\n")


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writing(cache_dir=None):
    writing_path = 'data/writingPrompts'
    
    with open(f'{writing_path}/valid.wp_source', 'r') as f:
        prompts = f.readlines()
    with open(f'{writing_path}/valid.wp_target', 'r') as f:
        stories = f.readlines()
    
    prompts = [process_prompt(prompt) for prompt in prompts]
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered


def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = datasets.load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')
