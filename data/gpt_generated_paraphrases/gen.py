import argparse
import json
import os 
import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from tqdm import tqdm
from collections import defaultdict

SYSTEM_PROMPT = f"You are a helpful assistant."
# GPT4 variant: 
PROMPT_BY_DOMAIN = {
    "wikipedia_(en)": f"Please help me paraphrase the following text chunk from Wikipedia in a different but concise style. Importantly, for sentences containing specific details, make minimal changes and ensure all details are included correctly in the paraphrase. Use a similar number of words.\n\n",
    "arxiv": f"Please help me paraphrase the following text chunk from a research paper in a different style. Importantly, for sentences containing specific details like mathematical definitions or proofs, only make minimal changes and ensure these details are included exactly in the paraphrase.  If the paper includes a title or authors, please keep them in the rephrase. If not, please DO NOT make up a title. Use a similar number of words.\n\n",
    "hackernews": f"Please help me paraphrase the following conversation chunk from a thread in HackerNews while maintaining the conversational style. Follow this structure for each comment in the thread: [user] - [comment]. Ensure all user's comments are represented in the paraphrase. Make sure all details in each user's comments are included correctly in the paraphrase, such as links. Be specific and don't generalize.\n\n"
}
# PROMPT_BY_DOMAIN = {
#     "wikipedia_(en)": f"Please help me paraphrase the following text chunk from Wikipedia in a different style. In doing so, ensure that no more than ten consecutive words are repeated. Please also ensure there aren't 50 consecutive identical characters between the paraphrase and original text. Importantly, for sentences containing specific details, only make minimal changes and ensure these details are included correctly in the paraphrase. Try not to overgeneralize.\n\n",
#     "arxiv": f"I will share some text chunks from ArXiv preprints. Please help me paraphrase them in a DIFFERENT STYLE but preserve all important information. If the paper includes a title or authors, please keep them in the rephrase. If not, please DO NOT make up a title\n\n",
#     "hacker_news": f"Please help me paraphrase the following conversation chunk from a thread in HackerNews while maintaining the conversational style. Follow this structure for each comment in the thread: [user] - [comment]. Ensure all user's comments are represented in the paraphrase. Make sure all details in each user's comments are included correctly in the paraphrase, such as links. Be specific and don't generalize.\n\n"
# }

def api_inference(input, domain, trials):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    current_model = "gpt-4-turbo-preview" #"gpt-3.5-turbo"

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
        return openai.ChatCompletion.create(**kwargs)
    
    ans = completion_with_backoff(
        model=current_model,
        max_tokens=512,
        n=trials,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{PROMPT_BY_DOMAIN[domain]}{input}"}
        ]
    )
    response_texts = [choice["message"]["content"] for choice in ans['choices']]
    return response_texts

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
    parser.add_argument('--n', default=1, type=int, help='number of samples to paraphrase')
    parser.add_argument('--trials', default=5, type=int, help='number of paraphrases per sample')
    parser.add_argument('--output_dir', default='./', help='output directory to place generated paraphrases')
    args = parser.parse_args()

    benchmark_path = args.benchmark_path
    domain = args.domain
    n = args.n
    trials = args.trials
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load in our member samples
    members = load(benchmark_path)

    # check if output file already exists
    output_file = os.path.join(output_dir, f"{domain}_paraphrases_{n}_samples_{trials}_trials.jsonl")
    if os.path.exists(output_file):
        print("using checkpoint")
        paraphrased_members = load_jsonl(output_file)
        existing_len = len(paraphrased_members)
        print(f"{existing_len} existing paraphrases")

        # Make sure there isn't mismatch in checkpoint length
        assert existing_len <= n

        # Only need to paraphrase remainder of samples
        members_sample = members[existing_len:n]

    else:
        print("generating from scratch")
        paraphrased_members = []

        # Only paraphrase the first n
        members_sample = members[:n]
    
    for m in tqdm(members_sample, desc='paraphrasing members'):
        paraphrases = api_inference(m, domain, trials)
        paraphrased_members.append({
            "original": m,
            "paraphrases": paraphrases
        })
    
        # Write every generated paraphrase to output_file
        write(paraphrased_members, output_file)
    

    # Write a version compatible with edited members script
    em_version = defaultdict(lambda: defaultdict(list))
    for pm in paraphrased_members:
        for i, p in enumerate(pm['paraphrases']):
            em_version['gpt'][str(i)].append(p)

    assert len(em_version['gpt']['0']) == n
    
    #json.dump(em_version, os.path.join(output_dir, f"em_version_{domain}_paraphrases_{n}_samples_{trials}_trials.jsonl"))