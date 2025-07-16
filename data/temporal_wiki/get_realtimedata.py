from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

revision_dates = [
    '2023-08-12',
    '2023-08-20',
    '2023-08-23',
    '2023-08-27',
    '2023-08-28',
    '2023-09-04',
    '2023-09-08',
    '2023-09-11',
    '2023-09-25',
    '2023-10-02',
    '2023-10-09',
    '2023-10-16',
    '2023-10-23',
    '2023-10-30',
    '2023-11-06',
    '2023-11-13',
    '2023-11-20',
    '2023-11-27',
    '2023-12-04',
    '2023-12-11',
    '2023-12-18',
    '2023-12-25',
    '2024-01-01',
    '2024-01-08',
    'main'
]

datasets = []
for revision in tqdm(revision_dates): 
    dataset = load_dataset("RealTimeData/wikitext_latest", revision=revision)["train"]
    datasets.append(dataset)

full_dataset = concatenate_datasets(datasets)
print(full_dataset)
full_dataset.to_json("wikitext_latest_full.json")     