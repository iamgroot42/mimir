"""
    Data used for experiments with MIMIR. Processed train/test splits for models trained on the Pile (for now).
    Processing data at HF end.
"""

from datasets import (
    GeneratorBasedBuilder,
    SplitGenerator,
    DownloadManager,
    BuilderConfig,
)
import json
import os

import datasets

from typing import List


_HOMEPAGE = "http://github.com/iamgroot42/mimir"

_DESCRIPTION = """\
Member and non-member splits for our MI experiments using MIMIR. Data is available for each source.
We also cache neighbors (generated for the NE attack).
"""

_CITATION = """\
@article{duan2024membership,
      title={Do Membership Inference Attacks Work on Large Language Models?}, 
      author={Michael Duan and Anshuman Suri and Niloofar Mireshghallah and Sewon Min and Weijia Shi and Luke Zettlemoyer and Yulia Tsvetkov and Yejin Choi and David Evans and Hannaneh Hajishirzi},
      year={2024},
      journal={arXiv:2402.07841},
}
"""

_DOWNLOAD_URL = "https://huggingface.co/datasets/iamgroot42/mimir/resolve/main/"


class MimirConfig(BuilderConfig):
    """BuilderConfig for Mimir dataset."""

    def __init__(self, *args, subsets: List[str]=[], **kwargs):
        """Constructs a MimirConfig.

        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(MimirConfig, self).__init__(**kwargs)
        self.subsets = subsets


class MimirDataset(GeneratorBasedBuilder):
    # Assuming 'VERSION' is defined
    VERSION = datasets.Version("1.3.0")

    # Define the builder configs
    BUILDER_CONFIG_CLASS = MimirConfig
    BUILDER_CONFIGS = [
        MimirConfig(
            name="arxiv",
            subsets=["ngram_7_0.2", "ngram_13_0.2", "ngram_13_0.8"],
            description="This split contains data from the Pile's Arxiv subset at various n-gram overlap thresholds"
        ),
        MimirConfig(
            name="dm_mathematics",
            subsets=["ngram_7_0.2", "ngram_13_0.2", "ngram_13_0.8"],
            description="This split contains data from the Pile's DM Mathematics subset at various n-gram overlap thresholds"
        ),
        MimirConfig(
            name="github",
            subsets=["ngram_7_0.2", "ngram_13_0.2", "ngram_13_0.8"],
            description="This split contains data from the Pile's GitHub subset at various n-gram overlap thresholds"
        ),
        MimirConfig(
            name="hackernews", 
            subsets=["ngram_7_0.2", "ngram_13_0.2", "ngram_13_0.8"],
            description="This split contains data from the Pile's HackerNews subset at various n-gram overlap thresholds"
        ),
        MimirConfig(
            name="pile_cc", 
            subsets=["ngram_7_0.2", "ngram_13_0.2", "ngram_13_0.8"],
            description="This split contains data from the Pile's Pile CC subset at various n-gram overlap thresholds"
        ),
        MimirConfig(
            name="pubmed_central", 
            subsets=["ngram_7_0.2", "ngram_13_0.2", "ngram_13_0.8"],
            description="This split contains data from the Pile's PubMed Central subset at various n-gram overlap thresholds"
        ),
        MimirConfig(
            name="wikipedia_(en)",
            subsets=["ngram_7_0.2", "ngram_13_0.2", "ngram_13_0.8"],
            description="This split contains data from the Pile's Wikipedia subset at various n-gram overlap thresholds"
        ),
        MimirConfig(
            name="full_pile", description="This split contains data from multiple sources in the Pile",
        ),
        MimirConfig(
            name="c4", description="This split contains data the C4 dataset",
        ),
        MimirConfig(
            name="temporal_arxiv", 
            subsets=["2020_08", "2021_01", "2021_06", "2022_01", "2022_06", "2023_01", "2023_06"],
            description="This split contains benchmarks where non-members are selected from various months from 2020-08 and onwards",
        ),
        MimirConfig(
            name="temporal_wiki", description="This split contains benchmarks where non-members are selected from 2023-08 and onwards",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features({
                "member": datasets.Value("string"),
                "nonmember": datasets.Value("string"),
                "member_neighbors": datasets.Sequence(datasets.Value("string")),
                "nonmember_neighbors": datasets.Sequence(datasets.Value("string"))
            }),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager):
        """Returns SplitGenerators."""
        # Path to the data files
        NEIGHBOR_SUFFIX = "_neighbors_25_bert_in_place_swap"
        parent_dir = (
            "cache_100_200_10000_512"
            if self.config.name == "full_pile"
            else "cache_100_200_1000_512"
        )

        if len(self.config.subsets) > 0:
            suffixes = [f"{subset}" for subset in self.config.subsets]
        else:
            suffixes = ["none"]

        file_paths = {}
        for subset_split_suffix in suffixes:
            internal_fp = {}

            subset_split_suffix_use = f"_{subset_split_suffix}" if subset_split_suffix != "none" else ""

            # Add standard member and non-member paths
            internal_fp['member'] = os.path.join(parent_dir, "train", f"{self.config.name}{subset_split_suffix_use}.jsonl")
            internal_fp['nonmember'] = os.path.join(parent_dir, "test", f"{self.config.name}{subset_split_suffix_use}.jsonl")

            # Load associated neighbors
            internal_fp['member_neighbors'] = os.path.join(
                parent_dir,
                "train_neighbors",
                f"{self.config.name}{subset_split_suffix_use}{NEIGHBOR_SUFFIX}.jsonl",
            )
            internal_fp['nonmember_neighbors'] = os.path.join(
                parent_dir,
                "test_neighbors",
                f"{self.config.name}{subset_split_suffix_use}{NEIGHBOR_SUFFIX}.jsonl",
            )
            file_paths[subset_split_suffix] = internal_fp

        # Now that we know which files to load, download them
        data_dir = {}
        for k, v_dict in file_paths.items():
            download_paths = []
            for v in v_dict.values():
                download_paths.append(_DOWNLOAD_URL + v)
            paths = dl_manager.download_and_extract(download_paths)
            internal_dict = {k:v for k, v in zip(v_dict.keys(), paths)}
            data_dir[k] = internal_dict

        splits = []
        for k in suffixes:
            splits.append(SplitGenerator(name=k, gen_kwargs={"file_path_dict": data_dir[k]}))
        return splits

    def _generate_examples(self, file_path_dict):
        """Yields examples."""
        # Open all four files in file_path_dict and yield examples (one from each file) simultaneously
        with open(file_path_dict["member"], "r") as f_member, open(file_path_dict["nonmember"], "r") as f_nonmember, open(file_path_dict["member_neighbors"], "r") as f_member_neighbors, open(file_path_dict["nonmember_neighbors"], "r") as f_nonmember_neighbors:
            for id, (member, nonmember, member_neighbors, nonmember_neighbors) in enumerate(zip(f_member, f_nonmember, f_member_neighbors, f_nonmember_neighbors)):
                yield id, {
                    "member": json.loads(member),
                    "nonmember": json.loads(nonmember),
                    "member_neighbors": json.loads(member_neighbors),
                    "nonmember_neighbors": json.loads(nonmember_neighbors),
                }