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


_HOMEPAGE = "http://github.com/iamgroot42/mimir"

_DESCRIPTION = """\
Member and non-member splits for our MI experiments using MIMIR. Data is available for each source.
We also cache neighbors (generated for the NE attack).
"""

_CITATION = """\
@article{duan2024do,
  title={Do Membership Inference Attacks Work on Large Language Models?},
  author={Duan*, Michael and \textbf{A. Suri*} and Mireshghallah, Niloofar and Min, Sewon and Shi, Weijia and Zettlemoyer, Luke and Tsvetkov, Yulia and Choi, Yejin and Evans, David and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:???},
  year={2024}
}
"""

_DOWNLOAD_URL = "https://huggingface.co/datasets/iamgroot42/mimir/resolve/main/"


class MimirConfig(BuilderConfig):
    """BuilderConfig for Mimir dataset."""

    def __init__(self, **kwargs):
        """Constructs a MimirConfig.

        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(MimirConfig, self).__init__(**kwargs)


class MimirDataset(GeneratorBasedBuilder):
    # Assuming 'VERSION' is defined
    VERSION = datasets.Version("1.0.0")

    # Define the builder configs
    BUILDER_CONFIG_CLASS = MimirConfig
    BUILDER_CONFIGS = [
        MimirConfig(
            name="the_pile_arxiv", description="This split contains data from Arxiv"
        ),
        MimirConfig(
            name="the_pile_full_pile", description="This split contains data from multiple sources in the Pile",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {"text": datasets.Sequence(datasets.Value("string"))}
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # Citation for the dataset
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager):
        """Returns SplitGenerators."""
        # Path to the data files
        NEIGHBOR_SUFFIX = "_neighbors_25_bert_in_place_swap"
        parent_dir = (
            "cache_100_200_10000_512"
            if self.config.name == "the_pile_full_pile"
            else "cache_100_200_1000_512"
        )

        file_paths = {
            "member": os.path.join(parent_dir, "train", self.config.name + ".jsonl"),
            "nonmember": os.path.join(parent_dir, "test", self.config.name + ".jsonl"),
        }
        # Load neighbor splits if they exist
        # TODO: This is not correct (should be checking URL, not local file structure). Fix later
        if os.path.exists(
            os.path.join(
                parent_dir,
                "train_neighbors",
                self.config.name + f"{NEIGHBOR_SUFFIX}.jsonl",
            )
        ):
            # Assume if train nieghbors exist, test neighbors also exist
            file_paths["member_neighbors"] = os.path.join(
                parent_dir,
                "train_neighbors",
                self.config.name + f"{NEIGHBOR_SUFFIX}.jsonl",
            )
            file_paths["nonmember_neighbors"] = os.path.join(
                parent_dir,
                "test_neighbors",
                self.config.name + f"{NEIGHBOR_SUFFIX}.jsonl",
            )

        # Now that we know which files to load, download them
        download_paths = [_DOWNLOAD_URL + v for v in file_paths.values()]
        data_dir = dl_manager.download_and_extract(download_paths)

        splits = []
        for i, k in enumerate(file_paths.keys()):
            splits.append(SplitGenerator(name=k, gen_kwargs={"file_path": data_dir[i]}))
        return splits

    def _generate_examples(self, file_path):
        """Yields examples."""
        # Open the specified .jsonl file and read each line
        with open(file_path, "r") as f:
            for id, line in enumerate(f):
                data = json.loads(line)
                if type(data) != list:
                    data = [data]
                yield id, {"text": data}
