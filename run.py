"""
    Main entry point for running experiments with MIMIR
"""
import numpy as np
import torch
from tqdm import tqdm
import datetime
import os
import json
import math
from collections import defaultdict
from typing import List, Dict

from simple_parsing import ArgumentParser
from pathlib import Path

from mimir.config import (
    ExperimentConfig,
    EnvironmentConfig,
    NeighborhoodConfig,
    ReferenceConfig,
    OpenAIConfig, 
    ReCaLLConfig
)
import mimir.data_utils as data_utils
import mimir.plot_utils as plot_utils
from mimir.utils import fix_seed
from mimir.models import LanguageModel, ReferenceModel, OpenAI_APIModel
from mimir.attacks.all_attacks import AllAttacks, Attack
from mimir.attacks.utils import get_attacker
from mimir.attacks.attack_utils import (
    get_roc_metrics,
    get_precision_recall_metrics,
    get_auc_from_thresholds,
)


def get_attackers(
    target_model,
    ref_models,
    config: ExperimentConfig,
):
    # Look at all attacks, and attacks that we have implemented
    attacks = config.blackbox_attacks
    implemented_blackbox_attacks = [a.value for a in AllAttacks]
    # check for unimplemented attacks
    runnable_attacks = []
    for a in attacks:
        if a not in implemented_blackbox_attacks:
            print(f"Attack {a} not implemented, will be ignored")
            pass
        runnable_attacks.append(a)
    attacks = runnable_attacks

    # Initialize attackers
    attackers = {}
    for attack in attacks:
        if attack != AllAttacks.REFERENCE_BASED:
            attackers[attack] = get_attacker(attack)(config, target_model)

    # Initialize reference-based attackers if specified
    if ref_models is not None:
        for name, ref_model in ref_models.items():
            attacker = get_attacker(AllAttacks.REFERENCE_BASED)(
                config, target_model, ref_model
            )
            attackers[f"{AllAttacks.REFERENCE_BASED}-{name.split('/')[-1]}"] = attacker
    return attackers


def get_mia_scores(
    data,
    attackers_dict: Dict[str, Attack],
    ds_object,
    target_model: LanguageModel,
    ref_models: Dict[str, ReferenceModel],
    config: ExperimentConfig,
    is_train: bool,
    n_samples: int = None,
    batch_size: int = 50,
    **kwargs
):
    # Fix randomness
    fix_seed(config.random_seed)

    n_samples = len(data["records"]) if n_samples is None else n_samples

    # Look at all attacks, and attacks that we have implemented
    neigh_config = config.neighborhood_config

    if neigh_config:
        n_perturbation_list = neigh_config.n_perturbation_list
        in_place_swap = neigh_config.original_tokenization_swap

    results = []
    neighbors = None
    if AllAttacks.NEIGHBOR in attackers_dict.keys() and neigh_config.load_from_cache:
        neighbors = data[f"neighbors"]
        print("Loaded neighbors from cache!")

    collected_neighbors = {
        n_perturbation: [] for n_perturbation in n_perturbation_list
    }

    recall_config = config.recall_config
    if recall_config:
        nonmember_prefix = kwargs.get("nonmember_prefix", None)
        num_shots = recall_config.num_shots
        avg_length = int(np.mean([len(target_model.tokenizer.encode(ex)) for ex in data["records"]]))
        recall_dict = {"prefix":nonmember_prefix, "num_shots":num_shots, "avg_length":avg_length}

    # For each batch of data
    # TODO: Batch-size isn't really "batching" data - change later
    for batch in tqdm(range(math.ceil(n_samples / batch_size)), desc=f"Computing criterion"):
        texts = data["records"][batch * batch_size : (batch + 1) * batch_size]

        # For each entry in batch
        for idx in range(len(texts)):
            sample_information = defaultdict(list)
            sample = (
                texts[idx][: config.max_substrs]
                if config.full_doc
                else [texts[idx]]
            )

            # This will be a list of integers if pretokenized
            sample_information["sample"] = sample
            if config.pretokenized:
                detokenized_sample = [target_model.tokenizer.decode(s) for s in sample]
                sample_information["detokenized"] = detokenized_sample

            # For each substring
            neighbors_within = {n_perturbation: [] for n_perturbation in n_perturbation_list}
            for i, substr in enumerate(sample):
                # compute token probabilities for sample
                s_tk_probs, s_all_probs = (
                    target_model.get_probabilities(substr, return_all_probs=True)
                    if not config.pretokenized
                    else target_model.get_probabilities(
                        detokenized_sample[i], tokens=substr, return_all_probs=True
                    )
                )

                # Always compute LOSS score. Also helpful for reference-based and many other attacks.
                loss = (
                    target_model.get_ll(substr, probs=s_tk_probs)
                    if not config.pretokenized
                    else target_model.get_ll(
                        detokenized_sample[i], tokens=substr, probs=s_tk_probs
                    )
                )
                sample_information[AllAttacks.LOSS].append(loss)

                # TODO: Shift functionality into each attack entirely, so that this is just a for loop
                # For each attack
                for attack, attacker in attackers_dict.items():
                    # LOSS already added above, Reference handled later
                    if attack.startswith(AllAttacks.REFERENCE_BASED) or attack == AllAttacks.LOSS:
                        continue

                    if attack == AllAttacks.RECALL:
                        score = attacker.attack(
                            substr,
                            probs = s_tk_probs,
                            detokenized_sample=(
                                detokenized_sample[i]
                                if config.pretokenized
                                else None
                            ),
                            loss=loss,
                            all_probs=s_all_probs,
                            recall_dict = recall_dict
                        )
                        sample_information[attack].append(score)


                    elif attack != AllAttacks.NEIGHBOR:
                        score = attacker.attack(
                            substr,
                            probs=s_tk_probs,
                            detokenized_sample=(
                                detokenized_sample[i]
                                if config.pretokenized
                                else None
                            ),
                            loss=loss,
                            all_probs=s_all_probs,
                        )
                        sample_information[attack].append(score)
                        
                    else:
                        # For each 'number of neighbors'
                        for n_perturbation in n_perturbation_list:
                            # Use neighbors if available
                            if neighbors:
                                substr_neighbors = neighbors[n_perturbation][
                                    batch * batch_size + idx
                                ][i]
                            else:
                                substr_neighbors = attacker.get_neighbors(
                                    [substr], n_perturbations=n_perturbation
                                )
                                # Collect this neighbor information if neigh_config.dump_cache is True
                                if neigh_config.dump_cache:
                                    neighbors_within[n_perturbation].append(
                                        substr_neighbors
                                    )

                            if not neigh_config.dump_cache:
                                # Only evaluate neighborhood attack when not caching neighbors
                                score = attacker.attack(
                                    substr,
                                    probs=s_tk_probs,
                                    detokenized_sample=(
                                        detokenized_sample[i]
                                        if config.pretokenized
                                        else None
                                    ),
                                    loss=loss,
                                    batch_size=4,
                                    substr_neighbors=substr_neighbors,
                                )

                                sample_information[
                                    f"{attack}-{n_perturbation}"
                                ].append(score)

            if neigh_config and neigh_config.dump_cache:
                for n_perturbation in n_perturbation_list:
                    collected_neighbors[n_perturbation].append(
                        neighbors_within[n_perturbation]
                    )

            # Add the scores we collected for each sample for each
            # attack into to respective list for its classification
            results.append(sample_information)

    if neigh_config and neigh_config.dump_cache:
        # Save p_member_text and p_nonmember_text (Lists of strings) to cache
        # For each perturbation
        for n_perturbation in n_perturbation_list:
            ds_object.dump_neighbors(
                collected_neighbors[n_perturbation],
                train=is_train,
                num_neighbors=n_perturbation,
                model=neigh_config.model,
                in_place_swap=in_place_swap,
            )

    if neigh_config and neigh_config.dump_cache:
        print(
            "Data dumped! Please re-run with load_from_cache set to True in neigh_config"
        )
        exit(0)

    # Perform reference-based attacks
    if ref_models is not None:
        for name, ref_model in ref_models.items():
            ref_key = f"{AllAttacks.REFERENCE_BASED}-{name.split('/')[-1]}"
            attacker = attackers_dict.get(ref_key, None)
            if attacker is None:
                continue

            # Update collected scores for each sample with ref-based attack scores
            for r in tqdm(results, desc="Ref scores"):
                ref_model_scores = []
                for i, s in enumerate(r["sample"]):
                    if config.pretokenized:
                        s = r["detokenized"][i]
                    score = attacker.attack(s, probs=None,
                                                loss=r[AllAttacks.LOSS][i])
                    ref_model_scores.append(score)
                r[ref_key].extend(ref_model_scores)

            attacker.unload()
    else:
        print("No reference models specified, skipping Reference-based attacks")

    # Rearrange the nesting of the results dict and calculated aggregated score for sample
    # attack -> member/nonmember -> list of scores
    samples = []
    predictions = defaultdict(lambda: [])
    for r in results:
        samples.append(r["sample"])
        for attack, scores in r.items():
            if attack != "sample" and attack != "detokenized":
                # TODO: Is there a reason for the np.min here?
                predictions[attack].append(np.min(scores))

    return predictions, samples


def compute_metrics_from_scores(
        preds_member: dict,
        preds_nonmember: dict,
        samples_member: List,
        samples_nonmember: List,
        n_samples: int):

    attack_keys = list(preds_member.keys())
    if attack_keys != list(preds_nonmember.keys()):
        raise ValueError("Mismatched attack keys for member/nonmember predictions")

    # Collect outputs for each attack
    blackbox_attack_outputs = {}
    for attack in attack_keys:
        preds_member_ = preds_member[attack]
        preds_nonmember_ = preds_nonmember[attack]

        fpr, tpr, roc_auc, roc_auc_res, thresholds = get_roc_metrics(
            preds_member=preds_member_,
            preds_nonmember=preds_nonmember_,
            perform_bootstrap=True,
            return_thresholds=True,
        )
        tpr_at_low_fpr = {
            upper_bound: tpr[np.where(np.array(fpr) < upper_bound)[0][-1]]
            for upper_bound in config.fpr_list
        }
        p, r, pr_auc = get_precision_recall_metrics(
            preds_member=preds_member_,
            preds_nonmember=preds_nonmember_
        )

        print(
            f"{attack}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}, tpr_at_low_fpr: {tpr_at_low_fpr}"
        )
        blackbox_attack_outputs[attack] = {
            "name": f"{attack}_threshold",
            "predictions": {
                "member": preds_member_,
                "nonmember": preds_nonmember_,
            },
            "info": {
                "n_samples": n_samples,
            },
            "raw_results": (
                {"member": samples_member, "nonmember": samples_nonmember}
                if not config.pretokenized
                else []
            ),
            "metrics": {
                "roc_auc": roc_auc,
                "fpr": fpr,
                "tpr": tpr,
                "bootstrap_roc_auc_mean": np.mean(roc_auc_res.bootstrap_distribution),
                "bootstrap_roc_auc_std": roc_auc_res.standard_error,
                "tpr_at_low_fpr": tpr_at_low_fpr,
                "thresholds": thresholds,
            },
            "pr_metrics": {
                "pr_auc": pr_auc,
                "precision": p,
                "recall": r,
            },
            "loss": 1 - pr_auc,
        }

    return blackbox_attack_outputs


def generate_data_processed(
    base_model,
    mask_model,
    raw_data_member,
    batch_size: int,
    raw_data_non_member: List[str] = None
):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "nonmember": [],
        "member": [],
    }

    seq_lens = []
    num_batches = (len(raw_data_member) // batch_size) + 1
    iterator = tqdm(range(num_batches), desc="Generating samples")
    for batch in iterator:
        member_text = raw_data_member[batch * batch_size : (batch + 1) * batch_size]
        non_member_text = raw_data_non_member[batch * batch_size : (batch + 1) * batch_size]

        # TODO make same len
        for o, s in zip(non_member_text, member_text):
            # o, s = data_utils.trim_to_shorter_length(o, s, config.max_words)

            # # add to the data
            # assert len(o.split(' ')) == len(s.split(' '))
            if not config.full_doc:
                seq_lens.append((len(s.split(" ")), len(o.split())))

            if config.tok_by_tok:
                for tok_cnt in range(len(o.split(" "))):
                    data["nonmember"].append(" ".join(o.split(" ")[: tok_cnt + 1]))
                    data["member"].append(" ".join(s.split(" ")[: tok_cnt + 1]))
            else:
                data["nonmember"].append(o)
                data["member"].append(s)

    # if config.tok_by_tok:
    n_samples = len(data["nonmember"])
    # else:
    #     n_samples = config.n_samples
    if config.pre_perturb_pct > 0:
        print(
            f"APPLYING {config.pre_perturb_pct}, {config.pre_perturb_span_length} PRE-PERTURBATIONS"
        )
        print("MOVING MASK MODEL TO GPU...", end="", flush=True)
        mask_model.load()
        data["member"] = mask_model.generate_neighbors(
            data["member"],
            config.pre_perturb_span_length,
            config.pre_perturb_pct,
            config.chunk_size,
            ceil_pct=True,
        )
        print("MOVING BASE MODEL TO GPU...", end="", flush=True)
        base_model.load()

    return data, seq_lens, n_samples


def generate_data(
    dataset: str,
    train: bool = True,
    presampled: str = None,
    specific_source: str = None,
    mask_model_tokenizer = None
):
    data_obj = data_utils.Data(dataset, config=config, presampled=presampled)
    data = data_obj.load(
        train=train,
        mask_tokenizer=mask_model_tokenizer,
        specific_source=specific_source,
    )
    return data_obj, data
    # return generate_samples(data[:n_samples], batch_size=batch_size)


def main(config: ExperimentConfig):
    env_config: EnvironmentConfig = config.env_config
    neigh_config: NeighborhoodConfig = config.neighborhood_config
    ref_config: ReferenceConfig = config.ref_config
    openai_config: OpenAIConfig = config.openai_config
    recall_config: ReCaLLConfig = config.recall_config

    if openai_config:
        openAI_model = OpenAI_APIModel(config)

    if openai_config is not None:
        import openai

        assert openai_config.key is not None, "Must provide OpenAI API key"
        openai.api_key = openai_config.key

    START_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
    START_TIME = datetime.datetime.now().strftime("%H-%M-%S-%f")

    # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
    # create it if it doesn't exist
    output_subfolder = f"{config.output_name}/"
    if openai_config is None:
        base_model_name = config.base_model.replace("/", "_")
    else:
        base_model_name = "openai-" + openai_config.model.replace("/", "_")

    exp_name = config.experiment_name

    # Add pile source to suffix, if provided
    # TODO: Shift dataset-specific processing to their corresponding classes
    # Results go under target model
    sf = os.path.join(exp_name, config.base_model.replace("/", "_"))
    if config.specific_source is not None:
        processed_source = data_utils.sourcename_process(config.specific_source)
        sf = os.path.join(sf, processed_source)
    SAVE_FOLDER = os.path.join(env_config.tmp_results, sf)

    new_folder = os.path.join(env_config.results, sf)
    ##don't run if exists!!!
    print(f"{new_folder}")
    if os.path.isdir((new_folder)):
        print(f"HERE folder exists, not running this exp {new_folder}")
        exit(0)

    if not (os.path.exists(SAVE_FOLDER) or config.dump_cache):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    if neigh_config:
        n_perturbation_list = neigh_config.n_perturbation_list
        in_place_swap = neigh_config.original_tokenization_swap
        # n_similarity_samples = args.n_similarity_samples # NOT USED

    cache_dir = env_config.cache_dir
    print(f"LOG: cache_dir is {cache_dir}")
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # generic generative model
    base_model = LanguageModel(config)

    # reference model if we are doing the ref-based attack
    ref_models = None
    if (
        ref_config is not None
        and AllAttacks.REFERENCE_BASED in config.blackbox_attacks
    ):
        ref_models = {
            model: ReferenceModel(config, model) for model in ref_config.models
        }

    # Prepare attackers
    attackers_dict = get_attackers(base_model, ref_models, config)

    # Load neighborhood attack model, only if we are doing the neighborhood attack AND generating neighbors
    mask_model = None
    if (
        neigh_config
        and (not neigh_config.load_from_cache)
        and (AllAttacks.NEIGHBOR in config.blackbox_attacks)
    ):
        attacker_ne = attackers_dict[AllAttacks.NEIGHBOR]
        mask_model = attacker_ne.get_mask_model()

    print("MOVING BASE MODEL TO GPU...", end="", flush=True)
    base_model.load()

    print(f"Loading dataset {config.dataset_nonmember}...")
    # data, seq_lens, n_samples = generate_data(config.dataset_member)
    data_obj_nonmem, data_nonmember = generate_data(
        config.dataset_nonmember,
        train=False,
        presampled=config.presampled_dataset_nonmember,
        mask_model_tokenizer=mask_model.tokenizer if mask_model else None,
    )
    print(f"Loading dataset {config.dataset_member}...")
    data_obj_mem, data_member = generate_data(
        config.dataset_member,
        presampled=config.presampled_dataset_member,
        mask_model_tokenizer=mask_model.tokenizer if mask_model else None,
    )

    #* ReCaLL Specific
    if AllAttacks.RECALL in config.blackbox_attacks:
        assert recall_config, "Must provide a recall_config"
        num_shots = recall_config.num_shots
        nonmember_prefix = data_nonmember[:num_shots]
    else:
        nonmember_prefix = None


    other_objs, other_nonmembers = None, None
    if config.dataset_nonmember_other_sources is not None:
        other_objs, other_nonmembers = [], []
        for other_name in config.dataset_nonmember_other_sources:
            data_obj_nonmem_others, data_nonmember_others = generate_data(
                config.dataset_nonmember,
                train=False,
                specific_source=other_name,
                mask_model_tokenizer=mask_model.tokenizer if mask_model else None,
            )
            other_objs.append(data_obj_nonmem_others)
            other_nonmembers.append(data_nonmember_others)

    if config.dump_cache and not (config.load_from_cache or config.load_from_hf):
        print("Data dumped! Please re-run with load_from_cache set to True")
        exit(0)

    if config.pretokenized:
        assert data_member.shape == data_nonmember.shape
        data = {
            "nonmember": data_nonmember,
            "member": data_member,
        }
        n_samples, seq_lens = data_nonmember.shape
    else:
        data, seq_lens, n_samples = generate_data_processed(
            base_model, mask_model,
            data_member,
            batch_size=config.batch_size,
            raw_data_non_member=data_nonmember,
        )

    # If neighborhood attack is used, see if we have cache available (and load from it, if we do)
    neighbors_nonmember, neighbors_member = None, None
    if (
        AllAttacks.NEIGHBOR in config.blackbox_attacks
        and neigh_config.load_from_cache
    ):
        neighbors_nonmember, neighbors_member = {}, {}
        for n_perturbations in n_perturbation_list:
            neighbors_nonmember[n_perturbations] = data_obj_nonmem.load_neighbors(
                train=False,
                num_neighbors=n_perturbations,
                model=neigh_config.model,
                in_place_swap=in_place_swap,
            )
            neighbors_member[n_perturbations] = data_obj_mem.load_neighbors(
                train=True,
                num_neighbors=n_perturbations,
                model=neigh_config.model,
                in_place_swap=in_place_swap,
            )

    print("NEW N_SAMPLES IS ", n_samples)

    if mask_model is not None:
        attacker_ne.create_fill_dictionary(data)

    if config.scoring_model_name:
        print(f"Loading SCORING model {config.scoring_model_name}...")
        del base_model
        # Clear CUDA cache
        torch.cuda.empty_cache()

        base_model = LanguageModel(config, name=config.scoring_model_name)
        print("MOVING BASE MODEL TO GPU...", end="", flush=True)
        base_model.load()

    # Add neighbordhood-related data to 'data' here if we want it to be saved in raw data. Otherwise, add jsut before calling attack

    # write the data to a json file in the save folder
    if not config.pretokenized:
        with open(os.path.join(SAVE_FOLDER, "raw_data.json"), "w") as f:
            print(f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data.json')}")
            json.dump(data, f)

        with open(os.path.join(SAVE_FOLDER, "raw_data_lens.json"), "w") as f:
            print(
                f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data_lens.json')}"
            )
            json.dump(seq_lens, f)

    # TODO: Remove below if not needed/used
    """
    tk_freq_map = None
    if config.token_frequency_map is not None:
        print("loading tk freq map")
        tk_freq_map = pickle.load(open(config.token_frequency_map, "rb"))
    """

    # TODO: Instead of extracting from 'data', construct directly somewhere above
    data_members = {
        "records": data["member"],
        "neighbors": neighbors_member,
    }
    data_nonmembers = {
        "records": data["nonmember"],
        "neighbors": neighbors_nonmember,
    }

    outputs = []
    if config.blackbox_attacks is None:
        raise ValueError("No blackbox attacks specified in config!")

    # Collect scores for members
    member_preds, member_samples = get_mia_scores(
        data_members,
        attackers_dict,
        data_obj_mem,
        target_model=base_model,
        ref_models=ref_models,
        config=config,
        is_train=True,
        n_samples=n_samples,
        nonmember_prefix = nonmember_prefix
    )
    # Collect scores for non-members
    nonmember_preds, nonmember_samples = get_mia_scores(
        data_nonmembers,
        attackers_dict,
        data_obj_nonmem,
        target_model=base_model,
        ref_models=ref_models,
        config=config,
        is_train=False,
        n_samples=n_samples,
        nonmember_prefix = nonmember_prefix
    )
    blackbox_outputs = compute_metrics_from_scores(
        member_preds,
        nonmember_preds,
        member_samples,
        nonmember_samples,
        n_samples=n_samples,
    )

    # TODO: For now, AUCs for other sources of non-members are only printed (not saved)
    # Will fix later!
    if config.dataset_nonmember_other_sources is not None:
        # Using thresholds returned in blackbox_outputs, compute AUCs and ROC curves for other non-member sources
        for other_obj, other_nonmember, other_name in zip(
            other_objs, other_nonmembers, config.dataset_nonmember_other_sources
        ):
            other_nonmem_preds, _ = get_mia_scores(
                other_nonmember,
                attackers_dict,
                other_obj,
                target_model=base_model,
                ref_models=ref_models,
                config=config,
                is_train=False,
                n_samples=n_samples,
            )

            for attack in blackbox_outputs.keys():
                member_scores = np.array(
                    member_preds[attack]["predictions"]["member"]
                )
                thresholds = blackbox_outputs[attack]["metrics"]["thresholds"]
                nonmember_scores = np.array(other_nonmem_preds[attack])
                auc = get_auc_from_thresholds(
                    member_scores, nonmember_scores, thresholds
                )
                print(
                    f"AUC using thresholds of original split on {other_name} using {attack}: {auc}"
                )
        exit(0)

    # Dump main config into SAVE_FOLDER
    config.save_json(os.path.join(SAVE_FOLDER, 'config.json'), indent=4)

    for attack, output in blackbox_outputs.items():
        outputs.append(output)
        with open(os.path.join(SAVE_FOLDER, f"{attack}_results.json"), "w") as f:
            json.dump(output, f)

    neighbor_model_name = neigh_config.model if neigh_config else None
    plot_utils.save_roc_curves(
        outputs,
        save_folder=SAVE_FOLDER,
        model_name=base_model_name,
        neighbor_model_name=neighbor_model_name,
    )
    plot_utils.save_ll_histograms(outputs, save_folder=SAVE_FOLDER)
    plot_utils.save_llr_histograms(outputs, save_folder=SAVE_FOLDER)

    # move results folder from env_config.tmp_results to results/, making sure necessary directories exist
    if not os.path.exists(os.path.dirname(new_folder)):
        os.makedirs(os.path.dirname(new_folder))
    os.rename(SAVE_FOLDER, new_folder)

    api_calls = 0
    if openai_config:
        api_calls = openai_config.api_calls
        print(f"Used an *estimated* {api_calls} API tokens (may be inaccurate)")


if __name__ == "__main__":
    # Extract relevant configurations from config file
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--config", help="Path to attack config file", type=Path)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    config = ExperimentConfig.load(args.config, drop_extra_fields=False)
    # Also give user the option to provide config values over CLI
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(ExperimentConfig, dest="exp_config", default=config)
    args = parser.parse_args(remaining_argv)
    config: ExperimentConfig = args.exp_config

    # Fix randomness
    fix_seed(config.random_seed)
    # Call main function
    main(config)
