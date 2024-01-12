import numpy as np
import torch
from tqdm import tqdm
import random
import datetime
import os
import json
import pickle
import math
from collections import defaultdict
from functools import partial
from typing import List

from simple_parsing import ArgumentParser
from pathlib import Path

from mimir.config import (
    ExperimentConfig,
    EnvironmentConfig,
    NeighborhoodConfig,
    ReferenceConfig,
    OpenAIConfig,
    ExtractionConfig,
)
import mimir.data_utils as data_utils
import mimir.plot_utils as plot_utils
from mimir.models import EvalModel, LanguageModel, ReferenceModel, OpenAI_APIModel
from mimir.attacks.blackbox_attacks import BlackBoxAttacks
from mimir.attacks.neighborhood import T5Model, BertModel, NeighborhoodAttack
from mimir.attacks.attack_utils import (
    f1_score,
    get_roc_metrics,
    get_precision_recall_metrics,
)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_perturbation_results(span_length: int = 10, n_perturbations: int = 1):
    print("MOVING MASK MODEL TO GPU...", end="", flush=True)
    mask_model.load()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    nonmember_text = data["nonmember"]
    member_text = data["member"]

    ceil_pct = neigh_config.ceil_pct
    kwargs = {}
    if "t5" in neigh_config.model:
        kwargs = {
            "span_length": span_length,
            "pct": neigh_config.pct_words_masked,
            "chunk_size": config.chunk_size,
            "ceil_pct": ceil_pct,
        }
    kwargs["n_perturbations"] = n_perturbations

    in_place_swap = neigh_config.original_tokenization_swap

    if neigh_config.load_from_cache:
        # Load from cache, if available (and requested)
        p_member_text = data_obj_mem.load_neighbors(
            train=True,
            num_neighbors=n_perturbations,
            model=neigh_config.model,
            in_place_swap=in_place_swap,
        )
        p_nonmember_text = data_obj_nonmem.load_neighbors(
            train=False,
            num_neighbors=n_perturbations,
            model=neigh_config.model,
            in_place_swap=in_place_swap,
        )
    else:
        p_member_text = mask_model.generate_neighbors(member_text, **kwargs)
        p_nonmember_text = mask_model.generate_neighbors(nonmember_text, **kwargs)

        for _ in range(n_perturbation_rounds - 1):
            try:
                p_member_text = mask_model.generate_neighbors(p_member_text, **kwargs)
                p_nonmember_text = mask_model.generate_neighbors(
                    p_nonmember_text, **kwargs
                )
            except AssertionError:
                break

    # assert len(p_member_text) == len(member_text) * n_perturbations, f"Expected {len(member_text) * n_perturbations} perturbed samples, got {len(p_member_text)}"
    # assert len(p_nonmember_text) == len(nonmember_text) * n_perturbations, f"Expected {len(nonmember_text) * n_perturbations} perturbed samples, got {len(p_nonmember_text)}"

    if neigh_config.dump_cache:
        if extraction_config is not None:
            raise NotImplementedError("Caching not implemented for extraction yet")

        # Save p_member_text and p_nonmember_text (Lists of strings) to cache
        data_obj_mem.dump_neighbors(
            p_member_text,
            train=True,
            num_neighbors=n_perturbations,
            model=neigh_config.model,
            in_place_swap=in_place_swap,
        )
        data_obj_nonmem.dump_neighbors(
            p_nonmember_text,
            train=False,
            num_neighbors=n_perturbations,
            model=neigh_config.model,
            in_place_swap=in_place_swap,
        )

        print(
            "Data dumped! Please re-run with load_from_cache set to True in neigh_config"
        )
        exit(0)

    for idx in range(len(nonmember_text)):
        results.append(
            {
                "nonmember": nonmember_text[idx],
                "member": member_text[idx],
                "perturbed_member": p_member_text[
                    idx * n_perturbations : (idx + 1) * n_perturbations
                ],
                "perturbed_nonmember": p_nonmember_text[
                    idx * n_perturbations : (idx + 1) * n_perturbations
                ],
            }
        )

    print("MOVING BASE MODEL TO GPU...", end="", flush=True)
    base_model.load()
    # print('MOVING ref MODEL TO GPU...', end='', flush=True)
    # load_model(ref_model, env_config.device_aux)

    for res in tqdm(results, desc="Computing log likelihoods"):
        # Get likelihoods for perturbed samples
        p_member_ll = base_model.get_lls(res["perturbed_member"])
        p_nonmember_ll = base_model.get_lls(res["perturbed_nonmember"])
        res["all_perturbed_member_ll"] = p_member_ll
        res["all_perturbed_nonmember_ll"] = p_nonmember_ll
        # Get likelihoods for original samples
        res["nonmember_ll"] = base_model.get_ll(res["nonmember"])
        res["member_ll"] = base_model.get_ll(res["member"])
        # Average neighbor likelihoods
        res["perturbed_member_ll"] = np.mean(p_member_ll)
        res["perturbed_nonmember_ll"] = np.mean(p_nonmember_ll)
        # Standard deviation of neighbor likelihoods
        res["perturbed_member_ll_std"] = (
            np.std(p_member_ll) if len(p_member_ll) > 1 else 1
        )
        res["perturbed_nonmember_ll_std"] = (
            np.std(p_nonmember_ll) if len(p_nonmember_ll) > 1 else 1
        )

    return results


# TODO: change keys
def run_perturbation_experiment(
    results, criterion, n_samples: int, span_length: int = 10, n_perturbations: int = 1
):
    # compute diffs with perturbed
    predictions = {"member": [], "nonmember": []}
    for res in results:
        if criterion == "d":
            predictions["member"].append(res["member_ll"] - res["perturbed_member_ll"])
            predictions["nonmember"].append(
                res["nonmember_ll"] - res["perturbed_nonmember_ll"]
            )
        elif criterion == "z":
            if res["perturbed_member_ll_std"] == 0:
                res["perturbed_member_ll_std"] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(
                    f"Number of unique perturbed original texts: {len(set(res['perturbed_member']))}"
                )
                print(f"Member text: {res['member']}")
            if res["perturbed_nonmember_ll_std"] == 0:
                res["perturbed_nonmember_ll_std"] = 1
                print("WARNING: std of perturbed sampled is 0, setting to 1")
                print(
                    f"Number of unique perturbed sampled texts: {len(set(res['perturbed_nonmember']))}"
                )
                print(f"Nonmember text: {res['nonmember']}")
            predictions["member"].append(
                (res["member_ll"] - res["perturbed_member_ll"])
                / res["perturbed_member_ll_std"]
            )
            predictions["nonmember"].append(
                (res["nonmember_ll"] - res["perturbed_nonmember_ll"])
                / res["perturbed_nonmember_ll_std"]
            )

    fpr, tpr, roc_auc, roc_auc_res = get_roc_metrics(
        preds_member=predictions["member"],
        preds_nonmember=predictions["nonmember"],
        perform_bootstrap=True,
    )
    tpr_at_low_fpr = {
        upper_bound: tpr[np.where(np.array(fpr) < upper_bound)[0][-1]]
        for upper_bound in config.fpr_list
    }
    p, r, pr_auc = get_precision_recall_metrics(
        preds_member=predictions["member"], preds_nonmember=predictions["nonmember"]
    )
    name = f"perturbation_{n_perturbations}_{criterion}"
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        "name": name,
        "predictions": predictions,
        "info": {
            "pct_words_masked": neigh_config.pct_words_masked,
            "span_length": span_length,
            "n_perturbations": n_perturbations,
            "n_samples": n_samples,
        },
        "raw_results": results,
        "metrics": {
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
            "bootstrap_roc_auc_mean": np.mean(roc_auc_res.bootstrap_distribution),
            "bootstrap_roc_auc_std": roc_auc_res.standard_error,
            "tpr_at_low_fpr": tpr_at_low_fpr,
        },
        "pr_metrics": {
            "pr_auc": pr_auc,
            "precision": p,
            "recall": r,
        },
        "loss": 1 - pr_auc,
    }


# TODO: Might make more sense to have this function called once each for mem and non-mem, instead of handling all of them in a loop inside here
# For instance, makes it easy to know exact source of data
def run_blackbox_attacks(
    data,
    ds_objects,
    target_model,
    ref_models,
    config: ExperimentConfig,
    n_samples: int = None,
    batch_size: int = 50,
):
    torch.manual_seed(0)
    np.random.seed(0)

    n_samples = len(data["member"]) if n_samples is None else n_samples

    # Structure: attack -> member scores/nonmember scores
    # For both members and nonmembers, we compute all attacks
    # listed in config all together for each sample
    attacks = config.blackbox_attacks
    neigh_config = config.neighborhood_config
    implemented_blackbox_attacks = [a.value for a in BlackBoxAttacks]
    # check for unimplemented attacks
    runnable_attacks = []
    for a in attacks:
        if a not in implemented_blackbox_attacks:
            print(f"Attack {a} not implemented, will be ignored")
            pass

        runnable_attacks.append(a)
    attacks = runnable_attacks

    neighborhood_attacker = NeighborhoodAttack(config, target_model)
    neighborhood_attacker.prepare()

    results = defaultdict(list)
    keys_care_about = ["nonmember", "member"]
    for classification in keys_care_about:
        print(f"Running for classification {classification}")

        neighbors = None
        if BlackBoxAttacks.NEIGHBOR in attacks and neigh_config.load_from_cache:
            neighbors = data[f"{classification}_neighbors"]
            print("Loaded neighbors from cache!")

        collected_neighbors = {
            n_perturbation: [] for n_perturbation in n_perturbation_list
        }

        # For each batch of data
        # TODO: Batch-size isn't really "batching" data - change later
        for batch in tqdm(
            range(math.ceil(n_samples / batch_size)), desc=f"Computing criterion"
        ):
            texts = data[classification][batch * batch_size : (batch + 1) * batch_size]

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
                    detokenized_sample = [
                        target_model.tokenizer.decode(s) for s in sample
                    ]
                    sample_information["detokenized"] = detokenized_sample

                # For each substring
                neighbors_within = {
                    n_perturbation: [] for n_perturbation in n_perturbation_list
                }
                for i, substr in enumerate(sample):
                    # compute token probabilities for sample
                    s_tk_probs = (
                        target_model.get_probabilities(substr)
                        if not config.pretokenized
                        else target_model.get_probabilities(
                            detokenized_sample[i], tokens=substr
                        )
                    )

                    # always compute loss score
                    loss = (
                        target_model.get_ll(substr, probs=s_tk_probs)
                        if not config.pretokenized
                        else target_model.get_ll(
                            detokenized_sample[i], tokens=substr, probs=s_tk_probs
                        )
                    )
                    # TODO: Instead of doing this outside, set config default to always include LOSS
                    sample_information[BlackBoxAttacks.LOSS].append(loss)

                    # TODO: Shift functionality into each attack entirely, so that this is just a for loop
                    # For each attack
                    for attack in attacks:
                        if attack == BlackBoxAttacks.ZLIB:
                            score = (
                                target_model.get_zlib_entropy(substr, probs=s_tk_probs)
                                if not config.pretokenized
                                else target_model.get_zlib_entropy(
                                    detokenized_sample[i],
                                    tokens=substr,
                                    probs=s_tk_probs,
                                )
                            )
                            sample_information[attack].append(score)
                        elif attack == BlackBoxAttacks.MIN_K:
                            score = (
                                target_model.get_min_k_prob(substr, probs=s_tk_probs)
                                if not config.pretokenized
                                else target_model.get_min_k_prob(
                                    detokenized_sample[i],
                                    tokens=substr,
                                    probs=s_tk_probs,
                                )
                            )
                            sample_information[attack].append(score)
                        elif attack == BlackBoxAttacks.NEIGHBOR:
                            # For each 'number of neighbors'
                            for n_perturbation in n_perturbation_list:
                                # Use neighbors if available
                                if neighbors:
                                    substr_neighbors = neighbors[n_perturbation][
                                        batch * batch_size + idx
                                    ][i]
                                else:
                                    substr_neighbors = (
                                        neighborhood_attacker.get_neighbors(
                                            [substr], n_perturbations=n_perturbation
                                        )
                                    )
                                    # Collect this neighbor information if neigh_config.dump_cache is True
                                    if neigh_config.dump_cache:
                                        neighbors_within[n_perturbation].append(
                                            substr_neighbors
                                        )

                                if not neigh_config.dump_cache:
                                    # Only evaluate neighborhood attack when not caching neighbors
                                    mean_substr_score = target_model.get_lls(
                                        substr_neighbors, batch_size=15
                                    )
                                    d_based_score = loss - mean_substr_score

                                    sample_information[
                                        f"{attack}-{n_perturbation}"
                                    ].append(d_based_score)

                if neigh_config and neigh_config.dump_cache:
                    for n_perturbation in n_perturbation_list:
                        collected_neighbors[n_perturbation].append(
                            neighbors_within[n_perturbation]
                        )

                # Add the scores we collected for each sample for each
                # attack into to respective list for its classification
                results[classification].append(sample_information)

        if neigh_config and neigh_config.dump_cache:
            ds_obj_use = ds_objects[classification]

            # Save p_member_text and p_nonmember_text (Lists of strings) to cache
            # For each perturbation
            for n_perturbation in n_perturbation_list:
                ds_obj_use.dump_neighbors(
                    collected_neighbors[n_perturbation],
                    train=True if classification == "member" else False,
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
    if BlackBoxAttacks.REFERENCE_BASED in attacks:
        if ref_models is None:
            print("No reference models specified, skipping Reference-based attacks")
        else:
            for name, ref_model in ref_models.items():
                # if "llama" not in name and "alpaca" not in name:
                ref_model.load()

                # Update collected scores for each sample with ref-based attack scores
                for classification, result in results.items():
                    for r in tqdm(result, desc="Ref scores"):
                        ref_model_scores = []
                        for i, s in enumerate(r["sample"]):
                            if config.pretokenized:
                                s = r["detokenized"][i]
                            ref_score = r[BlackBoxAttacks.LOSS][i] - ref_model.get_ll(s)
                            ref_model_scores.append(ref_score)
                        r[
                            f"{BlackBoxAttacks.REFERENCE_BASED}-{name.split('/')[-1]}"
                        ].extend(ref_model_scores)

                # if "llama" not in name and "alpaca" not in name:
                ref_model.unload()

    # Rearrange the nesting of the results dict and calculated aggregated score for sample
    # attack -> member/nonmember -> list of scores
    samples = defaultdict(list)
    predictions = defaultdict(lambda: defaultdict(list))
    for classification, result in results.items():
        for r in result:
            samples[classification].append(r["sample"])
            for attack, scores in r.items():
                if attack != "sample" and attack != "detokenized":
                    predictions[attack][classification].append(np.min(scores))

    # Collect outputs for each attack
    blackbox_attack_outputs = {}
    for attack, prediction in tqdm(predictions.items()):
        fpr, tpr, roc_auc, roc_auc_res = get_roc_metrics(
            preds_member=prediction["member"],
            preds_nonmember=prediction["nonmember"],
            perform_bootstrap=True,
        )
        tpr_at_low_fpr = {
            upper_bound: tpr[np.where(np.array(fpr) < upper_bound)[0][-1]]
            for upper_bound in config.fpr_list
        }
        p, r, pr_auc = get_precision_recall_metrics(
            preds_member=prediction["member"], preds_nonmember=prediction["nonmember"]
        )

        print(
            f"{attack}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}, tpr_at_low_fpr: {tpr_at_low_fpr}"
        )
        blackbox_attack_outputs[attack] = {
            "name": f"{attack}_threshold",
            "predictions": prediction,
            "info": {
                "n_samples": n_samples,
            },
            "raw_results": samples if not config.pretokenized else [],
            "metrics": {
                "roc_auc": roc_auc,
                "fpr": fpr,
                "tpr": tpr,
                "bootstrap_roc_auc_mean": np.mean(roc_auc_res.bootstrap_distribution),
                "bootstrap_roc_auc_std": roc_auc_res.standard_error,
                "tpr_at_low_fpr": tpr_at_low_fpr,
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
    raw_data_member, batch_size, raw_data_non_member: List[str] = None
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
        if extraction_config is not None:
            non_member_text = base_model.sample_from_model(
                member_text,
                min_words=30
                if config.dataset_member in ["pubmed"]
                else config.min_words,
                max_words=config.max_words,
                prompt_tokens=extraction_config.prompt_len,
            )
        else:
            non_member_text = raw_data_non_member[
                batch * batch_size : (batch + 1) * batch_size
            ]

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


def generate_data(dataset: str, train: bool = True, presampled: str = None):
    data_obj = data_utils.Data(dataset, config=config, presampled=presampled)
    data = data_obj.load(train=train, mask_tokenizer=mask_model.tokenizer if mask_model else None)
    return data_obj, data
    # return generate_samples(data[:n_samples], batch_size=batch_size)


def eval_supervised(data, model):
    print(f"Beginning supervised evaluation with {model}...")

    real, fake = data["nonmember"], data["member"]

    # TODO: Fix init call below
    eval_model = EvalModel(model)

    real_preds = eval_model.get_preds(real)
    fake_preds = eval_model.get_preds(fake)

    predictions = {
        "real": real_preds,
        "samples": fake_preds,
    }

    fpr, tpr, roc_auc, roc_auc_res = get_roc_metrics(
        preds_member=real_preds, preds_nonmember=fake_preds, perform_bootstrap=True
    )
    tpr_at_low_fpr = {
        upper_bound: tpr[np.where(np.array(fpr) < upper_bound)[0][-1]]
        for upper_bound in config.fpr_list
    }
    p, r, pr_auc = get_precision_recall_metrics(
        preds_member=real_preds, preds_nonmember=fake_preds
    )
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

    del eval_model
    # Clear CUDA cache
    torch.cuda.empty_cache()

    return {
        "name": model,
        "predictions": predictions,
        "info": {
            "n_samples": n_samples,
        },
        "metrics": {
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
            "bootstrap_roc_auc_mean": np.mean(roc_auc_res.bootstrap_distribution),
            "bootstrap_roc_auc_std": roc_auc_res.standard_error,
            "tpr_at_low_fpr": tpr_at_low_fpr,
        },
        "pr_metrics": {
            "pr_auc": pr_auc,
            "precision": p,
            "recall": r,
        },
        "loss": 1 - pr_auc,
    }


if __name__ == "__main__":
    # TODO: Shift below to main() function - variables here are global and may interfe with functions etc.

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

    env_config: EnvironmentConfig = config.env_config
    neigh_config: NeighborhoodConfig = config.neighborhood_config
    ref_config: ReferenceConfig = config.ref_config
    openai_config: OpenAIConfig = config.openai_config
    extraction_config: ExtractionConfig = config.extraction_config

    if neigh_config:
        if neigh_config.load_from_cache and neigh_config.dump_cache:
            raise ValueError(
                "Cannot dump and load from cache at the same time. Please set one of these to False"
            )

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
    precision_string = (
        "int8" if env_config.int8 else ("fp16" if env_config.half else "fp32")
    )
    sampling_string = (
        "top_k" if config.do_top_k else ("top_p" if config.do_top_p else "temp")
    )
    output_subfolder = f"{config.output_name}/" if config.output_name else ""
    if openai_config is None:
        base_model_name = config.base_model.replace("/", "_")
    else:
        base_model_name = "openai-" + openai_config.model.replace("/", "_")
    scoring_model_string = (
        f"-{config.scoring_model_name}" if config.scoring_model_name else ""
    ).replace("/", "_")

    if config.tok_by_tok:
        tok_by_tok_string = "--tok_true"
    else:
        tok_by_tok_string = "--tok_false"

    if neigh_config:
        if neigh_config.span_length == 2:
            span_length_string = ""
        else:
            span_length_string = f"--{neigh_config.span_length}"

    dataset_member_name = config.dataset_member.replace("/", "_")
    dataset_nonmember_name = config.dataset_nonmember.replace("/", "_")
    if extraction_config is not None:
        sf_ext = "extraction_"
    else:
        sf_ext = "mia_"

    default_prompt_len = (
        extraction_config.prompt_len if extraction_config else 30
    )  # hack: will fix later
    # suffix = "QUANTILE_TEST"
    # TODO - Either automate suffix construction, or use better names (e.g. save folder with results, and attack config in it)
    suffix = f"{sf_ext}{output_subfolder}{base_model_name}-{scoring_model_string}-{neigh_config.model}-{sampling_string}/{precision_string}-{neigh_config.pct_words_masked}-{neigh_config.n_perturbation_rounds}-{dataset_member_name}-{dataset_nonmember_name}-{config.n_samples}{span_length_string}{config.max_words}{config.min_words}_plen{default_prompt_len}_{tok_by_tok_string}"

    # Add pile source to suffix, if provided
    # TODO: Shift dataset-specific processing to their corresponding classes
    if config.specific_source is not None:
        processed_source = data_utils.sourcename_process(config.specific_source)
        suffix += f"-{processed_source}"
    SAVE_FOLDER = os.path.join(env_config.tmp_results, suffix)

    new_folder = os.path.join(env_config.results, suffix)
    ##don't run if exists!!!
    print(f"{new_folder}")
    if os.path.isdir((new_folder)):
        print(f"HERE folder exists, not running this exp {new_folder}")
        exit(0)

    if not (os.path.exists(SAVE_FOLDER) or config.dump_cache):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    # write args to file
    # if not config.dump_cache:
    #     config.save(os.path.join(SAVE_FOLDER, 'args.json'), indent=4)

    if neigh_config:
        n_perturbation_list = neigh_config.n_perturbation_list
        n_perturbation_rounds = neigh_config.n_perturbation_rounds
        in_place_swap = neigh_config.original_tokenization_swap
        # n_similarity_samples = args.n_similarity_samples # NOT USED

    cache_dir = env_config.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # generic generative model
    base_model = LanguageModel(config)

    # reference model if we are doing the ref-based attack
    ref_models = None
    if ref_config is not None:
        ref_models = {
            model: ReferenceModel(config, model) for model in ref_config.models
        }
        # print('MOVING ref MODEL TO GPU...', end='', flush=True)

    # Load neighborhood attack model, only if we are doing the neighborhood attack
    mask_model = None
    if neigh_config and BlackBoxAttacks.NEIGHBOR in config.blackbox_attacks:
        model_kwargs = dict()
        if not config.baselines_only and not neigh_config.random_fills:
            if env_config.int8:
                model_kwargs = dict(
                    load_in_8bit=True, device_map="auto", torch_dtype=torch.bfloat16
                )
            elif env_config.half:
                model_kwargs = dict(torch_dtype=torch.bfloat16)
            try:
                n_positions = 512  # Should fix later, but for T-5 this is 512 indeed
                # mask_model.config.n_positions
            except AttributeError:
                n_positions = config.max_tokens
        else:
            n_positions = config.max_tokens
        tokenizer_kwargs = {
            "model_max_length": n_positions,
        }
        print(f"Loading mask filling model {config.neighborhood_config.model}...")
        if "t5" in config.neighborhood_config.model:
            mask_model = T5Model(
                config, model_kwargs=model_kwargs, tokenizer_kwargs=tokenizer_kwargs
            )
        elif "bert" in config.neighborhood_config.model:
            mask_model = BertModel(config)
        else:
            raise ValueError(f"Unknown model {config.neighborhood_config.model}")
        # if config.dataset_member in ['english', 'german']:
        #     preproc_tokenizer = mask_tokenizer

    print("MOVING BASE MODEL TO GPU...", end="", flush=True)
    base_model.load()

    if extraction_config is not None:
        print(f"Loading dataset {config.dataset_member}...")
        data_obj_nonmem = None
        data_obj_mem, data = generate_data(
            config.dataset_member, presampled=config.presampled_dataset_member
        )

        data, seq_lens, n_samples = generate_data_processed(
            data[: config.n_samples], batch_size=config.batch_size
        )

    else:
        print(
            f"Loading dataset {config.dataset_member} and {config.dataset_nonmember}..."
        )
        # data, seq_lens, n_samples = generate_data(config.dataset_member)

        data_obj_nonmem, data_nonmember = generate_data(
            config.dataset_nonmember,
            train=False,
            presampled=config.presampled_dataset_nonmember,
        )
        data_obj_mem, data_member = generate_data(
            config.dataset_member, presampled=config.presampled_dataset_member
        )
        if config.dump_cache and not config.load_from_cache:
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
                data_member,
                batch_size=config.batch_size,
                raw_data_non_member=data_nonmember,
            )

        # If neighborhood attack is used, see if we have cache available (and load from it, if we do)
        neighbors_nonmember, neighbors_member = None, None
        if (
            BlackBoxAttacks.NEIGHBOR in config.blackbox_attacks
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

    if (
        neigh_config
        and neigh_config.load_from_cache is False
        and neigh_config.random_fills
        and "t5" in neigh_config
        and neigh_config.model
    ):
        if not config.pretokenized:
            # TODO: maybe can be done if detokenized, but currently not supported
            mask_model.create_fill_dictionary(data)

    if config.scoring_model_name:
        print(f"Loading SCORING model {config.scoring_model_name}...")
        del base_model
        # Clear CUDA cache
        torch.cuda.empty_cache()

        base_model = LanguageModel(config, name=config.scoring_model_name)
        print("MOVING BASE MODEL TO GPU...", end="", flush=True)
        base_model.load()

    # TODO: Remove extraction-based logic, or make it so where it is not interleaved with MI logic flow (makes it hard to keep track)
    if extraction_config is not None and not config.pretokenized:
        f1_scores = []
        precisions = []
        recalls = []
        for original, sampled in zip(data["member"], data["nonmember"]):
            original_tokens = original.split(" ")
            sampled_tokens = sampled.split(" ")
            f1, precision, recall = f1_score(original_tokens, sampled_tokens)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
        # Summary statistics
        summary_stats = {
            "f1_mean": sum(f1_scores) / len(f1_scores),
            "precision_mean": sum(precisions) / len(precisions),
            "recall_mean": sum(recalls) / len(recalls),
            "f1_min": min(f1_scores),
            "precision_min": min(precisions),
            "recall_min": min(recalls),
            "f1_max": max(f1_scores),
            "precision_max": max(precisions),
            "recall_max": max(recalls),
        }
        # Save to JSON file
        with open(os.path.join(SAVE_FOLDER, "extraction_stats.json"), "w") as f:
            json.dump(summary_stats, f)
        # Plot and save f1_scores
        plot_utils.save_f1_histogram(f1_scores, save_folder=SAVE_FOLDER)

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

    tk_freq_map = None
    if config.token_frequency_map is not None:
        print("loading tk freq map")
        tk_freq_map = pickle.load(open(config.token_frequency_map, "rb"))

    # Add neighborhood-related data entries to 'data'
    data["nonmember_neighbors"] = neighbors_nonmember
    data["member_neighbors"] = neighbors_member

    ds_objects = {"nonmember": data_obj_nonmem, "member": data_obj_mem}

    outputs = []
    if config.blackbox_attacks is not None:
        # perform blackbox attacks
        blackbox_outputs = run_blackbox_attacks(
            data,
            ds_objects=ds_objects,
            target_model=base_model,
            ref_models=ref_models,
            config=config,
            n_samples=n_samples,
        )

        # TODO: Skipping openai-detector (for now)
        # if config.max_tokens < 512:
        #     baseline_outputs.append(eval_supervised(data, model='roberta-base-openai-detector'))
        #     baseline_outputs.append(eval_supervised(data, model='roberta-large-openai-detector'))

        for attack, output in blackbox_outputs.items():
            outputs.append(output)
            with open(os.path.join(SAVE_FOLDER, f"{attack}_results.json"), "w") as f:
                json.dump(output, f)

    if not config.baselines_only:
        pass
        # # TODO: incorporate all attacks below into blackbox attack flow (or separate white box attack flow if necessary)
        # # run perturbation experiments
        """
        if (
            neigh_config and not config.pretokenized
        ):  # TODO: not supported for pretokenized text
            for n_perturbations in n_perturbation_list:
                perturbation_results = get_perturbation_results(
                    neigh_config.span_length, n_perturbations
                )
                for perturbation_mode in ["d", "z"]:
                    output = run_perturbation_experiment(
                        perturbation_results,
                        perturbation_mode,
                        n_samples=n_samples,
                        span_length=neigh_config.span_length,
                        n_perturbations=n_perturbations,
                    )
                    outputs.append(output)
                    with open(
                        os.path.join(
                            SAVE_FOLDER,
                            f"perturbation_{n_perturbations}_{perturbation_mode}_results.json",
                        ),
                        "w",
                    ) as f:
                        json.dump(output, f)
        """

        # # run tokenization attack, if requested
        # if config.tokenization_attack:
        #     from mimir.attacks.tokenization import token_attack
        #     def token_attack_wrapper(text): return token_attack(base_model, text)
        #     output = run_baseline_threshold_experiment(data, token_attack_wrapper, "tokenization_attack", n_samples=n_samples)
        #     outputs.append(output)
        #     with open(os.path.join(SAVE_FOLDER, f"tokenization_attack_results.json"), "w") as f:
        #         json.dump(output, f)

        # # run quantile attack, if requested
        # if config.quantile_attack:
        #     from datasets import load_dataset
        #     from mimir.attacks.quantile import QuantileAttack
        #     alpha = 0.1 # FPR
        #     quantile_attacker = QuantileAttack(config, alpha)

        #     nonmember_data_other = load_dataset("imdb", split="test")["text"] # use split "unsupervised" for 2x data

        #     quantile_attacker.prepare(nonmember_data_other, base_model)
        #     def quantile_attack_wrapper(text): return quantile_attacker.attack(base_model, text)
        #     output = run_baseline_threshold_experiment(data, quantile_attack_wrapper, "quantile_attack", n_samples=n_samples)
        #     outputs.append(output)
        #     with open(os.path.join(SAVE_FOLDER, f"quantile_attack_results.json"), "w") as f:
        #         json.dump(output, f)

        # Skipping openai-detector (for now)
        # write supervised results to a file
        # TODO: update to read from baseline result dict
        # if config.max_tokens < 512:
        #     with open(os.path.join(SAVE_FOLDER, f"roberta-base-openai-detector_results.json"), "w") as f:
        #         json.dump(baseline_outputs[-2], f)

        #     # write supervised results to a file
        #     with open(os.path.join(SAVE_FOLDER, f"roberta-large-openai-detector_results.json"), "w") as f:
        #         json.dump(baseline_outputs[-1], f)

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
    new_folder = os.path.join(env_config.results, suffix)
    if not os.path.exists(os.path.dirname(new_folder)):
        os.makedirs(os.path.dirname(new_folder))
    os.rename(SAVE_FOLDER, new_folder)

    api_calls = 0
    if openai_config:
        api_calls = openai_config.api_calls
    print(f"Used an *estimated* {api_calls} API tokens (may be inaccurate)")
