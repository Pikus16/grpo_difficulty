import re
from datasets import load_dataset, Dataset as HFDataset
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, TrainerState, TrainerControl
from tqdm import tqdm
import json
import os
import numpy as np
from glob import glob
import wandb
from src_utils import _get_responses_dir, _get_dataset_dir, _get_checkpoint_dir, build_test_model_and_tokenizer, load_whole_dataset, format_dataset_, sample_pass_at_k, calc_accuracy, write_to_file, load_output_file, extract_boxed_content
import argparse


# def do_single_run(
#     model_name,
#     adapter_name,
#     split,
#     dataset_name,
#     batch_size,
#     num_repeat
# ):
#     # Load Model
#     model, tokenizer = build_test_model_and_tokenizer(model_name=model_name, adapter_name=adapter_name)

#     # Load Dataset
#     whole_dataset = load_whole_dataset(
#         dataset_name = dataset_name,
#         split = split,
#         model_name=model_name
#     )
#     whole_dataset = format_dataset_(whole_dataset, tokenizer, dataset_name)

#     # Load current responses
#     if adapter_name is not None:
#         output_dir = os.path.join(
#              _get_responses_dir(),
#              dataset_name,
#              '/'.join(adapter_name.split('/')[-2:])
#         )
#     else:
#         output_dir = os.path.join(
#             _get_dataset_dir(),
#             dataset_name,
#             model_name.replace('/','-')
#         )

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     output_file = os.path.join(output_dir, f'{split}_responses.json')
#     all_responses = load_output_file(output_file)

#     for i in tqdm(range(len(all_responses), len(whole_dataset), batch_size)):
#         batch = whole_dataset[i: i + batch_size]
#         questions = batch['prompt']
#         responses = sample_pass_at_k(model, tokenizer, questions, k=num_repeat)
#         all_responses.extend(responses)
#         if output_file is not None:
#             if i % (10 * batch_size) == 0:
#                 #torch.cuda.empty_cache()
#                 write_to_file(output_file, all_responses)

#     if output_file is not None:
#         write_to_file(output_file, all_responses)

#     del model
#     torch.cuda.empty_cache()

#     # Get accuracies and pass@k
#     accs, pass_at_k = calc_accuracy(whole_dataset=whole_dataset,
#                   all_responses=all_responses,
#                   dataset_name=dataset_name)

#     if adapter_name is None:
#         # Write scores to file (since pretrained)
#         scores_file = os.path.join(output_dir, f'{split}_scores.json')
#         if not os.path.exists(scores_file):
#             print(f'Writing scores to {scores_file}')
#             write_to_file(scores_file, accs)
#     return np.mean(accs), np.mean(pass_at_k)


# ------------------- AIME EVAL SUPPORT (HF-based, self-contained) -------------------

def _auto_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _load_aime_dataset_from_hf(config_name: str = "AIME2025-I", split: str | None = None) -> HFDataset:
    ds = load_dataset("opencompass/AIME2025", config_name)
    if split is not None and split in ds:
        return ds[split]
    for candidate in ["validation", "test", "train"]:
        if candidate in ds:
            return ds[candidate]
    # If load_dataset returned a single split directly
    if isinstance(ds, HFDataset):
        return ds
    raise ValueError("Could not find a usable split in the AIME dataset")


def _format_aime_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    prompt = (
        f"{question}.\n"
        f"Put your final answer within \\boxed{{}}."
    )
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _prepare_aime_dataset(tokenizer: AutoTokenizer, config_name: str = "AIME2025-I", split: str | None = None) -> tuple[HFDataset, list]:
    ds = _load_aime_dataset_from_hf(config_name=config_name, split=split)

    # Try to infer question and answer fields
    def extract_q(example):
        for key in ["question", "problem", "prompt", "input", "query"]:
            if key in example and isinstance(example[key], str):
                return example[key]
        raise KeyError("Could not find a question field in the AIME example")

    def extract_a(example):
        for key in ["answer", "label", "solution", "target"]:
            if key in example:
                return example[key]
        raise KeyError("Could not find an answer field in the AIME example")

    ds = ds.map(lambda ex: {"question": extract_q(ex), "answer": extract_a(ex)})
    ds = ds.map(lambda ex: {"prompt": _format_aime_prompt(ex["question"], tokenizer)})

    answers = [ex["answer"] for ex in ds]
    return ds, answers


def _to_int(value) -> int | None:
    try:
        # Common cases: int already or string of digits
        if isinstance(value, int):
            return value
        s = str(value).strip()
        # Remove common wrappers like boxed() if present
        if s.startswith("\\boxed{") and s.endswith("}"):
            s = s[len("\\boxed{"):-1]
        # Keep only leading sign and digits
        m = re.search(r"[-+]?\d+", s)
        return int(m.group(0)) if m else None
    except Exception:
        return None


def _calc_aime_accuracy(ds: HFDataset, all_responses: list[list[str]], answers: list):
    assert len(ds) == len(all_responses) == len(answers)
    accs, pass_at_k = [], []
    for truth, responses in zip(answers, all_responses):
        truth_int = _to_int(truth)
        preds = []
        for r in responses:
            boxed = extract_boxed_content(r)
            preds.append(_to_int(boxed if boxed is not None else r))
        preds = np.array(preds)
        accs.append(np.mean(preds == truth_int))
        pass_at_k.append(1 if truth_int in preds else 0)
    return accs, pass_at_k


def do_single_run_aime(
    model_name: str,
    adapter_name: str | None,
    batch_size: int,
    num_repeat: int,
    config_name: str = "AIME2025-I",
    split: str | None = None
):
    device = _auto_device()
    model, tokenizer = build_test_model_and_tokenizer(model_name=model_name, adapter_name=adapter_name, device=device)

    # Outputs folder mirrors existing convention
    dataset_name = "AIME2025"
    if adapter_name is not None:
        output_dir = os.path.join(
            _get_responses_dir(),
            dataset_name,
            '/'.join(adapter_name.split('/')[-2:])
        )
    else:
        output_dir = os.path.join(
            _get_dataset_dir(),
            dataset_name,
            model_name.replace('/', '-')
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    split_tag = split if split is not None else "eval"
    output_file = os.path.join(output_dir, f'{split_tag}_responses.json')

    ds, answers = _prepare_aime_dataset(tokenizer, config_name=config_name, split=split)

    all_responses = load_output_file(output_file)

    for i in tqdm(range(len(all_responses), len(ds), batch_size)):
        batch = ds[i: i + batch_size]
        questions = batch['prompt']
        responses = sample_pass_at_k(model, tokenizer, questions, k=num_repeat, device=device, max_new_tokens=512)
        all_responses.extend(responses)
        if output_file is not None and i % (10 * batch_size) == 0:
            write_to_file(output_file, all_responses)

    if output_file is not None:
        write_to_file(output_file, all_responses)

    # Free GPU if present
    del model
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    accs, pass_at_k = _calc_aime_accuracy(ds, all_responses, answers)
    return np.mean(accs), np.mean(pass_at_k)


def main():
    parser = argparse.ArgumentParser(description="Run AIME eval with a HF model")
    parser.add_argument("--model_name", type=str, required=True, help="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_name", type=str, default=None, help="Optional LoRA/adapter path")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_repeat", type=int, default=8, help="pass@k samples per question")
    parser.add_argument("--config_name", type=str, default="AIME2025-I", help="HF config for opencompass/AIME2025")
    parser.add_argument("--split", type=str, default=None, help="Optional dataset split if available (e.g., validation)")
    args = parser.parse_args()

    acc, passk = do_single_run_aime(
        model_name=args.model_name,
        adapter_name=args.adapter_name,
        batch_size=args.batch_size,
        num_repeat=args.num_repeat,
        config_name=args.config_name,
        split=args.split
    )
    print(f"AIME | Accuracy: {acc:0.3f}, Pass@{args.num_repeat}: {passk:0.3f}")


if __name__ == "__main__":
    main()