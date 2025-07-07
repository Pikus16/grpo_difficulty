import re
from datasets import load_dataset
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os
import numpy as np
from glob import glob

def load_gsm8k_dataset(dataset_name='openai/gsm8k', subset='train'):
    def parse_dataset_answer(example):
        match = re.search(r"\n####\s*(.+)", example['answer']).group(1).strip().replace(',','')
        return {'parsed' : int(match)}
    ds = load_dataset(dataset_name, "main", split=subset)
    ds = ds.map(parse_dataset_answer)
    return ds

def extract_boxed_content(text: str) -> str:
    """
    Extracts the last value found inside LaTeX-style \\boxed{...} blocks.

    Args:
        text (str): The full text from the LLM output.

    Returns:
        Optional[str]: The last boxed value, or None if none found.
    """
    matches = re.findall(r'\\boxed\{(.*?)\}', text)
    try:
        return int(matches[-1])
    except:
        return None
    
def format_single_question_qwen(question: str, tokenizer):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': f"{question}.\nPut your final answer within \\boxed{{}}."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

def format_question_qwen(questions: list[str], tokenizer, device='cuda'):
    prompts = [format_single_question_qwen(q, tokenizer) for q in questions]
    return tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

def load_difficulty_subset(difficulty_level, subset_folder, dataset_name, model_name='unsloth/Qwen3-4B-unsloth-bnb-4bit', subset='train'):
    subset_file = os.path.join(
        subset_folder, 
        f"{dataset_name.replace('/','-')}_{subset}_{model_name.replace('/','-')}",
        f'{difficulty_level}.json'
    )
    assert os.path.exists(subset_file)
    ds = load_dataset("json", data_files=subset_file, split="train")
    ds = ds.map(lambda x: {"answer": x["parsed"]})
    ds = ds.remove_columns("parsed")
    return ds


def build_model_and_tokenizer(model_name, adapter_name=None, device: str = 'cuda'):
    # 1) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = (
        AutoModelForCausalLM
        .from_pretrained(model_name, trust_remote_code=True)
        .to(device)
    )
    if adapter_name is not None:
        model.load_adapter(adapter_name)
    model.eval()

    # 2) (Optional) Compile for speed if you're on PyTorch 2.x
    # if torch.backends.cuda.is_built():
    #     try:
    #         model = torch.compile(model)
    #     except Exception:
    #         pass

    return model, tokenizer


def sample_pass_at_k(
    model,
    tokenizer,
    questions: list[str],
    k: int = 8,
    max_new_tokens: int = 250,
    temperature: float = 1.0,
    top_p: float = 1,
) -> list[str]:
    
    inputs = format_question_qwen(questions, tokenizer)
    batch_size = len(questions)

    # Generate k samples in parallel
    with torch.no_grad():
        out_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=k,
            use_cache=True
        )

    # 5) Decode all samples
    decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

    # Group decoded into [ [k responses for q1], [k responses for q2], ... ]
    return [decoded[i * k: (i + 1) * k] for i in range(batch_size)]


def write_to_file(destination, all_responses):
    with open(destination, 'w') as f:
        json.dump(all_responses, f)

def do_single_run(model_name,
                  adapter_name,
                  dataset_name,
                  subset,
                  ds,
                  batch_size,
                  num_repeat,
                  answers,
                  output_folder):
    model, tokenizer = build_model_and_tokenizer(model_name=model_name, adapter_name=adapter_name)

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output_folder = os.path.join(output_folder, f"{dataset_name.replace('/','-')}_{subset}")
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if adapter_name is None:
            output_file = os.path.join(output_folder, f"{model_name.replace('/','-')}.json")
        else:
            output_file = os.path.join(output_folder, f"{os.path.basename(adapter_name)}.json")

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                all_responses = json.load(f)
            print(f"Loaded {len(all_responses)} responses from {output_file}")
        else:
            all_responses = []
    else:
        all_responses = []

    for i in tqdm(range(0, len(ds), batch_size)):
        if i < len(all_responses):
            continue
        
        batch = ds[i: i + batch_size]
        questions = batch['question']
        print(type(model))
        breakpoint()
        responses = sample_pass_at_k(model, tokenizer, questions, k=num_repeat)
        all_responses.extend(responses)
        if output_folder is not None:
            if i % (10 * batch_size) == 0:
                #torch.cuda.empty_cache()
                write_to_file(output_file, all_responses)

    if output_folder is not None:
        write_to_file(output_file, all_responses)

    # get accuracies and pass@k
    assert len(answers) == len(all_responses)
    accs, pass_at_k = [], []
    for answer, responses in zip(answers, all_responses):
        preds = np.array([extract_boxed_content(r) for r in responses])
        accs.append(np.mean(answer == preds))
        pass_at_k.append(1 if answer in preds else 0)
    return np.mean(accs), np.mean(pass_at_k)

def run_on_all_checkpoints(
    dataset_name: str,
    subset: str,
    model_name: str,
    num_repeat: int,
    output_folder: str,
    batch_size: int,
    adapter_folder: str,
    subset_folder: str,
    difficulty_level:int
):
    if subset_folder is None or difficulty_level is None:
        print(f'Loading dataset {dataset_name}, subset {subset}')
        ds = load_gsm8k_dataset(dataset_name, subset=subset)
        answers = [x['parsed'] for x in ds]
    else:
        print(f'Loading difficulty subset {difficulty_level}, dataset {dataset_name}, subset {subset_folder} on {subset} set')
        ds = load_difficulty_subset(
            difficulty_level=difficulty_level,
            subset_folder=subset_folder,
            dataset_name=dataset_name,
            model_name=model_name,
            subset=subset
        )
        answers = [x['answer'] for x in ds]
        
    assert os.path.exists(adapter_folder)

    all_adapters = glob(f'{adapter_folder}/checkpoint-*')

    checkpoint_numbers = sorted([int(os.path.basename(path).split('-')[1]) for path in all_adapters])
    print(f'Running on checkpoints: {checkpoint_numbers}')

    accuracies, passes = [], []
    for ckpt_num in checkpoint_numbers:
        adapter_name = f'{adapter_folder}/checkpoint-{ckpt_num}'
        assert os.path.exists(adapter_name)
        acc, pass_at_k = do_single_run(
            model_name=model_name,
            adapter_name=adapter_name,
            dataset_name=dataset_name,
            subset=subset,
            ds=ds,
            batch_size=batch_size,
            num_repeat=num_repeat,
            answers=answers,
            output_folder=output_folder
        )
        print(f"Checkpoint: {ckpt_num}: Accuracy: {acc:0.3f}, Pass@{num_repeat}: {pass_at_k:0.3f}")
        accuracies.append(acc)
        passes.append(pass_at_k)

    print(f'Running pretrained')
    pretrained_accuracy, pretrained_passes = do_single_run(
        model_name=model_name,
        adapter_name=None, # setting to None so pretrained
        dataset_name=dataset_name,
        subset=subset,
        ds=ds,
        batch_size=batch_size,
        num_repeat=num_repeat,
        answers=answers,
        output_folder=output_folder
    )
    print(f"Base: Accuracy: {pretrained_accuracy:0.3f}, Pass@{num_repeat}: {pretrained_passes:0.3f}")

    results = {
        'checkpoint': checkpoint_numbers,
        'accuracy': accuracies,
        f'pass@{num_repeat}': passes,
        'base accuracy' : pretrained_accuracy,
         f'base pass@{num_repeat}': pretrained_passes,
    }
    return results