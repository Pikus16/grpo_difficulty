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

def format_single_question_qwen(example, tokenizer, include_sequence=False):
    prompt = f"Solve the following genomic pathway question:\n{example['question']}\n"
    if include_sequence:
        prompt += f'\nReference Sequence: {example["reference_sequence"]}\nVariant Sequence: {example["variant_sequence"]}\n'
    prompt += f'Answer concisely, in 150 words or less. Put your final answer within \\boxed{{(DISEASE NAME)}}, with no extra formatting'

    return tokenizer.apply_chat_template(
        [{'role': 'user', 
          'content': prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

def load_kegg_dataset(
        tokenizer,
        split = 'train',
        difficulty_level: int = None,
        model_name: str = None):
    def _format_prompt(example):
        new_prompt = format_single_question_qwen(example, tokenizer)
        return {'prompt' : new_prompt}
    if difficulty_level is None:
        ds = load_dataset('wanglab/kegg', 'default', split=split)
    else:
        # load just subset of interest
        dirname = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(dirname,
                                'subsets', 
                                f"{model_name.replace('/','-')}", 
                                split, 
                                f'{difficulty_level}.json')
        ds = load_dataset(
            "json", 
            data_files=data_file, 
            split="train"
        )
    ds = ds.map(_format_prompt)
    return ds

def extract_boxed_content(text: str) -> str:
    """
    Extracts the last value found inside LaTeX-style \\boxed{...} blocks.

    Args:
        text (str): The full text from the LLM output.

    Returns:
        Optional[int]: The last boxed value, or None if none found.
    """
    matches = re.findall(r'\\boxed\{(.*?)\}', text)
    try:
        return str(matches[-1]).strip().lower()
    except:
        return None
    
def format_question_qwen(prompts: list[str], tokenizer, device='cuda'):
    return tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

def build_model_and_tokenizer(model_name, adapter_name=None, device: str = 'cuda'):
    # 1) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use half precision
        device_map="auto",          # Automatic device placement
        use_cache=True
    )
    if adapter_name is not None:
        model.load_adapter(adapter_name)
    model.eval()

    # 2) (Optional) Compile for speed if you're on PyTorch 2.x
    if torch.backends.cuda.is_built():
        try:
            model = torch.compile(model)
        except Exception:
            pass

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

def load_output_file(path) -> list:
    if os.path.exists(path):
        with open(path, 'r') as f:
            all_responses = json.load(f)
        print(f"Loaded {len(all_responses)} responses from {path}")
    else:
        all_responses = []
    return all_responses

def do_single_run(
    model_name,
    adapter_name,
    split,
    batch_size,
    num_repeat,
):
    model, tokenizer = build_model_and_tokenizer(model_name=model_name, adapter_name=adapter_name)
    ds = load_kegg_dataset(split=split, tokenizer=tokenizer)
    answers = [x['answer'] for x in ds]

    if adapter_name is not None:
        output_file = f'{adapter_name}/{split}_responses.json'
    else:
        output_dir = f"pretrained_responses/{model_name.replace('/','-')}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f'{split}_responses.json')
    all_responses = load_output_file(output_file)

    for i in tqdm(range(len(all_responses), len(ds), batch_size)):
        batch = ds[i: i + batch_size]
        questions = batch['prompt']
        responses = sample_pass_at_k(model, tokenizer, questions, k=num_repeat)
        all_responses.extend(responses)
        if output_file is not None:
            if i % (10 * batch_size) == 0:
                #torch.cuda.empty_cache()
                write_to_file(output_file, all_responses)

    if output_file is not None:
        write_to_file(output_file, all_responses)

    # get accuracies and pass@k
    assert len(answers) == len(all_responses)
    accs, pass_at_k = [], []
    for answer, responses in zip(answers, all_responses):
        preds = np.array([extract_boxed_content(r) for r in responses])
        accs.append(np.mean(answer == preds))
        pass_at_k.append(1 if answer in preds else 0)
    return np.mean(accs), np.mean(pass_at_k), accs

def check_subset_acc(all_accs, model_name, split, difficulty_level) -> float:
    dirname = os.path.dirname(os.path.abspath(__file__))
    base_accs_path = os.path.join(dirname, 'dset', 'pretrained_scores', model_name.replace('/','-'), f'{split}.json')
    with open(base_accs_path) as f:
        base_accs = json.load(f)
    diff_float = int(difficulty_level) / 100
    inds = np.where(np.array(base_accs) <= diff_float)[0]
    return np.mean(np.array(all_accs)[inds])

def run_on_all_checkpoints(
    model_name: str,
    num_repeat: int,
    batch_size: int,
    adapter_folder: str,
    split: str,
    difficulty_level: int
):
    results = {}
    if adapter_folder is not None:
        assert os.path.exists(adapter_folder)
        all_adapters = glob(f'{adapter_folder}/checkpoint-*')
        checkpoint_numbers = sorted([int(os.path.basename(path).split('-')[1]) for path in all_adapters])
        print(f'Running on checkpoints: {checkpoint_numbers}')

        if difficulty_level is not None:
            subset_accuracy = []

        accuracies, passes = [], []
        for ckpt_num in checkpoint_numbers:
            adapter_name = f'{adapter_folder}/checkpoint-{ckpt_num}'
            assert os.path.exists(adapter_name)
            acc, pass_at_k, per_example_acc = do_single_run(
                model_name,
                adapter_name,
                split,
                batch_size,
                num_repeat
            )
            print(f"Checkpoint: {ckpt_num}: Accuracy: {acc:0.3f}, Pass@{num_repeat}: {pass_at_k:0.3f}")
            accuracies.append(acc)
            passes.append(pass_at_k)
            if difficulty_level is not None:
                test_subset_acc = check_subset_acc(per_example_acc, model_name, split, difficulty_level)
                print(f'Test Set {difficulty_level} Accuracy: {test_subset_acc:0.3f}')
                subset_accuracy.append(test_subset_acc)
        results['checkpoint'] = checkpoint_numbers
        results['accuracy'] = accuracies
        results[f'pass@{num_repeat}'] = passes
        if difficulty_level is not None:
            results[f'{difficulty_level} accuracy'] = subset_accuracy

    print(f'Running pretrained')
    pretrained_accuracy, pretrained_passes, pretrained_per_example_acc = do_single_run(
        model_name,
        None, # setting to None so pretrained
        split,
        batch_size,
        num_repeat,
    )
    print(f"Base: Accuracy: {pretrained_accuracy:0.3f}, Pass@{num_repeat}: {pretrained_passes:0.3f}")
    
    results[ 'base accuracy'] = pretrained_accuracy
    results[f'base pass@{num_repeat}'] =  pretrained_passes
    if difficulty_level is not None:
        test_subset_acc = check_subset_acc(pretrained_per_example_acc, model_name, split, difficulty_level)
        print(f'Test Set {difficulty_level} Accuracy: {test_subset_acc:0.3f}')
        results[f'base {difficulty_level} accuracy'] = test_subset_acc
    return results
    
