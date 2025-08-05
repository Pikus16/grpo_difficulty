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

def _get_base_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _get_dataset_dir():
    return os.path.join(_get_base_path(), 'dsets')

def _get_responses_dir():
    return os.path.join(_get_base_path(), 'responses')

def _get_checkpoint_dir(dataset_name: str, name: str):
    checkpoint_base_dir = os.path.join(_get_base_path(), 'checkpoints')
    return os.path.join(
        checkpoint_base_dir,
        dataset_name,
        name
    )

def load_whole_dataset(dataset_name: str, split: str, model_name: str = None) -> HFDataset:
    dset_base_path =_get_dataset_dir()
    data_file = os.path.join(dset_base_path, dataset_name, f'{split}.json')
    assert os.path.exists(data_file)
    ds = load_dataset(
        "json", 
        data_files=data_file, 
        split="train"
    )
    if model_name is not None:
        # load base scores
        score_file =  os.path.join(dset_base_path, dataset_name, model_name.replace('/','-'), f'{split}_scores.json')
        if os.path.exists(score_file):
            with open(score_file) as f:
                scores = json.load(f)
            assert len(scores) == len(ds)
            ds = ds.add_column('pretrained_score', scores)
    return ds

def get_hardest_subset(whole_dataset: HFDataset, size: int) -> HFDataset:
    # Check that the 'pretrained_score' column exists
    assert 'pretrained_score' in whole_dataset.column_names, \
        "'pretrained_score' column is missing from the dataset"
    
    # Sort by pretrained_score ascending (lowest scores first)
    sorted_dataset = whole_dataset.sort("pretrained_score")

    # Take the first `size` examples
    return sorted_dataset.select(range(size))

def get_easiest_subset(whole_dataset: HFDataset, size: int) -> HFDataset:
    assert 'pretrained_score' in whole_dataset.column_names, \
        "'pretrained_score' column is missing from the dataset"
    sorted_ds = whole_dataset.sort("pretrained_score", reverse=True)
    return sorted_ds.select(range(size))

def get_random_subset(whole_dataset: HFDataset, size: int, seed: int = 42) -> HFDataset:
    return whole_dataset.shuffle(seed=seed).select(range(size))

def get_middle_subset(whole_dataset: HFDataset, size: int) -> HFDataset:
    assert 'pretrained_score' in whole_dataset.column_names, \
        "'pretrained_score' column is missing from the dataset"
    
    sorted_ds = whole_dataset.sort("pretrained_score")
    total = len(sorted_ds)
    
    # Centered around the middle
    start = max((total - size) // 2, 0)
    return sorted_ds.select(range(start, start + size))

def get_dataset_subset(whole_dataset:HFDataset, strategy: str, size: float | int) -> HFDataset:
    if isinstance(size, float):
        assert size <= 1.0 and size >= 0.0
        size = int(len(whole_dataset) * size)
    assert isinstance(size, int)
    if strategy == 'hardest':
        fn = get_hardest_subset
    elif strategy == 'easiest':
        fn = get_easiest_subset
    elif strategy == 'middle':
        fn = get_middle_subset
    elif strategy == 'random':
        fn = get_random_subset
    else:
        raise ValueError(f'Unknown strategy: {strategy}')
    
    return fn(whole_dataset, size)

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
    
def reformat_question(prompt: str, dataset_name: str):
    if dataset_name == 'kegg':
        prompt = f"Solve the following genomic pathway question:\n{prompt}\n"
        prompt += f'Answer concisely, in 150 words or less. Put your final answer within \\boxed{{(disease name)}}, with no extra formatting'
    elif dataset_name == 'shuffleobj':
        prompt = f"{prompt}.\nAnswer concisely, in 150 words or less. Put your final answer within \\boxed{{(X)}} (ex: \\boxed{{(A)}})"
    elif dataset_name == 'gsm8k':
        prompt = f"{prompt}.\nPut your final answer within \\boxed{{}}."
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return prompt
    
def format_single_question(question: str, tokenizer: AutoTokenizer, dataset_name: str):
    prompt = reformat_question(question, dataset_name)
    return tokenizer.apply_chat_template(
        [{'role': 'user', 
          'content': prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

def build_test_model_and_tokenizer(model_name, adapter_name=None, device: str = 'cuda'):
    # 1) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use half precision=
        use_cache=True
    ).to(device)
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
    device: str = 'cuda'
) -> list[str]:
    
    inputs = tokenizer(questions, return_tensors="pt", padding=True).to(device)
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

    # Remove inputs
    out_ids = out_ids[:, inputs["input_ids"].shape[1]:]

    # Decode all samples
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

def calc_accuracy(whole_dataset: HFDataset,
                  all_responses: list[list[str]],
                  dataset_name: str):
    # Get accuracies and pass@k
    if dataset_name == 'gsm8k':
        answers = [int(x['answer']) for x in whole_dataset]
        def process_fn(x):
            try:
                return int(x)
            except:
                return None
            
    elif dataset_name in ['kegg','shuffleobj']:
        answers = [x['answer'].lower() for x in whole_dataset]
        process_fn = lambda x: x
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    assert len(answers) == len(all_responses)
    accs, pass_at_k = [], []
    for answer, responses in zip(answers, all_responses):
        preds = np.array([
            process_fn(extract_boxed_content(r)) for r in responses
        ])
        accs.append(np.mean(answer == preds))
        pass_at_k.append(1 if answer in preds else 0)
    return accs, pass_at_k

def do_single_run(
    model_name,
    adapter_name,
    split,
    dataset_name,
    batch_size,
    num_repeat
):
    # Load Model
    model, tokenizer = build_test_model_and_tokenizer(model_name=model_name, adapter_name=adapter_name)

    # Load Dataset
    whole_dataset = load_whole_dataset(
        dataset_name = dataset_name,
        split = split,
        model_name=model_name
    )
    whole_dataset = format_dataset_(whole_dataset, tokenizer, dataset_name)

    # Load current responses
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
            model_name.replace('/','-')
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'{split}_responses.json')
    all_responses = load_output_file(output_file)

    for i in tqdm(range(0, len(whole_dataset), batch_size)):
        if i < len(all_responses):
            continue
        
        batch = whole_dataset[i: i + batch_size]
        questions = batch['prompt']
        responses = sample_pass_at_k(model, tokenizer, questions, k=num_repeat)
        all_responses.extend(responses)
        if output_file is not None:
            if i % (10 * batch_size) == 0:
                #torch.cuda.empty_cache()
                write_to_file(output_file, all_responses)

    if output_file is not None:
        write_to_file(output_file, all_responses)

    del model
    torch.cuda.empty_cache()

    # Get accuracies and pass@k
    accs, pass_at_k = calc_accuracy(whole_dataset=whole_dataset,
                  all_responses=all_responses,
                  dataset_name=dataset_name)

    if adapter_name is None:
        # Write scores to file (since pretrained)
        scores_file = os.path.join(output_dir, f'{split}_scores.json')
        if not os.path.exists(scores_file):
            print(f'Writing scores to {scores_file}')
            write_to_file(scores_file, accs)
    return np.mean(accs), np.mean(pass_at_k)

def format_dataset_(dataset: HFDataset, tokenizer: AutoTokenizer, dataset_name: str):
    def _format_prompt(example):
        new_prompt = format_single_question(
            example['question'],
            tokenizer,
            dataset_name
        )
        return {'prompt' : new_prompt}
    dataset = dataset.map(_format_prompt)
    dataset = dataset.remove_columns(['question'])
    return dataset

def run_on_all_checkpoints(
    model_name: str,
    num_repeat: int,
    batch_size: int,
    split: str,
    dataset_name: str,
    run_name: str
):
    results = {}
    if dataset_name is not None and run_name is not None:
        adapter_folder = _get_checkpoint_dir(dataset_name, run_name)
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
                split=split,
                dataset_name=dataset_name,
                batch_size=batch_size,
                num_repeat=num_repeat
            )
            
            print(f"Checkpoint: {ckpt_num}: Accuracy: {acc:0.3f}, Pass@{num_repeat}: {pass_at_k:0.3f}")
            accuracies.append(acc)
            passes.append(pass_at_k)
            torch.cuda.empty_cache()

        results['checkpoint'] = checkpoint_numbers
        results['accuracy'] = accuracies
        results[f'pass@{num_repeat}'] = passes

    print(f'Running pretrained')
    pretrained_accuracy, pretrained_passes = do_single_run(
        model_name=model_name,
        adapter_name=None,
        split=split,
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_repeat=num_repeat
    )
    print(f"Base: Accuracy: {pretrained_accuracy:0.3f}, Pass@{num_repeat}: {pretrained_passes:0.3f}")

    results[ 'base accuracy'] = pretrained_accuracy
    results[f'base pass@{num_repeat}'] =  pretrained_passes
    return results
    
class CumulativeSuccessCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self._cumulative = 0

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # `state.log_history[-1]` is the most recent logged metrics dict
        latest_state = state.log_history[-1]
        if 'reward' in latest_state:
            num_successes = int(latest_state['reward'] * args.num_generations)
            self._cumulative += num_successes
            # push to W&B
            wandb.log({"train/cumulative_successes": self._cumulative})
        return control