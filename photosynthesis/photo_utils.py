from datasets import Dataset
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os
import numpy as np
from glob import glob
import textstat

PROMPT = "Describe photosynthesis. Use as simple terms as possible"

def format_single_question_qwen(tokenizer, text = PROMPT):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)

def create_dataset_from_str(tokenizer, text = PROMPT) -> Dataset:
    text = format_single_question_qwen(tokenizer=tokenizer, text=text)
    data = {'prompt' : [text], 'answer' : ['']}
    return Dataset.from_dict(data)

def format_question_qwen(tokenizer, device='cuda'):
    prompt = format_single_question_qwen(tokenizer)
    return tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

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
    if torch.backends.cuda.is_built():
        try:
            model = torch.compile(model)
        except Exception:
            pass

    return model, tokenizer


def sample_pass_at_k(
    model,
    tokenizer,
    inputs,
    max_new_tokens: int = 250,
    temperature: float = 1.0,
    top_p: float = 1,
    num_to_repeat: int = 16
) -> list[str]:
    

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
            use_cache=True,
            num_return_sequences=num_to_repeat,
        )

    # 5) Decode all samples
    out_ids = out_ids[:,inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    return decoded


def write_to_file(destination, all_responses):
    with open(destination, 'w') as f:
        json.dump(all_responses, f)

def do_single_run(
    model_name,
    adapter_name,
    batch_size,
    num_to_gen,
    desired_threshold = None
):
    model, tokenizer = build_model_and_tokenizer(model_name=model_name, adapter_name=adapter_name)
    input_prompts = format_question_qwen(tokenizer=tokenizer)
    

    if adapter_name is not None:
        output_file = f'{adapter_name}/responses.json'
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                all_responses = json.load(f)
            print(f"Loaded {len(all_responses)} responses from {output_file}")
        else:
            all_responses = []
    else:
        output_file = None
        all_responses = []

    for i in tqdm(range(0, num_to_gen, batch_size)):
        if i < len(all_responses):
            continue
        
        responses = sample_pass_at_k(
            model,
            tokenizer,
            input_prompts,
            num_to_repeat = batch_size
        )
        all_responses.extend(responses)
        if output_file is not None:
            if i % (10 * batch_size) == 0:
                #torch.cuda.empty_cache()
                write_to_file(output_file, all_responses)

    if output_file is not None:
        write_to_file(output_file, all_responses)

    # get performance
    scores = np.array([textstat.flesch_kincaid_grade(r) for r in all_responses])
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    if desired_threshold is not None:
        below_thresh = np.mean(scores <= desired_threshold)
    else:
        below_thresh = None
    return avg_score, std_score, below_thresh, scores

def run_on_all_checkpoints(
    model_name: str,
    num_to_gen: int,
    batch_size: int,
    adapter_folder: str,
    desired_threshold: float
):
    avg_scores, std_scores, below_threshs = [], [], []
    if adapter_folder is not None:
        assert os.path.exists(adapter_folder)
        all_adapters = glob(f'{adapter_folder}/checkpoint-*')
        checkpoint_numbers = sorted([int(os.path.basename(path).split('-')[1]) for path in all_adapters])
        print(f'Running on checkpoints: {checkpoint_numbers}')

        
        
        for ckpt_num in checkpoint_numbers:
            adapter_name = f'{adapter_folder}/checkpoint-{ckpt_num}'
            assert os.path.exists(adapter_name)
            avg_score, std_score, below_thresh, _ = do_single_run(
                model_name,
                adapter_name,
                batch_size,
                num_to_gen,
                desired_threshold
            )
            print(f"Checkpoint: {ckpt_num}: Score: {avg_score:0.3f} +/- {std_score:0.3f}, % below thresh: {below_thresh:0.3f}")
            avg_scores.append(avg_score)
            std_scores.append(std_score)
            below_threshs.append(below_thresh)


    print(f'Running pretrained')
    pretrained_avg_score, pretrained_std_score, pretrained_below_thresh = do_single_run(
        model_name,
        None,
        batch_size,
        num_to_gen,
        desired_threshold
    )
    print(f"Base: Score: {pretrained_avg_score:0.3f} +/- {pretrained_std_score:0.3f}, % below thresh: {pretrained_below_thresh:0.3f}")
    
    results = {
        'checkpoint': checkpoint_numbers,
        'average_scores': avg_scores,
        'std_scores': std_scores,
        'below_threshs' : below_threshs,
        'pretrained_score' : pretrained_avg_score,
        'pretrained_std' : pretrained_std_score,
        'pretrained_below_thresh' : pretrained_below_thresh
    }
    return results
    
