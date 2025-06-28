from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
import json
import numpy as np
import re
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import torch
import click


# def load_gsm8k_dataset(dataset_name="madrylab/gsm8k-platinum", subset='test'):
#     def parse_dataset_answer(example):
#         match = re.search(r"\n####\s*(.+)", example['answer']).group(1).strip().replace(',','')
#         return {'parsed' : int(match)}
#     ds = load_dataset(dataset_name, "main", split=subset)
#     ds = ds.map(parse_dataset_answer)
#     return ds

# ---------- Utility Functions ----------
def load_gsm8k_subset(difficulty_level, dir_path='outputs/gsm8k_platinum/accuracy_subset', subset='test'):
    ds = load_dataset("json", data_files=f'{dir_path}/{difficulty_level}_{subset}.jsonl', split="train")
    return ds

def load_model(model_name, adapter=None):
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model_, tokenizer_ = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    if adapter is not None:
        model_.load_adapter(adapter)
    _ = FastLanguageModel.for_inference(model_) # Enable native 2x faster inference
    return model_, tokenizer_

def get_answer(tokenizer, model, prompt, num_times_to_repeat: int = 8):
    generation_kwargs = {
        "max_new_tokens": 250,
        "use_cache": True,
        "temperature": 0.9,
        "top_k": None,
        "do_sample": True,
    }

    formatted_prompts = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False, add_generation_prompt=True)] * num_times_to_repeat
    
    inputs = tokenizer(formatted_prompts, return_tensors = "pt", padding=True).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    outputs = outputs[:, inputs.input_ids.shape[1]:]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs

def get_answer_batch(tokenizer, model, prompts, num_times_to_repeat: int = 8):
    """
    Process multiple prompts in a single batch for better GPU utilization
    """
    generation_kwargs = {
        "max_new_tokens": 250,
        "use_cache": True,
        "temperature": 0.9,
        "top_k": None,
        "do_sample": True,
    }

    # Create all formatted prompts at once
    all_formatted_prompts = []
    for prompt in prompts:
        formatted_prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True)
        all_formatted_prompts.extend([formatted_prompt] * num_times_to_repeat)
    
    # Tokenize in larger batches
    inputs = tokenizer(all_formatted_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model.generate(**inputs, **generation_kwargs)
    
    outputs = outputs[:, inputs.input_ids.shape[1]:]
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Reshape outputs back to per-prompt format
    result = []
    for i in range(len(prompts)):
        start_idx = i * num_times_to_repeat
        end_idx = start_idx + num_times_to_repeat
        result.append(decoded_outputs[start_idx:end_idx])
    
    return result

ANSWER_PATTERN = re.compile(r"Answer:\s*(-?\d+)")
def parse_llm_answer(text):
    """
    Extracts the final answer from the LLM output.
    Expects the format: "Answer: XXX" where XXX is an integer.
    
    Args:
        text (str): The output from the LLM.
    
    Returns:
        float or None: The extracted float answer, or None if not found.
    """
    match = ANSWER_PATTERN.search(text)
    try:
        if match:
            return int(match.group(1))
    except:
        return None
    return None

def parse_outputs_parallel(outputs):
    """
    Parse outputs in parallel using ThreadPoolExecutor
    """
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(parse_llm_answer, outputs))

def write_to_file(raw_output, accuracies, path_to_write):
    data = {
        'outputs' : raw_output,
        'accuracies' : accuracies
    }
    with open(path_to_write, 'w') as f:
        json.dump(data, f)

@click.command()
@click.option('--adapter-name', '-a', default=None, help='Model adapter checkpoint name (e.g., "base" or checkpoint number).')
@click.option('--num-times-to-gen', '-n', default=8, type=int, help='Number of times to generate outputs per input.')
@click.option('--write_path', '-w', default=None, help='Where should write to file')
@click.option('--batch_size', '-b', default=4, type=int)
@click.option('--difficulty_level', '-d', type=int, default=25, help='difficulty for gsm8k')
@click.option('--model_name', '-m', type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
def run(adapter_name, num_times_to_gen, write_path, batch_size, difficulty_level, model_name):
    ds = load_gsm8k_subset(difficulty_level)
    if write_path is not None:
        print('Will write')
    #ds = load_gsm8k_dataset(dataset_name='openai/gsm8k', subset='train')
    model, tokenizer = load_model(model_name=model_name, adapter=adapter_name)
    
    total_samples = len(ds)
    raw_output = [None] * total_samples
    accuracies = [None] * total_samples
    pass_at = [None] * total_samples
    
    for i in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, total_samples)
        batch = ds[i:batch_end]
        
        if batch_size == 1 or batch_end - i == 1:
            qs = [batch['question']]
            answers = [batch['parsed']]
        else:
            qs = batch['question']
            answers = batch['parsed']
        
        prompts = [f"""Solve the following math word problem.

        {q}

        Think step-by-step. Then, provide the final answer as a single integer in the format "Answer: XXX" with no extra formatting.""" for q in qs]
        
        batch_outputs = get_answer_batch(tokenizer, model, prompts, num_times_to_repeat=num_times_to_gen)
        
        for j, (outputs, ans) in enumerate(zip(batch_outputs, answers)):
            pred = parse_outputs_parallel(outputs)
            idx = i + j
            raw_output[idx] = outputs
            pred_array = np.array(pred)
            accuracies[idx] = np.mean(pred_array == ans)
            
            if np.any(pred_array == ans):
                pass_at[idx] = 1
            else:
                pass_at[idx] = 0
        
        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
            if write_path is not None:
                write_to_file(raw_output, accuracies, write_path)
    if write_path is not None:
        write_to_file(raw_output, accuracies, write_path)

    print(f'Accuracy: {np.mean(accuracies):0.2f} +/-  {np.std(accuracies):0.2f}')
    print(f'Pass@{num_times_to_gen}: {np.mean(pass_at):0.2f} +/-  {np.std(pass_at):0.2f}')

if __name__ == '__main__':
    run()
    