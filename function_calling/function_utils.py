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
from typing import List, Dict, Any, Union, Optional

TASK_INSTRUCTION = """
You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.
""".strip()

FORMAT_INSTRUCTION = """
Answer only with the json format below, and include no other text. Your output will directly be fed into a json parser, so the final answer MUST strictly adhere to the following JSON format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make tool_calls an empty list '[]'.
[
    {"name": "TOOL_NAME_1, "arguments" : {"PARAMETER_NAME_1" : "VALUE_1", .....}},
    ....(more tool calls as required)
]
""".strip()

def load_function_dataset(subset, dset_path = 'dset'):
    json_filepath = f'{dset_path}/{subset}.json'
    assert os.path.exists(json_filepath)
    ds = load_dataset(
        "json", 
        data_files=json_filepath, 
        split="train"
    )
    ds = ds.rename_columns({'answer_str' : 'answer'})
    # def parse_answer(example):
    #     answer = example['answer_str']#json.loads(example['answer_str'])
    #     return {'answer' : answer}
    # ds = ds.map(parse_answer)
    #ds = ds.remove_columns(['id', 'answer_str'])
    return ds

# def sort_tool(tool: list) -> list:
#     # Turn each dict into a JSON string with sorted keys
#     canon = [json.dumps(call, sort_keys=True) for call in tool]
#     # Sort the list of JSON‐strings so order in the original list doesn't matter
#     return sorted(canon)

def parse_response(pred: str) -> Optional[list]:
    try:
        pred = pred.split("</think>")[-1].strip()
        return json.loads(pred)
    except:
        # formatting issue
        return None

# Helper function to build the input prompt for our model
def build_prompt(tools: list, query: str):
    prompt = f"[BEGIN OF TASK INSTRUCTION]\n{TASK_INSTRUCTION}\n[END OF TASK INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{tools}\n[END OF AVAILABLE TOOLS]\n\n"
    prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{FORMAT_INSTRUCTION}\n[END OF FORMAT INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
    return prompt
    
def format_single_question_qwen(question: str, tokenizer, tools):
    prompt = build_prompt(tools=tools, query=question)
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

def format_question_qwen(questions: list[str], tools, tokenizer, device='cuda'):
    prompts = [format_single_question_qwen(q, tokenizer, tools) for q in questions]
    return tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

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
    questions: list[str],
    tools,
    k: int = 8,
    max_new_tokens: int = 250,
    temperature: float = 1.0,
    top_p: float = 1,
) -> list[str]:
    
    inputs = format_question_qwen(questions, tools, tokenizer)
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

def do_single_run(
    model_name,
    adapter_name,
    subset,
    ds,
    batch_size,
    num_repeat,
    answers
):
    model, tokenizer = build_model_and_tokenizer(
        model_name=model_name,
        adapter_name=adapter_name
    )

    if adapter_name is not None:
        output_file = f'{adapter_name}/{subset}_responses.json'
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                all_responses = json.load(f)
            print(f"Loaded {len(all_responses)} responses from {output_file}")
        else:
            all_responses = []
    else:
        output_file = None
        all_responses = []

    for i in tqdm(range(0, len(ds), batch_size)):
        if i < len(all_responses):
            continue
        
        batch = ds[i: i + batch_size]
        questions = batch['question']
        tools = batch['tools']
        responses = sample_pass_at_k(model, tokenizer, questions, tools=tools, k=num_repeat)
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
        predictions = [parse_response(r) for r in responses]
        perf = np.array([compare_tool_calls(p,answer) for p in predictions])
        accs.append(np.mean(perf))
        pass_at_k.append(1 if any(perf) else 0)

        
    return np.mean(accs), np.mean(pass_at_k)

def run_on_all_checkpoints(
    model_name: str,
    num_repeat: int,
    batch_size: int,
    adapter_folder: str,
    subset: str
):
    ds = load_function_dataset(subset)
    answers = [x['answer'] for x in ds]

    print(f'Running pretrained')
    pretrained_accuracy, pretrained_passes = do_single_run(
        model_name=model_name,
        adapter_name=None, # setting to None so pretrained
        subset=subset,
        ds=ds,
        batch_size=batch_size,
        num_repeat=num_repeat,
        answers=answers,
    )
    print(f"Base: Accuracy: {pretrained_accuracy:0.3f}, Pass@{num_repeat}: {pretrained_passes:0.3f}")
        
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
            subset=subset,
            ds=ds,
            batch_size=batch_size,
            num_repeat=num_repeat,
            answers=answers,
        )
        print(f"Checkpoint: {ckpt_num}: Accuracy: {acc:0.3f}, Pass@{num_repeat}: {pass_at_k:0.3f}")
        accuracies.append(acc)
        passes.append(pass_at_k)

    results = {
        'checkpoint': checkpoint_numbers,
        'accuracy': accuracies,
        f'pass@{num_repeat}': passes,
        'base accuracy' : pretrained_accuracy,
         f'base pass@{num_repeat}': pretrained_passes,
    }
    return results

def normalize_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single tool call by sorting its arguments dictionary.
    """
    if tool_call is None:
        return None
    normalized = tool_call.copy()
    if "arguments" in normalized and isinstance(normalized["arguments"], dict):
        # Sort the arguments dictionary by keys for consistent ordering
        normalized["arguments"] = dict(sorted(normalized["arguments"].items()))
    return normalized

def normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize a list of tool calls by:
    1. Normalizing each individual tool call
    2. Sorting the list by a consistent key (name + serialized arguments)
    """
    normalized_calls = [normalize_tool_call(call) for call in tool_calls]
    
    # Sort by name first, then by serialized arguments for consistent ordering
    def sort_key(call):
        name = call.get("name", "")
        args = call.get("arguments", {})
        # Convert to JSON string for consistent comparison
        args_str = json.dumps(args, sort_keys=True)
        return (name, args_str)
    
    return sorted(normalized_calls, key=sort_key)

def compare_tool_calls(ground_truth: List[Dict[str, Any]], 
                      prediction: List[Dict[str, Any]]) -> bool:
    """
    Compare two lists of tool calls in an order-agnostic way.
    
    Returns True if they are equivalent, False otherwise.
    """
    # Normalize both lists
    norm_ground_truth = normalize_tool_calls(ground_truth)
    norm_prediction = normalize_tool_calls(prediction)
    
    # Compare the normalized lists
    return norm_ground_truth == norm_prediction