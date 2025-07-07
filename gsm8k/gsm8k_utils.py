import re
from datasets import load_dataset
import os

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
    

    