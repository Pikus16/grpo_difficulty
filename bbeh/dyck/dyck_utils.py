import re
from datasets import load_dataset
import os

def load_dyck_dataset(subset: str=None, dset_path=None):
    if dset_path is None:
        # get file directory
        dirname = os.path.dirname(os.path.abspath(__file__))
        dset_path = os.path.join(dirname, 'dset')
    if subset is None:
        ds = load_dataset('maveriq/bigbenchhard', 'dyck_languages', split='train')
        ds = ds.rename_column('input', 'question').rename_column('target', 'answer')
    elif subset == 'train' or subset == 'test':
        ds = load_dataset(
            "json", 
            data_files=f'{dset_path}/{subset}.json', 
            split="train")
    else:
        raise ValueError(f'Unknown subset: {subset}')
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
        return str(matches[-1])
    except:
        return None
    
def format_single_question_qwen(question: str, tokenizer):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': f"You are an expert in a language called dyck where you must complete the language sequence of unclosed brackets of all types (e.g., [], {{}}, <>).{question}.\nPut your final answer within \\boxed{{}}. Provide only the rest of the sequence, not the full sequence"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

def format_question_qwen(questions: list[str], tokenizer, device='cuda'):
    prompts = [format_single_question_qwen(q, tokenizer) for q in questions]
    return tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    
