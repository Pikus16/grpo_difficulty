import re
from datasets import load_dataset

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