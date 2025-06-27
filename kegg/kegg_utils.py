import unsloth
from unsloth import FastLanguageModel
from datasets import load_dataset
import re

def load_train_model_and_tokenizer(model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                             max_seq_length: int = 4000,
                             lora_rank: int = 16,
                             load_in_4bit = False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer
    
def load_test_model_and_tokenizer(model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                             adapter_path=None,
                             max_seq_length: int = 4000,
                             lora_rank: int = 16,
                             load_in_4bit = False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9,
    )
    if adapter_path is not None:
        model.load_adapter(adapter_path)
    _ = FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    return model, tokenizer

def format_kegg_prompt(example, include_sequence=False):
    prompt = f"Solve the following genomic pathway question:\n{example['question']}\n"
    if include_sequence:
        prompt += f'\nReference Sequence: {example["reference_sequence"]}\nVariant Sequence: {example["variant_sequence"]}\n'
    prompt += f'Think step-by-step. Then, provide the final answer on a new line in the format "Answer: DISEASE NAME" with no extra formatting'
    return {'prompt': prompt}

def load_kegg_dataset(split = 'train'):
    ds = load_dataset('wanglab/kegg', 'default', split=split)
    ds = ds.map(format_kegg_prompt)
    return ds

RESPONSE_PATTERN = re.compile(r"Answer:\s*(.+)")
def parse_llm_responses(text):
    """
    Extracts the final answer from the LLM output.
    Expects the format: "Answer: XXX" 
    
    Args:
        text (str): The output from the LLM.
    
    Returns:
        str or None: The extracted str answer, or None if not found.
    """
    match = RESPONSE_PATTERN.search(text)
    try:
        if match:
            return match.group(1).strip().lower()
    except:
        return None
    return None