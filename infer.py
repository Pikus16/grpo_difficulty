from unsloth import FastLanguageModel
import torch
from tqdm import tqdm
import json
import textstat

def load_model(name):
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    if name != 'base':
        model.load_adapter(f'models/photosynthesis_{name}/lora')
    _ = FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    return model, tokenizer

generation_kwargs = {
    "max_new_tokens": 250,
    "use_cache": True,
    "temperature": 0.9,
    "top_k": None,
    "do_sample": True,
}

NUM_TIMES_TO_GEN = 100
BATCH_SIZE = 8

prompt = 'Describe photosynthesis. Use as simple terms as possible'

def generate(model, tokenizer):
    formatted_prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
    [
        formatted_prompt
    ]*BATCH_SIZE, return_tensors = "pt").to("cuda")

    all_outputs = []
    for _ in tqdm(range(0, NUM_TIMES_TO_GEN, BATCH_SIZE)):
        outputs = model.generate(**inputs, **generation_kwargs)
        output = tokenizer.batch_decode(outputs)
        outputs = outputs[:, inputs.input_ids.shape[1]:]
        output = tokenizer.batch_decode(outputs)
        all_outputs.extend(output)
    return all_outputs

for name in ['0.1_longtrain']:
    model_, tokenizer_ = load_model(name)
    all_outputs = generate(model_, tokenizer_)
    with open(f'outputs/photosynthesis/{name}.json', 'w') as f:
        json.dump({'prompt': prompt, 'outputs': all_outputs}, f)