import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import click
import os
from gsm8k_utils import load_gsm8k_dataset

def build_model_and_tokenizer(model_name, device: str = 'cuda'):
    # 1) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = (
        AutoModelForCausalLM
        .from_pretrained(model_name, trust_remote_code=True)
        .to(device)
    )
    model.eval()

    # 2) (Optional) Compile for speed if you're on PyTorch 2.x
    if torch.backends.cuda.is_built():
        try:
            model = torch.compile(model)
        except Exception:
            pass

    return model, tokenizer

def format_question(questions: list[str], tokenizer):
    prompts = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': f"{q}.\nPut your final answer within \\boxed{{}}."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for q in questions
    ]
    return tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

def sample_pass_at_k(
    model,
    tokenizer,
    questions: list[str],
    k: int = 8,
    max_new_tokens: int = 250,
    temperature: float = 1.0,
    top_p: float = 1,
) -> list[str]:
    
    inputs = format_question(questions, tokenizer)
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

@click.command()
@click.option(
    '--dataset-name', '-d',
    default='openai/gsm8k',
    show_default=True,
    help="HuggingFace dataset name"
)
@click.option(
    '--subset', '-s',
    default='train',
    show_default=True,
    help="Dataset split to use"
)
@click.option(
    '--model-name', '-m',
    default='unsloth/Qwen3-4B-unsloth-bnb-4bit',
    show_default=True,
    help="Model name or path"
)
@click.option(
    '--num-repeat', '-k',
    default=8,
    show_default=True,
    help="Number of samples to generate per question"
)
@click.option(
    '--output_folder', '-o',
    default='gsm8k_output',
    show_default=True,
    help="Output folder to write all responses"
)
@click.option('--batch-size', '-b', default=4, 
              show_default=True, help="Number of questions to batch together")
def main(
    dataset_name: str,
    subset: str,
    model_name: str,
    num_repeat: int,
    output_folder: str,
    batch_size: int,
):
    ds = load_gsm8k_dataset(dataset_name, subset=subset)
    model, tokenizer = build_model_and_tokenizer(model_name=model_name)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    output_folder = os.path.join(output_folder, f"{dataset_name.replace('/','-')}_{subset}")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    output_file = os.path.join(output_folder, f"{model_name.replace('/','-')}.json")

    all_responses = []
    for i in tqdm(range(0, len(ds), batch_size)):
        batch = ds[i: i + batch_size]
        questions = batch['question']
        responses = sample_pass_at_k(model, tokenizer, questions, k=num_repeat)
        all_responses.extend(responses)

        if i % (10 * batch_size) == 0:
            torch.cuda.empty_cache()
            write_to_file(output_file, all_responses)

    write_to_file(output_file, all_responses)


if __name__ == '__main__':
    main()

