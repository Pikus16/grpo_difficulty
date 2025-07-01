import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import click
import os
import numpy as np
from dyck_utils import load_dyck_dataset, format_question_qwen, extract_boxed_content

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
    k: int = 8,
    max_new_tokens: int = 250,
    temperature: float = 1.0,
    top_p: float = 1,
) -> list[str]:
    
    inputs = format_question_qwen(questions, tokenizer)
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
    '--model-name', '-m',
    default='unsloth/Qwen3-4B-unsloth-bnb-4bit',
    show_default=True,
    help="Model name or path"
)
@click.option(
    '--num-repeat', '-k',
    default=10,
    show_default=True,
    help="Number of samples to generate per question"
)
@click.option(
    '--output_folder', '-o',
    default='dyck_outputs',
    show_default=True,
    help="Output folder to write all responses"
)
@click.option('--batch-size', '-b', default=16, 
              show_default=True, help="Number of questions to batch together")
@click.option('--adapter-name', '-a', default=None, 
              show_default=True, help="Adapter name")
def main(
    model_name: str,
    num_repeat: int,
    output_folder: str,
    batch_size: int,
    adapter_name: str
):
    ds = load_dyck_dataset()
    answers = [x['answer'] for x in ds]
        
    model, tokenizer = build_model_and_tokenizer(model_name=model_name, adapter_name=adapter_name)

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if adapter_name is None:
            output_file = os.path.join(output_folder, f"{model_name.replace('/','-')}.json")
        else:
            output_file = os.path.join(output_folder, f"{os.path.basename(adapter_name)}.json")

        print(f"Output file: {output_file}")

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                all_responses = json.load(f)
            print(f"Loaded {len(all_responses)} responses from {output_file}")
        else:
            all_responses = []
    else:
        all_responses = []

    for i in tqdm(range(0, len(ds), batch_size)):
        if i < len(all_responses):
            continue
        
        batch = ds[i: i + batch_size]
        questions = batch['question']
        responses = sample_pass_at_k(model, tokenizer, questions, k=num_repeat)
        all_responses.extend(responses)
        if output_folder is not None:
            if i % (10 * batch_size) == 0:
                #torch.cuda.empty_cache()
                write_to_file(output_file, all_responses)

    if output_folder is not None:
        write_to_file(output_file, all_responses)

    # get accuracies and pass@k
    assert len(answers) == len(all_responses)
    accs, pass_at_k = [], []
    for answer, responses in zip(answers, all_responses):
        preds = np.array([extract_boxed_content(r) for r in responses])
        accs.append(np.mean(answer == preds))
        pass_at_k.append(1 if answer in preds else 0)
    print(f"Accuracy: {np.mean(accs)}")
    print(f"Pass@{num_repeat}: {np.mean(pass_at_k)}")


if __name__ == '__main__':
    main()
