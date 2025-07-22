import unsloth
from unsloth import FastLanguageModel
#from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import os

os.environ['WANDB_PROJECT'] = "GRPO_DIFFICULTY"
os.environ['WANDB_NAME'] = "SFT_on_successes"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

dataset = load_dataset("json", data_files="success_data.jsonl", split="train")

def formatting_prompts_func(examples):
    prompts = examples["prompt"]
    completions      = examples["completion"]
    assert len(prompts) == len(completions)
    
    texts = []
    for prompt, completion in zip(prompts, completions):
        
        text = tokenizer.apply_chat_template([prompt, completion],  tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)



model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

training_args = SFTConfig(
    max_length=2048,
    output_dir="quick_run",
    #completion_only_loss=True ,
    report_to = "wandb",
    max_steps=1000
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    args=training_args
)


#trainer = train_on_responses_only(trainer)

trainer.train()

model.save_lora(f"/home/ubuntu/code/grpo_difficulty/models/photosynthesis/sft_success/lora")