cd /home/pratyush/grpo_difficulty/src

python burton_test.py --model_name unsloth/Qwen3-4B-unsloth-bnb-4bit
# AIME | Accuracy: 0.208, Pass@8: 0.333
python burton_test.py --model_name unsloth/Qwen3-4B-unsloth-bnb-4bit \
  --adapter_name /home/pratyush/grpo_difficulty/checkpoints/gsm8k/8gen_1000steps_unsloth-Qwen3-4B-unsloth-bnb-4bit_strategyeasiest_subsetperc0.1/final
# AIME | Accuracy: 0.200, Pass@8: 0.333
python burton_test.py --model_name unsloth/Qwen3-4B-unsloth-bnb-4bit \
  --adapter_name /home/pratyush/grpo_difficulty/checkpoints/gsm8k/8gen_1000steps_unsloth-Qwen3-4B-unsloth-bnb-4bit_strategyhardest_subsetperc0.1/final
  #AIME | Accuracy: 0.175, Pass@8: 0.400
python burton_test.py --model_name unsloth/Qwen3-4B-unsloth-bnb-4bit \
  --adapter_name /home/pratyush/grpo_difficulty/checkpoints/gsm8k/8gen_1000steps_unsloth-Qwen3-4B-unsloth-bnb-4bit_strategymiddle_subsetperc0.1/final
# AIME | Accuracy: 0.200, Pass@8: 0.267
python burton_test.py --model_name unsloth/Qwen3-4B-unsloth-bnb-4bit \
  --adapter_name /home/pratyush/grpo_difficulty/checkpoints/gsm8k/8gen_1000steps_unsloth-Qwen3-4B-unsloth-bnb-4bit_strategyrandom_subsetperc0.1/final
# AIME | Accuracy: 0.175, Pass@8: 0.333