import os
import sys
import torch
import random
import argparse
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--llm_model_path", type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument("--lora_path", type=str, default='/checkpoints/meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument("--save_path", type=str, default='./merged_model/')
args = parser.parse_args()

save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

ROOT_PATH = Path(__file__).parent
sys.path.append(f"{ROOT_PATH}")

checkpoint = args.lora_path

model = AutoModelForCausalLM.from_pretrained(
    args.llm_model_path,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
# tokenizer.padding_side = 'right'
tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, trust_remote_code=True)

model = PeftModel.from_pretrained(model, checkpoint)

merged_model = model.merge_and_unload()

merged_model.save_pretrained(f'{save_path}', max_shard_size="2GB")
tokenizer.save_pretrained(f'{save_path}')