import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse
# Merge Strategy : Merge with Up-Quantization of Base Model 
# argparse 
parser = argparse.ArgumentParser(description='Merge adapters with up-quantization')
parser.add_argument(
    '--base_model', 
    type=str, 
    default='meta-llama/Meta-Llama-3.1-8B-Instruct',
    help='Model ID to use (default: Meta-Llama-3.1-8B-Instruct) or local dir to path'
)
parser.add_argument(
    '--adapter_path',
    type=str,
    default='./FineTune/model/experiments/8bit/meta-llama/Meta-Llama-3.1-8B-Instruct/',
    help='path to the adapter weights'
)
parser.add_argument(
    '--output_path',
    type=str,
    default='./FineTune/merged_model/8bit/',
    help='path to save the merged model'
)
args = parser.parse_args()
# trained model path
ADAPTER_PATH = args.adapter_path
BASE_MODEL = args.base_model
OUTPUT_PATH = args.output_path



def merge_with_upquantization(base_model_id, adapter_path, output_path):
    """
    Merge QLoRA adapter with up-quantized base model
    """
    
    # Step 1: Load base model in higher precision (BF16 instead of 8-bit)
    print("Loading base model in BF16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,  # Up-quantize from 8-bit to BF16
        device_map="auto",
        trust_remote_code=True,
        # No quantization_config - load in full precision
    )
    
    # Step 2: Load your QLoRA adapter
    print("Loading QLoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model, 
        adapter_path,
        torch_dtype=torch.bfloat16
    )
    
    # Step 3: Merge at higher precision
    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    
    # Step 4: Ensure consistent precision
    print("Ensuring consistent precision...")
    for param in merged_model.parameters():
        param.data = param.data.to(torch.bfloat16)
    
    # Step 5: Save merged model
    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("Merge completed!")
    return merged_model

# Execute merge
merged_model = merge_with_upquantization(BASE_MODEL, ADAPTER_PATH, OUTPUT_PATH)