# %% 
# import necessary libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import json
import time
import torch
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainerCallback
from memory_utils import *
import argparse
from trl import SFTConfig, SFTTrainer

# %% 
# argparse for command line arguments
parser = argparse.ArgumentParser(description='LoRA Fine-tuning Comparison Experiments')
parser.add_argument(
    '--experiment', 
    type=str, 
    required=True,
    choices=['no_quant', '8bit', '4bit'],
    help='Type of experiment: no_quant, 8bit, or 4bit'
)
parser.add_argument(
    '--num_samples',
    type=int,
    default=100,
    help='Number of random samples to use for training (default: 100)'
)
parser.add_argument(
    '--random_seed',
    type=int,
    default=42,
    help='Random seed for reproducibility (default: 42)'
)
parser.add_argument(
    '--model_id',
    type=str,
    default='meta-llama/Meta-Llama-3.1-8B-Instruct',
    help='Model ID to use (default: Meta-Llama-3.1-8B-Instruct)'
)
parser.add_argument(
    '--data_file',
    type=str,
    default='../data/train_chat_en.json',
    help='Path to training data file (default: ./data/train_chat_en.json)'
)
parser.add_argument(
    '--output_base',
    type=str,
    default='./model/experiments',
    help='Base directory for outputs (default: ./model/experiments)'
)
parser.add_argument(
    '--epochs',
    type=int,
    default=1,
    help='Number of training epochs (default: 1)'
)
parser.add_argument(
    '--lora_r',
    type=int,
    default=256,
    help='LoRA rank (default: 256)'
)
parser.add_argument(
    '--lora_alpha',
    type=int,
    default=128,
    help='LoRA alpha (default: 128)'
)
parser.add_argument(
    '--test_mode',
    type=bool,
    default=True,
    help='Test mode (default: True)'
)

args = parser.parse_args()
# %%
model_id  = args.model_id  # e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAMPLE_LIMIT = args.num_samples  # e.g., 100
EPOCHS = args.epochs  # e.g., 1
LORA_R = args.lora_r  # e.g., 256
LORA_ALPHA = args.lora_alpha  # e.g., 128
DATA = args.data_file  # e.g., "../data/train_chat_en.json"
output_base = args.output_base  # e.g., "./model/experiments"
BATCH_SIZE = 1
GRAD_ACC_STEPS = 16
RANDOM_SEED = args.random_seed  # e.g., 42
EXPERIMENT_TYPE = args.experiment  # Options: "no_quant", "qlora_4bit", "qlora_8bit"
OUTPUT_DIR = f"{output_base}/{EXPERIMENT_TYPE}/{model_id}"
TEST_MODE = args.test_mode  # Set to False for full training
# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# %% 
memory_tracker = MultiGPUMemoryTracker()
memory_tracker.mark_start()

# Print GPU information
print(f"\n{'='*60}")
print("GPU Information")
print(f"{'='*60}")
print(f"Number of GPUs available: {memory_tracker.num_gpus}")
for i in range(memory_tracker.num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"{'='*60}\n")

# Initial memory state
memory_tracker.print_memory("Initial State")

# Load dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA, split="train")

"""
if TEST_MODE == True:
    dataset = dataset.select(range(SAMPLE_LIMIT))
    print(f"TEST MODE: Dataset limited to {len(dataset)} examples")
else:
    dataset = dataset # Use full dataset
    print(f"Dataset loaded: {len(dataset)} examples")
"""
print(f"Dataset loaded: {len(dataset)} examples")

memory_tracker.print_memory("After Dataset Load")



# BitsAndBytes Configuration
if EXPERIMENT_TYPE == "4bit":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        )
# QLoRA Configuration (8-bit)
elif EXPERIMENT_TYPE == "8bit":
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )
else:
    bnb_config = None


# Load model 
print(f"\nLoading model: {model_id}")
print(f"Quantization: {EXPERIMENT_TYPE}")
if EXPERIMENT_TYPE == "no_quant":
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
elif EXPERIMENT_TYPE == "8bit":
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "24GB", 1: "24GB", 3:"24GB"},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    
elif EXPERIMENT_TYPE == "4bit":
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
memory_tracker.print_memory(f"After Model Load {EXPERIMENT_TYPE}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# LoRA configuration
peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM"
)

# Print configuration
print_training_config(model_id, peft_config, BATCH_SIZE, GRAD_ACC_STEPS, quantization_bits=EXPERIMENT_TYPE)

if EXPERIMENT_TYPE == "8bit":
    # Training arguments (only works with old version of transformers,trl,peft, torch)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        seed=RANDOM_SEED,
    )

    max_seq_length = 5120

    # Create trainer with memory callback
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        callbacks=[MemoryLoggingCallback(memory_tracker)]
    )
else:
    sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,                     # Start with one epoch
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    optim="adamw_torch_fused",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    logging_steps=10,                      # Log metrics every 1                     # Log metrics every 250 steps
    save_strategy="epoch",                  # Save checkpoint every N steps                         # Save a checkpoint every 250 steps
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    packing=False,
    max_length=5120
    )

    experiment_callback = MemoryLoggingCallback(memory_tracker)
    trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_config,
    callbacks=[experiment_callback],)

trainable_params, total_params = count_model_parameters(trainer.model)

memory_tracker.print_memory("After Trainer Setup (Before Training)")

# Mark end of initialization
memory_tracker.mark_init_end()

# Reset peak memory stats before training
memory_tracker.reset_peak_stats()

print("Starting training...")
print(f"This will train for {EPOCHS} epochs")
print(f"Total steps: {len(dataset) // (BATCH_SIZE * GRAD_ACC_STEPS) * EPOCHS}\n")

# Start training
train_start_time = time.time()
trainer.train()
train_time = time.time() - train_start_time

memory_tracker.mark_training_end()
memory_tracker.print_memory("After Training Complete")

# save model 
trainer.save_model()
print(f"\nModel saved to: {OUTPUT_DIR}")

# Get final statistics
peak_memory_gb = memory_tracker.get_peak_memory_all_gpus()
timing_stats = memory_tracker.get_timing_stats()

# Create results dictionary
results = {
    "experiment_type": EXPERIMENT_TYPE,
    "model_id": model_id,
    "num_samples": len(dataset),
    "random_seed": RANDOM_SEED,
    "initialization_time_seconds": timing_stats["initialization_time_seconds"],
    "training_time_seconds": timing_stats["training_time_seconds"],
    "peak_memory_gb": peak_memory_gb,
    "num_gpus": memory_tracker.num_gpus,
    "training_config": {
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACC_STEPS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACC_STEPS,
        #"seq_length": max_seq_length,
        "lora_rank": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        #"quantization_bits": 8,
        "trainable_params_millions": trainable_params / 1e6,
        "total_params_billions": total_params / 1e9,
    }
}

# Save results to JSON
results_path = os.path.join(OUTPUT_DIR, "training_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {results_path}")

# Save detailed memory log
memory_log_path = os.path.join(OUTPUT_DIR, "memory_log.json")
memory_tracker.save_log(memory_log_path)

# Print final summary
print(f"\n{'='*60}")
print("Training Summary")
print(f"{'='*60}")
print(f"Initialization Time: {timing_stats['initialization_time_seconds']:.2f} seconds")
print(f"Training Time:       {timing_stats['training_time_seconds']:.2f} seconds")
print(f"Total Time:          {timing_stats['initialization_time_seconds'] + timing_stats['training_time_seconds']:.2f} seconds")
print(f"\nPeak GPU Memory:")
for gpu_id, memory in peak_memory_gb.items():
    print(f"  {gpu_id}: {memory:.2f} GB")
print(f"\nTrainable Parameters: {trainable_params/1e6:.2f}M")
print(f"Total Parameters:     {total_params/1e9:.2f}B")
print(f"{'='*60}\n")

# Print results in the requested format
print("\n" + "="*60)
print("RESULTS JSON:")
print("="*60)
print(json.dumps(results, indent=2))
print("="*60)