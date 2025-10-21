import torch
from transformers import TrainerCallback
import time 
import json

class MultiGPUMemoryTracker:
    """Track GPU memory usage across all available GPUs throughout training"""
    
    def __init__(self):
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.memory_log = []
        self.start_time = None
        self.init_end_time = None
        self.training_end_time = None
        
    def get_memory_stats(self, stage=""):
        """Get current GPU memory statistics for all GPUs"""
        if not torch.cuda.is_available():
            return None
            
        torch.cuda.synchronize()
        
        stats = {
            "stage": stage,
            "timestamp": time.time(),
            "gpus": {}
        }
        
        for gpu_id in range(self.num_gpus):
            stats["gpus"][f"gpu_{gpu_id}"] = {
                "allocated_gb": torch.cuda.memory_allocated(gpu_id) / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved(gpu_id) / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated(gpu_id) / 1024**3,
                "max_reserved_gb": torch.cuda.max_memory_reserved(gpu_id) / 1024**3,
            }
        
        self.memory_log.append(stats)
        return stats
    
    def print_memory(self, stage=""):
        """Print memory statistics for all GPUs"""
        stats = self.get_memory_stats(stage)
        if stats:
            print(f"\n{'='*60}")
            print(f"GPU Memory - {stage}")
            print(f"{'='*60}")
            for gpu_id in range(self.num_gpus):
                gpu_key = f"gpu_{gpu_id}"
                gpu_stats = stats["gpus"][gpu_key]
                print(f"\nGPU {gpu_id}:")
                print(f"  Allocated:     {gpu_stats['allocated_gb']:.2f} GB")
                print(f"  Reserved:      {gpu_stats['reserved_gb']:.2f} GB")
                print(f"  Max Allocated: {gpu_stats['max_allocated_gb']:.2f} GB")
                print(f"  Max Reserved:  {gpu_stats['max_reserved_gb']:.2f} GB")
            print(f"{'='*60}\n")
    
    def reset_peak_stats(self):
        """Reset peak memory statistics for all GPUs"""
        for gpu_id in range(self.num_gpus):
            torch.cuda.reset_peak_memory_stats(gpu_id)
    
    def get_peak_memory_all_gpus(self):
        """Get peak memory for all GPUs"""
        peak_memory = {}
        for gpu_id in range(self.num_gpus):
            peak_memory[f"gpu_{gpu_id}"] = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
        return peak_memory
    
    def save_log(self, filepath):
        """Save memory log to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.memory_log, f, indent=2)
        print(f"Memory log saved to: {filepath}")
    
    def mark_start(self):
        """Mark the start time"""
        self.start_time = time.time()
    
    def mark_init_end(self):
        """Mark the end of initialization"""
        self.init_end_time = time.time()
    
    def mark_training_end(self):
        """Mark the end of training"""
        self.training_end_time = time.time()
    
    def get_timing_stats(self):
        """Get timing statistics"""
        return {
            "initialization_time_seconds": self.init_end_time - self.start_time if self.init_end_time and self.start_time else 0,
            "training_time_seconds": self.training_end_time - self.init_end_time if self.training_end_time and self.init_end_time else 0,
        }


class MemoryLoggingCallback(TrainerCallback):
    """Callback to log memory during training"""
    
    def __init__(self, tracker):
        self.tracker = tracker
        self.step_count = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log memory every N steps"""
        self.step_count += 1
        if self.step_count % 10 == 0:  # Log every 10 steps
            self.tracker.get_memory_stats(f"Step {self.step_count}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log memory at end of each epoch"""
        self.tracker.print_memory(f"Epoch {state.epoch} End")


def count_model_parameters(model):
    """Count trainable and total parameters"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n{'='*60}")
    print("Model Parameters")
    print(f"{'='*60}")
    print(f"Total Parameters:     {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Trainable %:          {100 * trainable_params / total_params:.4f}%")
    print(f"{'='*60}\n")
    
    return trainable_params, total_params


def print_training_config(model_id, lora_config, batch_size, grad_acc, quantization_bits):
    """Print training configuration for memory calculation"""
    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Quantization: {quantization_bits}-bit")
    print(f"LoRA Rank (r): {lora_config.r}")
    print(f"LoRA Alpha: {lora_config.lora_alpha}")
    print(f"LoRA Dropout: {lora_config.lora_dropout}")
    print(f"Target Modules: {lora_config.target_modules}")
    print(f"Batch Size: {batch_size}")
    print(f"Gradient Accumulation: {grad_acc}")
    print(f"Effective Batch Size: {batch_size * grad_acc}")
    print(f"{'='*60}\n")