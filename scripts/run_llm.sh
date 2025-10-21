#!/bin/bash
#SBATCH --job-name=run_llm
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

# export Huggingface token for downloading private models (or models requiring access e.g., llama 3)

# Configuration 
#model="/home/psig/elmtex_prabin/FineTune/merged_model/8bit"  # Path to the merged model
#served_model_name="Llama-3.1-8B-Instruct-qlora-8bit"
#download_dir="/models/vllm"
CUDA_DEVICES=${1:-"2,3"}
MODEL=${2:-"./FineTune/merged_model/lora"}
SERVED_MODEL_NAME=${3:-"Llama-3.1-8B-Instruct-lora"}
TP_SIZE=${4:-"2"}  

#port=9000
# check model path exisit 
# 1. Check if model directory exists:;
echo "1. Checking model path..."
if [ -d "$MODEL" ]; then
    echo "Model directory exists"
    ls -lh $MODEL
else
    echo "✗ Model directory NOT found: $MODEL"
    exit 1
fi
#get port and GPUs dynamically
#if [ -z "$1" ] || [ -z "$2" ]; then
    #echo "Usage: $0 <port_number> <tensor_parallel_size>"
    #exit 1
#fi
port=8000
#tensor_parallel_size=2

echo "Model: $MODEL"
echo "Served model name for api calls: $SERVED_MODEL_NAME"
#echo "Download directory for model: $download_dir"
echo "Hostname: $(hostname -I | grep -o '\b10\.[0-9]\+\.[0-9]\+\.[0-9]\+\b' | head -n 1)"
echo "Port: $port"
echo "Running model..."

export CUDA_VISIBLE_DEVICES="2,3"
vllm serve $MODEL \
--served-model-name $SERVED_MODEL_NAME\
--host "0.0.0.0" \
--port $port \
--max-model-len 15360 \
--tensor-parallel-size $TP_SIZE \
--gpu-memory-utilization 0.85 \
--dtype bfloat16 \

#--download-dir $download_di
#15360