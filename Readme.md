# QLoRA can closely match accuracy of LoRA while requiring lower compute resources, for clinical data extraction.

This repository contains code used for the study "QLoRA can closely match accuracy of LoRA while requiring lower compute resources, for clinical data extraction". 

## Setup 
1. Clone The reposity 
```bash
git clone https://github.com/dschc/qlora-vs-lora-clinical-SDE.git 

cd qlora-vs-lora-clinical-SDE 

```

2. Crate a python virtual environment 
```bash 
python3 -m venv llmenv 
source lllmenv/bin/activate
```

3. Install dependencies: 
```bash 
pip install -r requirments.txt
```

## Dataset 

### Downlaod 
Download the datseet from [Zenodo](https://zenodo.org/records/14793810) and saved it in '/data' director 

### Training dataset
prepare the training dataset
 ```bash 
 python scripts/preepare_train_chat_data.py 
 ```

 ### Advanced Prompt with ICL 
 1. Run the ICL server 
 ```bash 
 python icl_retriever_app.py
 ``` 

 2. Create the advanced prompt with ICL 
 ```bash 
 python create_advance_prompt.py
 ```

## Model Preparation 
1. Fine tune 

required to pass 

--experiment ['no_quant', '8bit', '4bit'] -no_quant for lora finetune and others for qlora finetune 

you can pass --num_samples with --test_mode True, to test the pipelines. 

it also accept lora_alpha, lora_r, epochs, model_id, data_file, output_base 


```bash 
python FineTune/fineTune.py --experiment '4bit' \
        --num_samples 40 \
        --test_mode true
```
2. Merge Model 

for Qlora model
```bash 
python script/merge_model_qlora.py \
        --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --adapter_path .FineTune/checkpoints/4bit/
        --output_path .FineTune/Merged/4bit/ 
```

for Lora Model 
```bash 
python script/merge_model_lora.py \ 
```

## Inference and Evaluation 

1. Deploy LLM server
```bash 
scripts/run_vllm.sh 2,3 ./FineTuned/merged_model/qlora4bit qlora4bit 2 
```

2. Naive inference 
```bash
python cached_results_naive_batched.py \
      --model_api 'http://localhost:8000/v1' \
      --model_name qlora4bit \
      --batch_size 16 \
      --max_concurrent 8 
```

3. Advanced inference 
```bash 
python cached_advance_prompt_batched.py \
      ----model_api 'http://localhost:8000/v1' \
      --model_name qlora4bit \
      --prompt_file "./prompts/prompt_test_advanced_icl.json" \
      --batch_size 16 \
      --max_concurrent 8 

4. evaluation 
```bash 
python evaluate.py \
      --gpu_id 0 \
      --bert_gpu_id 1
```


## Acknowledgement
This work incorporates and modifies code from the open-source repository [elmtex](https://gitlab.cc-asp.fraunhofer.de/health-open/elmtex). We sincerely thank the original authors and contributors for their valuable efforts in developing and sharing the project.