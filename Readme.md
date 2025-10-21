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

1. Download the datseet from [Zenodo](https://zenodo.org/records/14793810) and saved it in '/data' director 

2. prepare the dataset 

## Finetuning 

## Inference and Evaluation 


## Acknowledgement
This work incorporates and modifies code from the open-source repository [elmtex](https://gitlab.cc-asp.fraunhofer.de/health-open/elmtex). We sincerely thank the original authors and contributors for their valuable efforts in developing and sharing the project.