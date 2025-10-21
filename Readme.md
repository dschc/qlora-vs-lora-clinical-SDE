#QLoRA can closely match accuracy of LoRA while requiring lower compute resources, for clinical data extraction.
# ELMTEX: Fine-Tuning Large Language Models for Structured Clinical Information Extraction. A Case Study on Clinical Reports

![ELMTEX project experiments.](/resources/approach.png)

## Folder structure 
- elmtex 
  - data
    - de 
    - en
  - model 
  - prompts
  - scripts
  



## Requirements and Setup

To set up the environment, follow these steps:

```bash
# Clone the repository
git clone https://gitlab.cc-asp.fraunhofer.de/health-open/elmtex.git
cd elmtex

# Install dependencies
pip install -r requirements.txt
```

## Download ELMTEX Dataset

Download the dataset from [Zenodo](https://zenodo.org/records/14793810) and place it in the `data/` directory.

## Experiments

We use [vLLM](https://github.com/vllm-project/vllm) to run large language models (LLMs). Example script: [run_llm.sh](/scripts/run_llm.sh).

### Naive Prompt

First, we cache the model responses using the [naive prompt](/prompts/naive.txt):

```bash
python cache_results_naive.py
```

Then, run the evaluation script (ensure `TEST_FILE_NAME` is correctly set):

```bash
python evaluate.py
```

### Advanced Prompt with In-Context Learning (ICL)

To run experiments using an advanced prompt with ICL, first start the ICL retriever application:

```bash
python icl_retriever_app.py
```

Next, cache model responses using the [advanced ICL prompt](/prompts/advanced_icl.txt):

```bash
python cache_results_advanced_icl.py
```

Then, run the evaluation script (ensure `TEST_FILE_NAME` is correctly set):

```bash
python evaluate.py
```

### Fine-Tuning

#### Step 1: Prepare Training Data

```bash
python scripts/prepare_train_chat_data.py
```

#### Step 2: Fine-Tune the Model

GPU access is required for this step:

```bash
python finetune.py
```

#### Step 3: Merge LoRA Adapters with the Pretrained Model

```bash
python scripts/merge_model_lora.py
```

#### Step 4: Cache Model Responses Using Naive Prompt

After fine-tuning, we cache the model responses using the naive prompt script:

```bash
python cache_results_naive.py
```

#### Step 5: Run Evaluation

Ensure `TEST_FILE_NAME` is correctly set before running:

```bash
python evaluate.py
```

## License

This repository is licensed under the [Apache 2.0 License](/LICENSE).

## Citation

### Paper

```bash
@misc{guluzade2025elmtexfinetuninglargelanguage,
      title={ELMTEX: Fine-Tuning Large Language Models for Structured Clinical Information Extraction. A Case Study on Clinical Reports},
      author={Aynur Guluzade and Naguib Heiba and Zeyd Boukhers and Florim Hamiti and Jahid Hasan Polash and Yehya Mohamad and Carlos A. Velasco},
      year={2025},
      eprint={2502.05638},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.05638},
}
```

### Dataset

```bash
@dataset{guluzade_2025_14793810,
  author       = {Guluzade, Aynur and
                  Heiba, Naguib and
                  Boukhers, Zeyd and
                  Hamiti, Florim and
                  Polash, Jahid Hasan and
                  Mohamad, Yehya and
                  Velasco, Carlos A.},
  title        = {ELMTEX Dataset: Fine-Tuning Large Language Models for Structured Clinical Information Extraction. A Case Study on Clinical Reports},
  month        = feb,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14793810},
  url          = {https://doi.org/10.5281/zenodo.14793810},
}
```

### Appendix

```bash
@misc{guluzade_2025_14837206,
  author       = {Guluzade, Aynur and
                  Heiba, Naguib and
                  Boukhers, Zeyd and
                  Hamiti, Florim and
                  Polash, Jahid Hasan and
                  Mohamad, Yehya and
                  Velasco, Carlos A.},
  title        = {Appendix - ELMTEX: Fine-Tuning Large Language Models for Structured Clinical Information Extraction. A Case Study on Clinical Reports},
  month        = feb,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14837206},
  url          = {https://doi.org/10.5281/zenodo.14837206},
}
```
