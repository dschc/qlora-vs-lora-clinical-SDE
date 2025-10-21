# %% 
import json
import requests
from tqdm import tqdm
from openai import OpenAI
import os
import argparse
from datetime import date

# %%
# icl api
#ICL_APP_API = "http://localhost:8181/retrieve"
# Arguements
parser = argparse.ArgumentParser(description='Cache Advance Prompt')
parser.add_argument(
    '--model_api',
    type=str,
    default='http://localhost:8000/v1'
)
parser.add_argument(
    '--model_name',
    type=str,
    default="Llama-3.1-8B-Instruct-qlora-4bit--host",
    help='Name of the model exposed by vllm'
)
parser.add_argument(
    '--output_dir',
    type=str,
    default='./data/en/',
    help='path to the folder to save file'
)
parser.add_argument(
    '--prompt_file',
    type=str,
    default="./prompts/prompt_test_advanced_icl.json",
    help='Advanced Prompt final json file'
)


args = parser.parse_args()

MODEL_NAME = args.model_name
MODEL_API = args.model_api
DATA_PATH = args.output_dir
PROMPT_PATH = args.prompt_file


# prepare vllm client
#MODEL_NAME = "Llama-3.1-8B-Instruct-qlora-8bit"
#MODEL_API = 'http://localhost:8000/v1'
vllm_client = OpenAI(
    api_key="EMPTY",
    base_url=MODEL_API
)
datestamp = date.today().strftime("%Y%m%d")

# %%

LANG = "en"
OUTPUT_FILE_NAME = f"test_{MODEL_NAME}_advanced_icl_{datestamp}"
#SYSTEM_PROMPT = open("./prompts/advanced_icl.txt", "r").read()


# %% 
# read data
data = json.load(open(PROMPT_PATH))

# %%
for i, d in enumerate(tqdm(data)):
    clinical_report = d['report']
    system_prompt = d["system_prompt"]
    response_content = None

    try:

        model_response = vllm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"### Clinical Report:\n```\n{clinical_report}\n```\n\nOutput:"}
            ],
            seed=1234,
            temperature=0.1
        )
        response_content = model_response.choices[0].message.content
        cleaned_output = response_content.replace("```json\n", "").replace("\n```", "")
        response_json = json.loads(cleaned_output)

        d['model_prediction'] = response_json
    except Exception as e:
        print(e)
        if response_content:
            d['model_prediction'] = response_content
        else:
            d['model_prediction'] = None
    
    os.makedirs(f"{DATA_PATH}/cached_results/", exist_ok=True)

    if (i+1) % 50 == 0:
        with open(f"{DATA_PATH}/cached_results/{OUTPUT_FILE_NAME}.json", "w") as _file:
            json.dump(data, _file, ensure_ascii=False, indent=4)



with open(f"{DATA_PATH}/cached_results/{OUTPUT_FILE_NAME}.json", "w") as _file:
    json.dump(data, _file, ensure_ascii=False, indent=4)