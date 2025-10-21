import json
import requests
from tqdm import tqdm
from openai import OpenAI
import os
import argparse 

## Parse 
parser = argparse.ArgumentParser(description='Create advance prompts')
parser.add_argument(
    '--prompt_path',
    default='./prompts',
    help = 'path to the prompt files, and store created new prompt'
)
parser.add_argument(
    '--icl_api',
    default="http://localhost:8181/retrieve",
    help='API for the retreivel server'
)
parser.add_argument(
    '--top_k',
    type= int, 
    default= 5, 
    help='Number of ICL examples to extract'
)

parser.add_argument(
    '--data_path',
    type=str,
    default='./data/en'
)

args = parser.parse_args()
prompt_path = args.prompt_path
DATA_PATH = args.data_path
top_k = args.top_k
ICL_APP_API = args.icl_api #"http://localhost:8181/retrieve"

# prepare vllm client
#MODEL_NAME = "Llama-3.1-8B-Instruct-qlora-4bit"
#MODEL_API = 'http://localhost:8000/v1'

LANG = "en"
OUTPUT_FILE_NAME = f"prompt_test_advanced_icl"

SYSTEM_PROMPT = open(f"{prompt_path}/advanced_icl.txt", "r").read()
#DATA_PATH = f"./data/{LANG}"

"""
vllm_client = OpenAI(
    api_key="EMPTY",
    base_url=MODEL_API
)
"""

# read data
data = json.load(open(f"{DATA_PATH}/test.json"))


def get_icl_examples(report, top_k=top_k):
    data = {
        'query': report,
        'top_k': top_k
    }

    response = requests.post(ICL_APP_API, json=data)

    if response.status_code == 200:
        response_data = json.loads(response.text)
        return response_data.get('results')
    else:
        print("API request failed with status code:", response.status_code)
        return None

def prepare_prompt(examples):
    formatted_prompt = "### Examples:\n\n"
    for i, example in enumerate(examples):
        report = example.get('report')
        summary = example.get('summary')
        formatted_prompt += f"### Clinical Report {i+1}:\n"
        formatted_prompt += f"```\n"
        formatted_prompt += report + "\n"
        formatted_prompt += f"```\n\n"
        formatted_prompt += f"Output:\n\n"
        formatted_prompt += f"```json\n"
        formatted_prompt += summary + "\n"
        formatted_prompt += f"```\n\n"
        formatted_prompt += f"---\n"

    formatted_prompt += "Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000."

    return f"{SYSTEM_PROMPT}\n{formatted_prompt}"

for i, d in enumerate(tqdm(data)):
    clinical_report = d['report']
    response_content = None

    try:
        icl_examples = get_icl_examples(clinical_report)
        d["system_prompt"] = prepare_prompt(icl_examples)
    except Exception as e:
        print(e)
        if response_content:
            d['system_prompt'] = response_content
        else:
            d['system_prompt'] = None
    
    os.makedirs(f"{DATA_PATH}/prompts/", exist_ok=True)

    if (i+1) % 50 == 0:
        with open(f"{prompt_path}/{OUTPUT_FILE_NAME}.json", "w") as _file:
            json.dump(data, _file, ensure_ascii=False, indent=4)



with open(f"{prompt_path}/{OUTPUT_FILE_NAME}.json", "w") as _file:
    json.dump(data, _file, ensure_ascii=False, indent=4)