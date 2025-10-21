# %%
import json
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import os
import argparse
from datetime import date

# Parser 
parser = argparse.ArgumentParser(description='Cache Naive Prompt')
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
    '--data_dir',
    type=str,
    default='./data/en/',
    help='path to the folder with data file and save results'
)
parser.add_argument(
    '--prompt_file',
    type=str,
    default="./prompts/naive.txt",
    help='system naive prompt .txt file'
)

args = parser.parse_args()
# %%
# prepare vllm client
#MODEL_NAME = "Llama-3.1-8B-Instruct"
MODEL_NAME = args.model_name #"Llama-3.1-8B-Instruct-qlora-8bit"
MODEL_API = args.model_api #'http://localhost:8000/v1'
datestamp = date.today().strftime("%Y%m%d")
SYSTEM_PROMPT_PATH = args.prompt_file
DATA_PATH = args.data_dir


LANG = "en"
TASK = "naive" # or "finetuned"
OUTPUT_FILE_NAME = f"test_{MODEL_NAME}_{TASK}_{datestamp}"
SYSTEM_PROMPT = open(SYSTEM_PROMPT_PATH, "r").read()


# %%
class ClinicalReport(BaseModel):
    life_style: Optional[str] # Lifestyle information (e.g., smoking, alcohol, exercise)
    family_history: Optional[str] # Family medical history (e.g., genetic disorders, chronic diseases)
    social_history: Optional[str] # Social background (e.g., marital status, substance use)
    medical_surgical_history: Optional[str] # Past medical conditions, surgeries, treatments
    signs_symptoms: Optional[str] # Presenting symptoms, their duration, and severity
    comorbidities: Optional[str] # Coexisting medical conditions
    diagnostic_techniques_procedures: Optional[str] # Diagnostic tests, imaging, and procedures performed
    diagnosis: Optional[str] # Primary and secondary diagnoses
    laboratory_values: Optional[str] # Specific lab results (e.g., blood tests, INR)
    pathology: Optional[str] # Pathological findings (e.g., biopsy results)
    pharmacological_therapy: Optional[str] # Medications prescribed, including dosages and frequency
    interventional_therapy: Optional[str] # Surgical or non-surgical interventions performed
    patient_outcome_assessment: Optional[str] # Patient outcome, follow-up plans, and health status
    age: Optional[str] # Patient's age at the time of the report
    gender: Optional[str] # Patient's gender

vllm_client = OpenAI(
    api_key="EMPTY",
    base_url=MODEL_API
)

# read data
data = json.load(open(f"{DATA_PATH}/test.json"))

for i, d in enumerate(tqdm(data)):
    if 'model_prediction' in d:
        continue

    clinical_report = d['report']
    response_content = None

    try:
        model_response = vllm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Clinical Report: {clinical_report}"}
            ],
            seed=1234,
            temperature=0.1,
            # extra_body=dict(
            #     guided_json=ClinicalReport.model_json_schema(),
            #     guided_decoding_backend="outlines"
            # )
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
    if not os.path.exists(f"{DATA_PATH}/cached_results/"):
        os.makedirs(f"{DATA_PATH}/cached_results/", exist_ok=True)
    if (i+1) % 10 == 0:
        with open(f"{DATA_PATH}/cached_results/{OUTPUT_FILE_NAME}.json", "w") as _file:
            json.dump(data, _file, ensure_ascii=False, indent=4)

with open(f"{DATA_PATH}/cached_results/{OUTPUT_FILE_NAME}.json", "w") as _file:
    json.dump(data, _file, ensure_ascii=False, indent=4)