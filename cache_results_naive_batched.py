# %%
import json
import asyncio
from tqdm import tqdm
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Optional
import os
import argparse
from datetime import date

# Parser 
parser = argparse.ArgumentParser(description='Cache Naive Prompt with Async Batch Processing')
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
parser.add_argument(
    '--batch_size',
    type=int,
    default=16,
    help='Batch size for processing (default: 16)'
)
parser.add_argument(
    '--max_concurrent',
    type=int,
    default=8,
    help='Maximum concurrent requests (default: 8)'
)

args = parser.parse_args()

# %%
# Configuration
MODEL_NAME = args.model_name
MODEL_API = args.model_api
datestamp = date.today().strftime("%Y%m%d")
SYSTEM_PROMPT_PATH = args.prompt_file
DATA_PATH = args.data_dir
BATCH_SIZE = args.batch_size
MAX_CONCURRENT = args.max_concurrent

LANG = "en"
TASK = "naive"
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

# Prepare async vLLM client
vllm_client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=MODEL_API
)

# Read data
data = json.load(open(f"{DATA_PATH}/test.json"))

# Filter out already processed items
unprocessed_data = [(i, d) for i, d in enumerate(data) if 'model_prediction' not in d]
total_items = len(data)
unprocessed_count = len(unprocessed_data)

print(f"Total items: {total_items}")
print(f"Already processed: {total_items - unprocessed_count}")
print(f"Items to process: {unprocessed_count}")

async def process_single_item(client, index, item, semaphore):
    """Process a single item with semaphore for concurrency control"""
    async with semaphore:
        clinical_report = item['report']
        
        try:
            model_response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Clinical Report: {clinical_report}"}
                ],
                seed=1234,
                temperature=0.1,
                # Uncomment below if you want to use guided JSON generation
                # extra_body=dict(
                #     guided_json=ClinicalReport.model_json_schema(),
                #     guided_decoding_backend="outlines"
                # )
            )
            response_content = model_response.choices[0].message.content
            cleaned_output = response_content.replace("```json\n", "").replace("\n```", "")
            response_json = json.loads(cleaned_output)
            
            item['model_prediction'] = response_json
            return index, item, None
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            item['model_prediction'] = None
            return index, item, error_msg

async def process_batch_async(client, batch_data, max_concurrent):
    """Process a batch of items concurrently"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = [process_single_item(client, idx, item, semaphore) for idx, item in batch_data]
    results = await asyncio.gather(*tasks)
    
    return results

async def main_async():
    """Main async function to process all data"""
    if unprocessed_count == 0:
        print("All items already processed!")
        return
    
    print(f"\nProcessing {unprocessed_count} items in batches of {BATCH_SIZE} with {MAX_CONCURRENT} concurrent requests...")
    
    processed_count = 0
    error_count = 0
    
    # Process data in batches
    for batch_idx in tqdm(range(0, len(unprocessed_data), BATCH_SIZE), desc="Processing batches"):
        batch_end = min(batch_idx + BATCH_SIZE, len(unprocessed_data))
        batch = unprocessed_data[batch_idx:batch_end]
        
        # Process the batch asynchronously
        results = await process_batch_async(vllm_client, batch, MAX_CONCURRENT)
        
        # Update the original data with results
        for original_idx, processed_item, error in results:
            data[original_idx] = processed_item
            processed_count += 1
            if error:
                error_count += 1
                print(f"\nError processing item {original_idx}: {error}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(f"{DATA_PATH}/cached_results/{MODEL_NAME}"):
            os.makedirs(f"{DATA_PATH}/cached_results/{MODEL_NAME}", exist_ok=True)

        # Save progress periodically
        if (batch_end) % 10 == 0 or batch_end == len(unprocessed_data):
            with open(f"{DATA_PATH}/cached_results/{MODEL_NAME}/{OUTPUT_FILE_NAME}.json", "w") as _file:
                json.dump(data, _file, ensure_ascii=False, indent=4)
            print(f"\nProgress saved: {batch_idx + len(batch)}/{unprocessed_count} items completed ({error_count} errors)")
    
    # Final save
    with open(f"{DATA_PATH}/cached_results/{MODEL_NAME}/{OUTPUT_FILE_NAME}.json", "w") as _file:
        json.dump(data, _file, ensure_ascii=False, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Completed!")
    print(f"Total items processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Success rate: {((processed_count - error_count) / processed_count * 100):.2f}%")
    print(f"Results saved to: {DATA_PATH}/cached_results/{OUTPUT_FILE_NAME}.json")
    print(f"{'='*60}")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main_async())