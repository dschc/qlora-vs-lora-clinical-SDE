# %% 
import json
import asyncio
from tqdm import tqdm
from openai import AsyncOpenAI
import os
import argparse
from datetime import date

# %%
# Arguments
parser = argparse.ArgumentParser(description='Cache Advance Prompt with Async Batch Processing')
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
parser.add_argument(
    '--batch_size',
    type=int,
    default=16,
    help='Batch size for concurrent processing (default: 16)'
)
parser.add_argument(
    '--max_concurrent',
    type=int,
    default=8,
    help='Maximum concurrent requests (default: 8)'
)

args = parser.parse_args()

MODEL_NAME = args.model_name
MODEL_API = args.model_api
DATA_PATH = args.output_dir
PROMPT_PATH = args.prompt_file
BATCH_SIZE = args.batch_size
MAX_CONCURRENT = args.max_concurrent

# Prepare async vllm client
vllm_client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=MODEL_API
)
datestamp = date.today().strftime("%Y%m%d")

LANG = "en"
OUTPUT_FILE_NAME = f"test_{MODEL_NAME}_advanced_icl_{datestamp}"

# %% 
# Read data
data = json.load(open(PROMPT_PATH))

# %%
async def process_single_item(client, item, semaphore):
    """Process a single item with semaphore for concurrency control"""
    async with semaphore:
        clinical_report = item['report']
        system_prompt = item["system_prompt"]
        
        try:
            model_response = await client.chat.completions.create(
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
            
            item['model_prediction'] = response_json
            return item, None
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            item['model_prediction'] = None
            return item, error_msg

async def process_batch_async(client, batch_data, max_concurrent):
    """Process a batch of items concurrently"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = [process_single_item(client, item, semaphore) for item in batch_data]
    results = await asyncio.gather(*tasks)
    
    return results

async def main_async():
    """Main async function to process all data"""
    print(f"Processing {len(data)} items in batches of {BATCH_SIZE} with {MAX_CONCURRENT} concurrent requests...")
    
    processed_count = 0
    error_count = 0
    
    # Process data in batches
    for batch_idx in tqdm(range(0, len(data), BATCH_SIZE), desc="Processing batches"):
        batch_end = min(batch_idx + BATCH_SIZE, len(data))
        batch = data[batch_idx:batch_end]
        
        # Process the batch asynchronously
        results = await process_batch_async(vllm_client, batch, MAX_CONCURRENT)
        
        # Update the original data with results
        for i, (processed_item, error) in enumerate(results):
            data[batch_idx + i] = processed_item
            processed_count += 1
            if error:
                error_count += 1
                print(f"\nError processing item {batch_idx + i}: {error}")
        
        # Create output directory if it doesn't exist
        os.makedirs(f"{DATA_PATH}/cached_results/", exist_ok=True)
        
        # Save progress periodically (every 50 items or at the end)
        if (batch_end) % 50 == 0 or batch_end == len(data):
            with open(f"{DATA_PATH}/cached_results/{OUTPUT_FILE_NAME}.json", "w") as _file:
                json.dump(data, _file, ensure_ascii=False, indent=4)
            print(f"\nProgress saved: {batch_end}/{len(data)} items completed ({error_count} errors)")
    
    # Final save
    with open(f"{DATA_PATH}/cached_results/{OUTPUT_FILE_NAME}.json", "w") as _file:
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