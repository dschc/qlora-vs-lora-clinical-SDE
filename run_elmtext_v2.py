import subprocess
import time
import os
import json
import threading
import math
import sys # Import sys for stderr

def start_vllm_instance(vllm_port, gpu_ids_for_vllm, tensor_parallel_size):
    """Starts a vLLM server instance on a given port and specific GPUs."""
    gpu_ids_str = ",".join(map(str, gpu_ids_for_vllm))
    print(f"Starting vLLM on port {vllm_port} using GPUs: {gpu_ids_str} with tensor parallelism {tensor_parallel_size}...", file=sys.stderr)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str # Set CUDA_VISIBLE_DEVICES for this process
    
    # Redirect stdout/stderr of vLLM to files for later inspection
    vllm_stdout = open(f"vllm_stdout_{vllm_port}.log", "w")
    vllm_stderr = open(f"vllm_stderr_{vllm_port}.log", "w")
    
    process = subprocess.Popen(['./scripts/run_llm.sh', str(vllm_port), str(tensor_parallel_size)], 
                               env=env, stdout=vllm_stdout, stderr=vllm_stderr)
    print(f"vLLM process started with PID: {process.pid}", file=sys.stderr)
    return process, vllm_stdout, vllm_stderr

def start_icl_instance(icl_port, gpu_id_for_icl):
    """Starts an ICL retriever app instance on a given port and specific GPU."""
    print(f"Starting ICL retriever on port {icl_port} using GPU: {gpu_id_for_icl}...", file=sys.stderr)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id_for_icl) # Set CUDA_VISIBLE_DEVICES for this process
    
    # Redirect stdout/stderr of ICL to files for later inspection
    icl_stdout = open(f"icl_stdout_{icl_port}.log", "w")
    icl_stderr = open(f"icl_stderr_{icl_port}.log", "w")

    process = subprocess.Popen(['uvicorn', 'icl_retriever_app:app', '--host', '0.0.0.0', '--port', str(icl_port)], 
                               env=env, stdout=icl_stdout, stderr=icl_stderr)
    print(f"ICL process started with PID: {process.pid}", file=sys.stderr)
    return process, icl_stdout, icl_stderr

def run_cache_results(data_subset_path, output_file_path, icl_api, vllm_api, model_name, lang, worker_id):
    """Runs a single instance of the cache_results_advanced_icl.py script."""
    print(f"Worker {worker_id}: Running cache_results for {data_subset_path} with ICL:{icl_api}, vLLM:{vllm_api}...", file=sys.stderr)
    
    # Capture stdout/stderr for the cache_results script
    results_stdout = open(f"cache_results_worker_{worker_id}_stdout.log", "w")
    results_stderr = open(f"cache_results_worker_{worker_id}_stderr.log", "w")

    process = subprocess.Popen([
        'python', 'cache_results_advanced_icl.py',
        '--data_file', data_subset_path,
        '--output_file', output_file_path,
        '--icl_api_url', icl_api,
        '--vllm_api_url', vllm_api,
        '--model_name', model_name,
        '--lang', lang
    ], stdout=results_stdout, stderr=results_stderr)
    
    process.wait() # Wait for this specific process to finish
    
    results_stdout.close()
    results_stderr.close()

    if process.returncode != 0:
        print(f"Worker {worker_id}: cache_results script failed with exit code {process.returncode} for {data_subset_path}. Check cache_results_worker_{worker_id}_stderr.log for details.", file=sys.stderr)
    else:
        print(f"Worker {worker_id}: Finished processing {data_subset_path}. Output expected at {output_file_path}.", file=sys.stderr)

def split_data(input_file, num_chunks, output_dir):
    """Splits a JSON file into multiple smaller JSON files."""
    print(f"Loading data from {input_file} for splitting...", file=sys.stderr)
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input data file not found at {input_file}", file=sys.stderr)
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {input_file}: {e}", file=sys.stderr)
        return []
        
    print(f"Total records: {len(data)}", file=sys.stderr)

    if not data:
        print("Warning: Input data is empty. No chunks will be created.", file=sys.stderr)
        return []

    chunk_size = math.ceil(len(data) / num_chunks) # Use math.ceil for correct chunking
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    os.makedirs(output_dir, exist_ok=True)
    file_paths = []
    for i, chunk in enumerate(chunks):
        chunk_file = os.path.join(output_dir, f"test_chunk_{i}.json")
        try:
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, ensure_ascii=False, indent=4)
            file_paths.append(chunk_file)
        except IOError as e:
            print(f"Error: Could not write chunk file {chunk_file}: {e}", file=sys.stderr)
    print(f"Data split into {len(file_paths)} chunks, saved to {output_dir}.", file=sys.stderr)
    return file_paths

if __name__ == "__main__":
    # --- Configuration ---
    NUM_WORKERS = 3 # Changed to 3 for testing with 3 cases

    START_VLLM_PORT = 8000
    START_ICL_PORT = 8181
    MAIN_DATA_FILE = "./data/de/test_scripti.json"
    SUBSET_DATA_DIR = "./data/temp_subsets"
    OUTPUT_RESULTS_DIR = "./data/cached_results_parallel"

    # --- GPU Allocation Strategy for 4 GPUs ---
    DESIRED_VLLM_TENSOR_PARALLEL_SIZE = 1
    VLLM_GPU_IDS = [0]
    ICL_GPU_ID = 1   
    vllm_processes = []
    icl_processes = []
    cache_results_threads = []
    
    # File handles for stdout/stderr of services
    vllm_stdout_files = []
    vllm_stderr_files = []
    icl_stdout_files = []
    icl_stderr_files = []

    os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
    os.makedirs(SUBSET_DATA_DIR, exist_ok=True)

    # 1. Split the data
    print(f"Splitting data from {MAIN_DATA_FILE} into {NUM_WORKERS} chunks...", file=sys.stderr)
    data_subset_files = split_data(MAIN_DATA_FILE, NUM_WORKERS, SUBSET_DATA_DIR)
    if not data_subset_files:
        print("Exiting due to data splitting failure or empty data.", file=sys.stderr)
        sys.exit(1)
    print("Data splitting complete.", file=sys.stderr)

    # 2. Start vLLM and ICL instances
    print("Starting vLLM and ICL instances...", file=sys.stderr)
    
    # Start vLLM process on the designated GPUs, passing tensor_parallel_size
    vllm_proc, vllm_out, vllm_err = start_vllm_instance(START_VLLM_PORT, VLLM_GPU_IDS, DESIRED_VLLM_TENSOR_PARALLEL_SIZE)
    vllm_processes.append(vllm_proc)
    vllm_stdout_files.append(vllm_out)
    vllm_stderr_files.append(vllm_err)

    # Start ICL process on its designated GPU
    icl_proc, icl_out, icl_err = start_icl_instance(START_ICL_PORT, ICL_GPU_ID)
    icl_processes.append(icl_proc)
    icl_stdout_files.append(icl_out)
    icl_stderr_files.append(icl_err)

    # Give services time to start up
    print("Waiting for vLLM and ICL instances to start (90 seconds)...", file=sys.stderr)
    time.sleep(90)
    print("Services should be up.", file=sys.stderr)

    # 3. Run cache_results in parallel
    print("Launching cache_results processes...", file=sys.stderr)
    for i in range(NUM_WORKERS): 
        if i >= len(data_subset_files):
            print(f"Warning: Not enough data chunks for worker {i}. Skipping.", file=sys.stderr)
            continue
            
        data_subset_path = data_subset_files[i]
        output_file_path = os.path.join(OUTPUT_RESULTS_DIR, f"results_part_{i}.json")
        
       
        icl_api = f"http://0.0.0.0:{START_ICL_PORT}/retrieve" 
        vllm_api = f"http://localhost:{START_VLLM_PORT}/v1" 
     
        
        thread = threading.Thread(target=run_cache_results, 
                                  args=(data_subset_path, output_file_path, icl_api, vllm_api, "Llama-3.1-8B-Instruct", "de", i))
        cache_results_threads.append(thread)
        thread.start()

    # Wait for all cache_results threads to complete
    for t in cache_results_threads:
        t.join()
    print("All cache_results processes completed.", file=sys.stderr)

    # 4. Terminate vLLM and ICL processes
    print("Terminating vLLM and ICL instances...", file=sys.stderr)
    for p in vllm_processes:
        p.terminate()
        p.wait()
    for f in vllm_stdout_files + vllm_stderr_files:
        f.close()

    for p in icl_processes:
        p.terminate()
        p.wait()
    for f in icl_stdout_files + icl_stderr_files:
        f.close()
    print("All services terminated.", file=sys.stderr)

    # Merge results
    final_results = []
    print("Merging results from parallel runs...", file=sys.stderr)
    for i in range(NUM_WORKERS):
        part_file = os.path.join(OUTPUT_RESULTS_DIR, f"results_part_{i}.json")
        if os.path.exists(part_file):
            if os.path.getsize(part_file) == 0:
                print(f"Warning: Part file {part_file} is empty. Skipping.", file=sys.stderr)
                continue
            try:
                with open(part_file, 'r', encoding='utf-8') as f:
                    data_from_part = json.load(f)
                    final_results.extend(data_from_part)
                print(f"Successfully loaded {len(data_from_part)} records from {part_file}.", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {part_file}: {e}. Skipping this file. Check its content.", file=sys.stderr)
            except Exception as e:
                print(f"An unexpected error occurred while reading {part_file}: {e}", file=sys.stderr)
        else:
            print(f"Warning: Part file not found: {part_file}. This might indicate the cache_results script failed to produce output.", file=sys.stderr)
    
    final_output_file = os.path.join(OUTPUT_RESULTS_DIR, "merged_results.json")
    try:
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        print(f"All results merged into {final_output_file}. Total records: {len(final_results)}", file=sys.stderr)
    except IOError as e:
        print(f"Error: Could not write merged results to {final_output_file}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during merging and saving the final file: {e}", file=sys.stderr)

