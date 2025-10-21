""" 
The Fine tuning is not included in this pipeline, please run the pipeline separatly
"""
# run_pipeline.py
import subprocess
import logging
import time
import sys
import os

# Attempt to import pynvml for GPU monitoring
try:
    import pynvml
except ImportError:
    pynvml = None
    print("WARNING: pynvml library not found. GPU usage will not be monitored.")
    print("         Install with: pip install nvidia-ml-py3")

# --- 1. Global Configuration ---
local_time = time.localtime()
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
LOG_FILE = f"pipeline_{timestamp}.log"

# --- 2. Logger Setup ---
def setup_logger():
    """Configures the logger to output to both a file and the console."""
    # Clear the log file at the beginning of a run
    with open(LOG_FILE, "w") as f:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logger initialized.")

# --- 3. GPU Monitoring Utility ---
def get_gpu_usage():
    """Fetches and formats GPU usage statistics if pynvml is available."""
    if not pynvml:
        return "N/A (pynvml not installed)"
    try:
        pynvml.nvmlInit()
        # Adjust the device index if you have multiple GPUs and want to monitor a specific one
        device_index = 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        usage_str = (
            f"GPU {device_index} | "
            f"Mem: {mem_info.used / (1024**2):.0f}/{mem_info.total / (1024**2):.0f} MB "
            f"({mem_info.used * 100 / mem_info.total:.1f}%) | "
            f"Util: {util_info.gpu}%"
        )
        return usage_str
    except Exception as e:
        return f"Error fetching GPU usage: {e}"
    finally:
        if pynvml:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

# --- 4. Script Execution Function ---
def run_sync_script(command, step_name):
    """Runs a synchronous script, logs its execution, and handles errors."""
    logging.info(f"--- Starting step: {step_name} ---")
    logging.info(f"Executing command: `{' '.join(command)}`")
    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.stdout:
            logging.info(f"[{step_name}] STDOUT:\n{result.stdout.strip()}")
        if result.stderr:
            logging.warning(f"[{step_name}] STDERR:\n{result.stderr.strip()}")

    except subprocess.CalledProcessError as e:
        logging.error(f"!!! FATAL ERROR during step: {step_name} !!!")
        logging.error(f"Return code: {e.returncode}")
        logging.error(f"[{step_name}] STDOUT:\n{e.stdout.strip()}")
        logging.error(f"[{step_name}] STDERR:\n{e.stderr.strip()}")
        raise RuntimeError(f"Step '{step_name}' failed.") from e
    except FileNotFoundError:
        logging.error(f"!!! FATAL ERROR: Command not found for step '{step_name}'. Is '{command[0]}' in your PATH?")
        raise

    duration = time.time() - start_time
    logging.info(f"GPU Status: {get_gpu_usage()}")
    logging.info(f"--- Finished step: {step_name} in {duration:.2f} seconds ---")

# --- 5. Main Pipeline Orchestrator ---
def main():
    """Main function to orchestrate the entire pipeline."""
    setup_logger()

    # =================================================================
    # --- PIPELINE CONFIGURATION ---
    # parameters for specific run.
    # =================================================================

    # VLLM settings 
    VLLM_CUDA_DEVICES = "2,3"
    VLLM_MODEL = "/home/psig/elmtex_prabin/FineTune/merged_model/lora/"
    VLLM_SERVED_MODEL_NAME = "Llama-3.1-8B-Instruct-lora" 
    VLLM_TENSOR_PARALLEL = "2" 

    # General Settings
    LANG = "en"
    MODEL_API = "http://localhost:8000/v1"

    # ICL Retriever Settings
    ICL_SOURCE_TRAIN_FILE = f"data/{LANG}/train.json"
    
    ICL_EXAMPLES_TO_RETRIEVE = 3

    # Evaluation Settings
    EVAL_SPACY_GPU = 2
    EVAL_BERTSCORE_GPU = 1
    # =================================================================

    # Create directories if they don't exist
    os.makedirs(f"data/{LANG}/cached_results", exist_ok=True)
    os.makedirs(f"data/{LANG}/retrieved_examples", exist_ok=True)
    os.makedirs(f"data/{LANG}/prompts", exist_ok=True)

    # Define the scripts and their commands. These will be run in order.
    # The server command now includes the parameters defined above.
    vllm_server_command = [
        "bash",
        "scripts/run_llm.sh",
        VLLM_CUDA_DEVICES,
        VLLM_MODEL,
        VLLM_SERVED_MODEL_NAME,
        VLLM_TENSOR_PARALLEL
    ]

    pipeline_steps = [
        ("Cache Naive Results", ["python", 
                                 "cache_results_naive.py",
                                 "--model_name",VLLM_SERVED_MODEL_NAME,
                                 ]),
        #("Retrieve ICL Examples", ["python", "icl_retriever_app.py"]),
        #("Create Advanced Prompts", ["python", "create_advance_prompt.py"]),
        ("Cache Advanced Prompts", ["python", 
                                    "cache_advance_prompt.py",
                                    "--model_name",VLLM_SERVED_MODEL_NAME]),
        ("Evaluate Results", ["python", "evaluate.py"]),
    ]

    logging.info("============================================")
    logging.info("  Starting Pipeline  ")
    logging.info("============================================")
    logging.info(f"Initial GPU Status: {get_gpu_usage()}")

    server_process = None

    try:
        # Step 1: Start the vLLM server as a background process
        logging.info("--- Starting step: Launch vLLM Server ---")
        logging.info(f"Executing command: `{' '.join(vllm_server_command)}`")
        server_process = subprocess.Popen(
            vllm_server_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout to capture all output
            text=True,
            encoding='utf-8'
        )
        logging.info(f"vLLM server process started with PID: {server_process.pid}.")

        # Wait for the server to initialize.
        wait_time = 45
        logging.info(f"Waiting {wait_time} seconds for the vLLM server to load the model...")
        time.sleep(wait_time)
        logging.info(f"Server should be ready. GPU Status: {get_gpu_usage()}")

        # Check if the server started correctly. If it exited, something went wrong.
        if server_process.poll() is not None:
             output, _ = server_process.communicate()
             logging.error("!!! vLLM server failed to start. It exited unexpectedly. !!!")
             logging.error(f"Server output:\n{output}")
             raise RuntimeError("vLLM server startup failed.")

        # Step 2: Run the synchronous pipeline steps
        for name, command in pipeline_steps:
            run_sync_script(command, name)

        logging.info("✅ Pipeline completed successfully!")

    except (Exception, KeyboardInterrupt) as e:
        logging.error(f"An error occurred during pipeline execution: {e}")
        logging.error("Pipeline execution HALTED.")

    finally:
        # Step 3: Ensure the vLLM server is terminated
        if server_process and server_process.poll() is None:
            logging.info(f"--- Tearing down: Shutting down vLLM server (PID: {server_process.pid}) ---")
            server_process.terminate()
            try:
                output, _ = server_process.communicate(timeout=20)
                logging.info("vLLM server terminated gracefully.")
                if output:
                    logging.info(f"Final server output:\n{output.strip()}")
            except subprocess.TimeoutExpired:
                logging.warning("Server did not terminate gracefully. Forcing shutdown (kill).")
                server_process.kill()
                logging.warning("Server process has been killed.")

        logging.info("============================================")
        logging.info("               Pipeline End               ")
        logging.info("============================================")

if __name__ == "__main__":
    main()