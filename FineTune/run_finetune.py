# Run Fine tune
import subprocess 
import datetime
import os 

# Configurations

script_to_run = 'finetune_script.py'
r = [8, 16, 32, 64, 128]
alpha = [16, 32, 64, 128, 256]

for i in r:
    for j in alpha:
        output_logs = f'logs/finetune_ouput_r{i}a{j}.log'
        error_logs = f'logs/finetune_errors_r{i}a{j}.log'
        output_dir = f'./model/experiments/4bit_r{i}a{j}'

        try: 
            result = subprocess.run(
                ['python', script_to_run, 
                '--experiment','4bit',
                '--num_samples','10000',
                '--output_base',output_dir,
                '--lora_r','8',
                '--lora_alpha','64']
                capture_output=True,
                text=True,
                check=True
            )

            with open(output_logs,'w') as f:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"-------log generated at {timestamp}--------\n")
                f.write(result.stdout)
            if result.stderr:
                with open(error_logs,'w') as f:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"-------error log generated at {timestamp}--------\n")
                    f.write(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"finetunbe.py with r={i} and alpha={j} failed with error: {e}")
            with open(output_logs,'w') as f:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"-------log generated at {timestamp}--------\n")
                f.write(e.stdout)
            with open(error_logs,'w') as f:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"-------error log generated at {timestamp}--------\n")
                f.write(e.stderr)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")