import os
import subprocess
from tqdm import tqdm
import sys

def run_fine_tuning():
    # Set environment variables
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("\nStarting fine-tuning process...\n")
    
    # Define the command components
    command = [
        'mlx_lm.lora',
        '--model', 'mlx-community/Llama-3.2-1B-Instruct-4bit',
        '--train',
        '--data', '/Users/logan/DOE-ORACLE-LLM-1/ciobrain/fine_tuning/data',
        '--adapter-path', '/Users/logan/DOE-ORACLE-LLM-1/ciobrain/fine_tuning/adapters',
        '--iters', '300',
        '--batch-size', '2',
        '--grad-checkpoint',
        '--save-every', '50',
        '--num-layers', '4',
        '--learning-rate', '5e-4'
    ]
    
    # Create progress bar with green color
    pbar = tqdm(total=300, 
                desc="Training Progress",
                bar_format='\033[92m{l_bar}{bar:20}{r_bar}\033[0m',
                position=0,
                leave=True)
    
    # Initialize statistics
    total_loss = 0
    total_memory = 0
    loss_count = 0
    memory_count = 0
    current_iter = 0
    
    # Run the command
    try:
        print("Training command:", " ".join(command))
        print("\nTraining with parameters:")
        print(f"- Model: mlx-community/Llama-3.2-1B-Instruct-4bit")
        print(f"- Iterations: 300")
        print(f"- Batch size: 2")
        print(f"- Learning rate: 5e-4")
        print(f"- Adapter layers: 4")
        print(f"- Saving checkpoints every 50 iterations")
        print("\nStarting training process...\n")
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Update progress while process is running
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Update progress bar for any iteration
                if "Iter" in output:
                    try:
                        iter_num = int(output.split("Iter")[1].split(":")[0].strip())
                        if iter_num > current_iter:
                            pbar.update(iter_num - current_iter)
                            current_iter = iter_num
                            pbar.refresh()
                    except:
                        pass
                
                # Collect statistics
                if "Train loss" in output:
                    try:
                        loss = float(output.split("Train loss")[1].split(",")[0].strip())
                        total_loss += loss
                        loss_count += 1
                    except:
                        pass
                
                if "Peak mem" in output:
                    try:
                        memory = float(output.split("Peak mem")[1].split("GB")[0].strip())
                        total_memory += memory
                        memory_count += 1
                    except:
                        pass
                
                # Only print important messages
                if "Loading" in output or "Starting" in output or "Saved" in output:
                    print(output.strip())
        
        # Print final statistics
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            print(f"\nAverage Training Loss: {avg_loss:.3f}")
        if memory_count > 0:
            avg_memory = total_memory / memory_count
            print(f"Average Peak Memory: {avg_memory:.3f} GB")
        
        pbar.close()
        print("\nFine-tuning completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during fine-tuning: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        pbar.close()

if __name__ == "__main__":
    run_fine_tuning() 