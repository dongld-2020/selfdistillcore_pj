# auto_run.py
# This script will run all algorithms in the paper for one dataset.
# Please carefullly set the parameter in the config.py file which is location in ./src
# The outcomes of the each algorithm for one dataset will be located in ./outcomes
# Please store the selfdistillcore file csv in other location or rename it before runing this script with deference K% to avoid rewrite

import subprocess
import sys
import time

def main():
    # Algorithms list to be run in order
    algorithms = ['fedavg', 'fedprox', 'scaffold', 'selfdistillcore', 
                 'fedadam', 'fedavgm', 'fedema', 'fedzip']
    
    valid_datasets = ['mnist', 'bloodmnist', 'organamnist', 'cifar10']
    
    # Input dataset, blank is MNIST
    while True:
        dataset = input(f"Input dataset {valid_datasets} (blank is mnist): ").strip().lower()
        if not dataset:
            dataset = "mnist"
        if dataset in valid_datasets:
            break
        print("Dataset is not valid, please input again.")
    
    print(f"\nStart running on: {dataset}")
    print(f"Will run all {len(algorithms)} algorithms: {', '.join(algorithms)}")
    print("=" * 60)
    
    # Run in order
    for i, algorithm in enumerate(algorithms, 1):
        print(f"\n[{i}/{len(algorithms)}] Running algorithm: {algorithm}")
        print("-" * 40)
        
        # Run command
        cmd = [sys.executable, "run.py"]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Transfer input in to simulation (algorithms and dataset)
            process.stdin.write(algorithm + "\n")
            process.stdin.write(dataset + "\n")
            process.stdin.flush()
            
            # Waiting for process ending
            process.wait()
            
            print(f"Finised algorithm: {algorithm}")
            
            # Relax 2 seconds
            if i < len(algorithms):
                print("Program is going to run next algorithm...")
                time.sleep(2)
            
        except Exception as e:
            print(f"Errors in {algorithm}: {e}")
            continue
        finally:
            if 'process' in locals() and process.poll() is None:
                process.terminate()
    
    print("\n" + "=" * 60)
    print("All algorithm has finished!")
    print(f"Total run {len(algorithms)} algorithms on dataset: {dataset}")

if __name__ == "__main__":
    main()
