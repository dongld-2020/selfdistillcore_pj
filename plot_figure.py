import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_files):
    
    all_data = []
    algorithms = []
    training_params = None

    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"File {csv_file} not found!")
            continue
        df = pd.read_csv(csv_file)
    
        filename_parts = csv_file.split('_')
        algorithm = filename_parts[1]  
        algorithms.append(algorithm.capitalize())

        
        params = {
            'clients': filename_parts[2].replace('clients', ''),  # NUM_CLIENTS
            'rounds': filename_parts[3].replace('rounds', ''),  # NUM_ROUNDS
            'clientsPerRound': filename_parts[4].replace('clientsPerRound', ''),  # NUM_CLIENTS_PER_ROUND
            'epochs': filename_parts[5].replace('epochs', ''),  # LOCAL_EPOCHS
            'alpha': filename_parts[6].replace('alpha', ''),  # ALPHA
            'lr': filename_parts[7].replace('lr', ''),  # LEARNING_RATE
            'seed': filename_parts[8].replace('seed', '').replace('.csv', '')  # GLOBAL_SEED
        }

        
        if training_params is None:
            training_params = params
        else:
            if training_params != params:
                print(f"Warning: Training parameters in {csv_file} do not match previous files!")
                print(f"Expected: {training_params}")
                print(f"Got: {params}")

        all_data.append((algorithm, df))

    if not all_data:
        print("No valid CSV files to plot!")
        return

    
    params_str = (f"clients{training_params['clients']}_rounds{training_params['rounds']}_"
                 f"clientsPerRound{training_params['clientsPerRound']}_epochs{training_params['epochs']}_"
                 f"alpha{training_params['alpha']}_lr{training_params['lr']}_seed{training_params['seed']}")

    
    algorithms_str = ", ".join(algorithms)

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  

    
    for algorithm, df in all_data:
        if algorithm != 'selfdistillcore':            
            ax1.plot(df['Round'], df['Loss'], label=f'{algorithm.capitalize()} Loss', linestyle='--', linewidth=2)
        else:
            ax1.plot(df['Round'], df['Loss'], label=f'Selfdistillcore Loss', linewidth=3)
            
    ax1.set_title(f'Validation Loss Over Rounds Comparing', fontsize=20)
    ax1.set_xlabel('Communication Round', fontsize=20)
    ax1.set_ylabel('Validation Loss', fontsize=20)
    ax1.legend(fontsize=20)
    ax1.grid(True)
    ax1.tick_params(axis='both', labelsize=20)  

    
    for algorithm, df in all_data:
        if algorithm != 'selfdistillcore':
            ax2.plot(df['Round'], df['Accuracy'], label=f'{algorithm.capitalize()} Accuracy', linestyle='--', linewidth=2)
        else:
            ax2.plot(df['Round'], df['Accuracy'], label=f'Selfdistillcore Accuracy', linewidth=3)
    ax2.set_title(f'Accuracy Over Rounds Comparing', fontsize=20)
    ax2.set_xlabel('Communication Round', fontsize=20)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=20)
    ax2.legend(fontsize=20)
    ax2.grid(True)
    ax2.tick_params(axis='both', labelsize=20)  
    
    comparison_filename = f'comparison_{params_str}.png'
    plt.tight_layout()  
    plt.savefig(comparison_filename)
    plt.show()
    plt.close()
    print(f"Saved {comparison_filename}")

if __name__ == "__main__":
    
    print("Enter the CSV files to compare (e.g., results_fedavg_..., results_fedprox_...).")
    print("Separate multiple files with a comma (,). Press Enter to finish.")
    csv_input = input("CSV files: ").strip()
    
    if not csv_input:
        print("No files provided! Please run again and enter at least one CSV file.")
    else:
        csv_files = [f.strip() for f in csv_input.split(',')]
        plot_metrics(csv_files)

