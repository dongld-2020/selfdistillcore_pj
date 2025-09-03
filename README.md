Federated Learning Framework with Self-DistillCore
This project is a customized Federated Learning (FL) framework designed to compare the performance of various FL algorithms, including FedAvg, FedProx, Scaffold, FedAdam, FedEMA, FedAvgM, and Self-DistillCore. The framework is highly configurable and supports multiple datasets and model architectures, making it suitable for research experiments.

Key Features
Multi-Algorithm Support: Compare the effectiveness of several popular FL algorithms.

Diverse Model Architectures: Compatible with models such as LeNet5, ResNet18Blood, VGG11Light, and ResNet32NoBatchNorm.

Non-IID Data Generation: Uses a Dirichlet distribution to create non-IID datasets, simulating real-world scenarios.

Automated Logging and Visualization: Automatically saves performance metrics like accuracy, loss, and communication costs to CSV files and generates comparison plots.

Directory Structure
.
├── src/
│   ├── __init__.py.py      # Marks the directory as a Python package
│   ├── client.py           # Client-side logic
│   ├── config.py           # Global configuration parameters
│   ├── model.py            # Model architectures (LeNet5, ResNet, etc.)
│   └── server.py           # Federated server logic
├── check_retention_factor.py # Script to check the retention factor for Self-DistillCore
├── plot_data_distibution.py  # Script for analyzing and visualizing data distribution
├── plot_figure.py          # Script for plotting results from CSV files
├── run.py                  # Main script to run experiments
└── plot figure arguments.docx # Example file for plot script arguments
Installation
Clone the repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install the required libraries:

Bash

pip install torch torchvision pandas numpy matplotlib scikit-learn medmnist
Usage
1. Running an Experiment
Use the run.py file to start the training process.

Bash

python run.py
You can edit the algorithm, dataset_name, and model_name variables directly in the run.py file to configure your experiment.

Results will be saved to a CSV file in the root directory, named with the format: results_{algorithm}_{params}.csv.

2. Visualizing Results
After an experiment is complete, you can use plot_figure.py to generate comparison plots.

Bash

python plot_figure.py
The script will prompt you to enter the names of the CSV files you want to compare, separated by a comma. For example:
results_selfdistillcore_clients50_rounds100_epochs5_alpha0.2_lr0.01_seed42_resnet18blood.csv, results_fedavg_clients50_rounds100_epochs5_alpha0.2_lr0.01_seed42_resnet18blood.csv

Plots comparing loss and accuracy will be generated and saved.

Configuration
All global parameters are defined in the src/config.py file. You can modify this file to customize your experiments.

General Parameters: GLOBAL_SEED, NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS.

Data Parameters: ALPHA (for Dirichlet distribution).

Model Parameters: LEARNING_RATE, BATCH_SIZE.

Network Parameters: SERVER_PORT.

Algorithm-specific Parameters:

SelfDistillCore: INITIAL_RETENTION, FINAL_RETENTION, GROWTH_RATE, K_PERCENT.

FedProx: MU.

FedAdam: SERVER_LR, BETA1, BETA2, TAU.

FedEMA: BETA_EMA, LAMBDA.

FedAvgM: BETA.

Data Analysis
You can run plot_data_distibution.py to analyze and visualize the non-IID data distribution across clients.

Bash

python plot_data_distibution.py
This script will print statistics such as the average number of samples and classes per client, as well as the Gini coefficient to measure the degree of non-IID.
