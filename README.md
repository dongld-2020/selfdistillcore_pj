## SelfDistillCore Federated Learning Project

[![GitHub license](https://img.shields.io/github/license/dongld-2020/selfdistillcore_pj)](https://github.com/dongld-2020/selfdistillcore_pj/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/dongld-2020/selfdistillcore_pj)](https://github.com/dongld-2020/selfdistillcore_pj/issues)
[![GitHub stars](https://img.shields.io/github/stars/dongld-2020/selfdistillcore_pj)](https://github.com/dongld-2020/selfdistillcore_pj/stargazers)

This project implements the **SelfDistillCore** algorithm for Federated Learning (FL), along with several baseline FL algorithms for comparison. It supports training on various datasets using PyTorch, focusing on non-IID data distributions. The project includes server-client architecture, model training, evaluation, and visualization tools.

## Key Features
- **Federated Learning Algorithms**:
  - SelfDistillCore (core contribution: knowledge distillation with retention factor for sparse updates)
  - FedAvg
  - FedProx
  - SCAFFOLD
  - FedAdam
  - FedAvgM
  - FedEMA
- **Supported Datasets**:
  - MNIST
  - BloodMNIST (from MedMNIST)
  - OrganAMNIST (from MedMNIST)
  - CIFAR-10
- **Models**:
  - LeNet5 (for MNIST)
  - ResNet18Blood (custom ResNet for BloodMNIST)
  - VGG11Light (for OrganAMNIST)
  - ResNet32NoBatchNorm (for CIFAR-10)
- **Non-IID Data Partitioning**: Uses Dirichlet distribution for heterogeneous data splits.
- **Visualization Tools**:
  - Plot data distribution across clients.
  - Plot training metrics (loss, accuracy) for comparison.
- **Configurable Parameters**: Learning rate, epochs, alpha (for Dirichlet), retention factors, etc.

## Project Structure
selfdistillcore_pj/
├── src/
│   ├── client.py          # Client-side training logic
│   ├── config.py          # Configuration parameters (seeds, epochs, etc.)
│   ├── model.py           # Model definitions (LeNet5, ResNet, etc.)
│   ├── server.py          # Server-side aggregation and evaluation
│   └── utils.py           # Utilities (data partitioning, evaluation)
├── check_retention_factor.py  # Script to check retention factor growth
├── plot_data_distribution.py  # Script to visualize client data distribution
├── plot_figure.py         # Script to plot metrics from CSV results
├── run.py                 # Main script to run the FL training
├── data/                  # Dataset storage (auto-downloaded)
├── logs/                  # Log files
├── results_*.csv          # Output CSV files with metrics
└── README.md              # This file
text## Requirements
- Python 3.8+
- PyTorch 1.12+
- Torchvision
- MedMNIST (for BloodMNIST and OrganAMNIST)
- NumPy, Pandas, Matplotlib, Scikit-learn

Install dependencies:
```bash
pip install torch torchvision medmnist numpy pandas matplotlib scikit-learn
Usage
1. Running the Federated Learning Training
Use run.py to start the training process. It will prompt for the algorithm and dataset.
bashpython run.py

Prompts:

Algorithm: fedavg, fedprox, scaffold, selfdistillcore, fedadam, fedavgm, fedema
Dataset: mnist, bloodmnist, organamnist, cifar10


Output:

Logs in ./logs/
Results CSV: results_<algorithm>_clients<NUM>_rounds<NUM>_epochs<NUM>_alpha<ALPHA>_lr<LR>_seed<SEED>_<model>.csv
Metrics include accuracy, loss, precision, recall, F1-score, per-class accuracy, confusion matrix, and communication cost.



2. Plotting Data Distribution
Visualize how data is distributed across clients (non-IID).
bashpython plot_data_distribution.py

Outputs statistics and plots for MNIST, BloodMNIST, OrganAMNIST, CIFAR-10.
Includes Gini coefficient for non-IID measure.

3. Plotting Metrics
Compare metrics (loss, accuracy) from multiple CSV result files.
bashpython plot_figure.py

Prompt: Enter CSV files (comma-separated, e.g., results_fedavg_...,results_selfdistillcore_...)
Outputs: PNG file with comparison plots.

4. Checking Retention Factor (for SelfDistillCore)
bashpython check_retention_factor.py

Plots retention factor growth over rounds.

Configuration
Edit src/config.py for hyperparameters:

GLOBAL_SEED: 42
NUM_CLIENTS: 50
NUM_ROUNDS: 200
LOCAL_EPOCHS: 5
ALPHA: 0.2 (Dirichlet parameter)
LEARNING_RATE: 0.01
K_PERCENT: 0.20 (sparsity for SelfDistillCore)
Algorithm-specific params (e.g., MU for FedProx, BETA_EMA for FedEMA)

Results and Evaluation

Training saves metrics per round in CSV.
Use plot_figure.py for visual comparisons.
Focus on non-IID scenarios; SelfDistillCore aims to improve convergence with sparse updates.

Citation
If you use this code, please cite:
text@misc{selfdistillcore_pj,
  author = {Dong Le Dinh},
  title = {SelfDistillCore Federated Learning Project},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dongld-2020/selfdistillcore_pj}}
}
Contributing
Contributions are welcome! Please open an issue or submit a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact

GitHub: dongld-2020
Email: dongld@ueh.edu.vn
