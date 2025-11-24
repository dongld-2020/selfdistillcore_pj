#config.py

import logging
import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



GLOBAL_SEED = 126
NUM_CLIENTS = 50
NUM_ROUNDS = 50
#NUM_CLIENTS_PER_ROUND = 5
LOCAL_EPOCHS = 3


DATA_DIR = './data'
ALPHA = 0.1



INITIAL_RETENTION = 0.1  
FINAL_RETENTION = 0.9    
GROWTH_RATE = 0.01    
#retention_factor = initial_retention + (final_retention - initial_retention) *(1 - np.exp(-growth_rate * round_num))
#knowledge_bank[name] = retention_factor * knowledge_bank[name] + (1 - retention_factor) * sparse_delta_avg[name]
K_PERCENT = 0.2


LEARNING_RATE = 0.01
BATCH_SIZE = 32


SERVER_PORT = 9999
BUFFER_SIZE = 4096




MU = 0.01


SERVER_LR = 0.01  # Server learning rate (eta)
BETA1 = 0.9      # Momentum parameter
BETA2 = 0.99     # Second moment parameter
TAU = 1e-3       # Small constant for numerical stability


BETA = 0.9       # Momentum parameter for FedAvgM


BETA_EMA = 0.5   # EMA decay factor (beta)
LAMBDA = 0.002   # Negative entropy regularization coefficient


FEDZIP_SPARSITY = 0.2  # Top-z sparsity (10% of weights kept)
FEDZIP_NUM_CLUSTERS = 3  # Number of clusters for k-means quantization
FEDZIP_ENCODING_METHOD = 'difference'  # Options: 'huffman', 'position', 'difference'


LOG_DIR = './logs'
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(LOG_DIR, log_file))
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    return logger