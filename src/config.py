import logging
import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")

# Tham số chung
GLOBAL_SEED = 42
NUM_CLIENTS = 50
NUM_ROUNDS = 200
#NUM_CLIENTS_PER_ROUND = 5
LOCAL_EPOCHS = 5

# Tham số dữ liệu
DATA_DIR = './data'
ALPHA = 0.2

#Bài sau kết hợp feedback loss và xác định adaptive lambda dựa trên average loss tăng hay giảm so với vòng trước, 
#nếu loss tăng thì giữ lại history nhiều, ngược lại nếu loss giảm thì freshupdate nhiều

# Tham số SelfDistillCore: Bắt đầu lơn la giu lai nhieu
INITIAL_RETENTION = 0.1  # Không dùng nếu cố định
FINAL_RETENTION = 0.5    # Không dùng nếu cố định
GROWTH_RATE = 0.01     # Không dùng nếu cố định
#retention_factor = initial_retention + (final_retention - initial_retention) *(1 - np.exp(-growth_rate * round_num))
#knowledge_bank[name] = retention_factor * knowledge_bank[name] + (1 - retention_factor) * sparse_delta_avg[name]
K_PERCENT = 0.20

# Tham số mô hình
LEARNING_RATE = 0.01
BATCH_SIZE = 32

# Tham số mạng
SERVER_PORT = 9999
BUFFER_SIZE = 4096



# Tham số FedProx
MU = 0.01

# Tham số FedAdam
SERVER_LR = 0.01  # Server learning rate (eta)
BETA1 = 0.9      # Momentum parameter
BETA2 = 0.99     # Second moment parameter
TAU = 1e-3       # Small constant for numerical stability

# Tham số FedAvgM
BETA = 0.9       # Momentum parameter for FedAvgM

# Tham số FedEMA
BETA_EMA = 0.5   # EMA decay factor (beta)
LAMBDA = 0.002   # Negative entropy regularization coefficient



# Tham số logging
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