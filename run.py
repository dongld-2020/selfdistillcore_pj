#run.py
import sys
import os
import random
import multiprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import threading
import time
from src.model import LeNet5, ResNet20WithGroupNorm, VGG11Light, ResNet32WithGroupNorm
from src.server import start_server
from src.client import start_client
from src.utils import non_iid_partition_dirichlet, set_seed
from src.config import GLOBAL_SEED, NUM_ROUNDS, NUM_CLIENTS, DATA_DIR, BATCH_SIZE, DEVICE


def set_deterministic():
    """Thiết lập môi trường deterministic hoàn toàn"""

    set_seed(GLOBAL_SEED)
    

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

    torch.use_deterministic_algorithms(True, warn_only=True)
    

    np.random.seed(GLOBAL_SEED)
    

    random.seed(GLOBAL_SEED)
    

    def seed_worker(worker_id):
        worker_seed = GLOBAL_SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(worker_seed)
            torch.cuda.manual_seed_all(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)
    
    return seed_worker, g


def load_dataset(dataset_name):
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'bloodmnist':
        transform = transforms.Compose([
            transforms.RandomCrop(28, padding=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        try:
            import medmnist
            from medmnist import BloodMNIST
            train_split = BloodMNIST(split='train', root=DATA_DIR, download=True, transform=transform)
            test_split = BloodMNIST(split='test', root=DATA_DIR, download=True, transform=transform)
            train_dataset = ConcatDataset([train_split, test_split])
            train_labels = np.concatenate([train_split.labels, test_split.labels], axis=0)
            train_dataset.labels = train_labels
            val_dataset = None
            test_dataset = BloodMNIST(split='val', root=DATA_DIR, download=True, transform=transform_test)
        except ImportError:
            raise ImportError("Please install medmnist: pip install medmnist")
    elif dataset_name.lower() == 'organamnist':
        augment = True
        normalize_mean = [0.5]
        normalize_std = [0.5]
        download = True
        transform = transforms.Compose([
            transforms.RandomCrop(28, padding=1) if augment else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(10) if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        try:
            import medmnist
            from medmnist import OrganAMNIST
            train_dataset = OrganAMNIST(split='train', root=DATA_DIR, download=download, transform=transform)
            test_dataset = OrganAMNIST(split='test', root=DATA_DIR, download=download, transform=transform_test)
            train_labels = train_dataset.labels
            train_dataset.labels = train_labels
            val_dataset = None
        except ImportError:
            raise ImportError("Please install medmnist: pip install medmnist")
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=1),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(10),            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)
    else:
        raise ValueError("Dataset must be 'mnist', 'bloodmnist', 'organamnist', or 'cifar10'")
    return train_dataset, test_dataset

def run_server(global_model, selected_clients_list, algorithm, proportions, test_loader, global_control, model_name):
    print("Starting server...")
    global_control = start_server(global_model, selected_clients_list, algorithm=algorithm, proportions=proportions, test_loader=test_loader, global_control=global_control, model_name=model_name)
    print("Server finished.")
    return global_control

def run_clients(global_model, selected_clients, algorithm, client_datasets, global_control, model_name, global_seed):
    client_processes = []
    for client_id in selected_clients:
        print(f"Starting client {client_id}...")
        client_seed = global_seed + int(client_id)
        p = multiprocessing.Process(target=start_client, args=(client_id, client_datasets[client_id], global_model, algorithm, global_control, model_name, client_seed))
        client_processes.append(p)
        p.start()
    for t in client_processes:
        p.join()
    print("All clients for this round finished.")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    

    seed_worker, generator = set_deterministic()
    # Set start method to 'spawn' for CUDA compatibility
    multiprocessing.set_start_method('spawn')
    
    while True:
        algorithm = input("Enter the federated learning algorithm (fedavg, fedprox, scaffold, selfdistillcore, fedadam, fedavgm, fedema, fedzip): ").strip().lower()
        if algorithm in ['fedavg', 'fedprox', 'scaffold', 'selfdistillcore', 'fedadam', 'fedavgm', 'fedema', 'fedzip']:
            break
        print("Invalid input! Please enter 'fedavg', 'fedprox', 'scaffold', 'selfdistillcore', 'fedadam', 'fedavgm', 'fedema' or 'fedzip'.")
    
    while True:
        dataset_name = input("Enter the dataset (mnist, bloodmnist, organamnist, cifar10): ").strip().lower()
        if dataset_name in ['mnist', 'bloodmnist', 'organamnist', 'cifar10']:
            break
        print("Invalid input! Please enter 'mnist', 'bloodmnist', 'organamnist', or 'cifar10'.")
    
    if dataset_name.lower() == 'mnist':
        model_name = 'lenet5'
    elif dataset_name.lower() == 'bloodmnist':
        model_name = 'ResNet20WithGroupNorm'        
    elif dataset_name.lower() == 'organamnist':
        model_name = 'vgg11light'        
    elif dataset_name.lower() == 'cifar10':
        model_name = 'ResNet32WithGroupNorm'
    
    print(f"Running with algorithm: {algorithm}, dataset: {dataset_name}, model: {model_name}")
    

    set_deterministic()
    
    train_dataset, test_dataset = load_dataset(dataset_name)
    client_datasets, proportions = non_iid_partition_dirichlet(train_dataset, NUM_CLIENTS, partition="hetero")
    

    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=generator,
        num_workers=0  
    )
    
    if model_name == 'lenet5':
        global_model = LeNet5().to(DEVICE)
    elif model_name == 'ResNet20WithGroupNorm':
        global_model = ResNet20WithGroupNorm().to(DEVICE)        
    elif model_name == 'vgg11light':
        global_model = VGG11Light().to(DEVICE)        
    elif model_name == 'ResNet32WithGroupNorm':
        global_model = ResNet32WithGroupNorm().to(DEVICE)
    
    global_control = None
    selected_clients_list = []
    
    
    for round_num in range(NUM_ROUNDS):
    
        round_seed = GLOBAL_SEED + round_num
        random_state = np.random.RandomState(round_seed)
        selected_clients = random_state.choice(NUM_CLIENTS, random_state.randint(10, 16), replace=False)
        selected_clients_list.append(selected_clients)
    
    server_thread = threading.Thread(
        target=run_server, 
        args=(global_model, selected_clients_list, algorithm, proportions, test_loader, global_control, model_name)
    )
    server_thread.daemon = True
    server_thread.start()
    time.sleep(2)
    
    for round_num in range(NUM_ROUNDS):
        print(f"\nStarting clients for round {round_num+1}")
        run_clients(global_model, selected_clients_list[round_num], algorithm, client_datasets, global_control, model_name, GLOBAL_SEED)
        time.sleep(2)
    
    server_thread.join()
    print("Training completed.")
