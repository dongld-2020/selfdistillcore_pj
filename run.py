import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import threading
import time
from src.model import LeNet5, ResNet18Blood, VGG11Light, ResNet32NoBatchNorm
from src.server import start_server
from src.client import start_client
from src.utils import non_iid_partition_dirichlet
from src.config import GLOBAL_SEED, NUM_ROUNDS, NUM_CLIENTS, DATA_DIR, BATCH_SIZE
from src.config import DEVICE

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

def run_clients(global_model, selected_clients, algorithm, client_datasets, global_control, model_name):
    client_threads = []
    for client_id in selected_clients:
        print(f"Starting client {client_id}...")
        seed = GLOBAL_SEED + int(client_id)
        t = threading.Thread(target=start_client, args=(client_id, seed, client_datasets[client_id], global_model, algorithm, global_control, model_name))
        client_threads.append(t)
        t.start()
    for t in client_threads:
        t.join()
    print("All clients for this round finished.")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    while True:
        algorithm = input("Enter the federated learning algorithm (fedavg, fedprox, scaffold, selfdistillcore, fedadam, fedavgm, fedema): ").strip().lower()
        if algorithm in ['fedavg', 'fedprox', 'scaffold', 'selfdistillcore', 'fedadam', 'fedavgm', 'fedema']:
            break
        print("Invalid input! Please enter 'fedavg', 'fedprox', 'scaffold', 'selfdistillcore', 'fedadam', 'fedavgm', or 'fedema'.")
    while True:
        dataset_name = input("Enter the dataset (mnist, bloodmnist, organamnist, cifar10): ").strip().lower()
        if dataset_name in ['mnist', 'bloodmnist', 'organamnist', 'cifar10']:
            break
        print("Invalid input! Please enter 'mnist', 'bloodmnist', 'organamnist', or 'cifar10'.")
    if dataset_name.lower() == 'mnist':
        model_name = 'lenet5'
    elif dataset_name.lower() == 'bloodmnist':
        model_name = 'resnet18blood'        
    elif dataset_name.lower() == 'organamnist':
        model_name = 'vgg11light'        
    elif dataset_name.lower() == 'cifar10':
        model_name = 'resnet32nobatchnorm'
    print(f"Running with algorithm: {algorithm}, dataset: {dataset_name}, model: {model_name}")
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    train_dataset, test_dataset = load_dataset(dataset_name)
    client_datasets, proportions = non_iid_partition_dirichlet(train_dataset, NUM_CLIENTS, partition="hetero")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    if model_name == 'lenet5':
        global_model = LeNet5().to(DEVICE)
    elif model_name == 'resnet18blood':
        global_model = ResNet18Blood().to(DEVICE)        
    elif model_name == 'vgg11light':
        global_model = VGG11Light().to(DEVICE)        
    elif model_name == 'resnet32nobatchnorm':
        global_model = ResNet32NoBatchNorm().to(DEVICE)
    global_control = None
    selected_clients_list = []
    for round_num in range(NUM_ROUNDS):
        np.random.seed(GLOBAL_SEED + round_num)
        selected_clients = np.random.choice(NUM_CLIENTS, np.random.randint(10, 16), replace=False)
        selected_clients_list.append(selected_clients)
    server_thread = threading.Thread(target=run_server, args=(global_model, selected_clients_list, algorithm, proportions, test_loader, global_control, model_name))
    server_thread.daemon = True
    server_thread.start()
    time.sleep(2)
    for round_num in range(NUM_ROUNDS):
        print(f"\nStarting clients for round {round_num+1}")
        run_clients(global_model, selected_clients_list[round_num], algorithm, client_datasets, global_control, model_name)
        time.sleep(2)
    server_thread.join()
    print("Training completed.")