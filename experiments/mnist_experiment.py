"""
MNIST Experiment for Dissipative Learning Machine
==================================================

Next step: Validate DLM on a real dataset.

Usage:
    python experiments/mnist_experiment.py
"""

import sys
import os

# Get absolute paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dissipative_learning_machine import (
    DissipativeLearningMachine,
    StandardNetwork
)


def load_mnist(batch_size=64):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten
    ])
    
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, is_dlm=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in loader:
        optimizer.zero_grad()
        output = model(X)
        
        if is_dlm:
            loss = model.compute_loss(output, y, alpha=0.1)
        else:
            loss = F.cross_entropy(output, y)
        
        loss.backward()
        optimizer.step()
        
        if is_dlm:
            model.step_dynamics()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in loader:
            output = model(X)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total


def main():
    print("=" * 60)
    print("MNIST Experiment - Dissipative Learning Machine")
    print("=" * 60)
    
    # Hyperparameters
    layer_sizes = [784, 256, 128, 10]
    epochs = 10
    lr = 0.001
    
    print(f"\nArchitecture: {layer_sizes}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    
    # Load data
    print("\nLoading MNIST...")
    train_loader, test_loader = load_mnist(batch_size=64)
    
    # Create models
    dlm = DissipativeLearningMachine(
        layer_sizes=layer_sizes,
        temperature=0.5,
        dissipation_rate=0.02,
        energy_injection_rate=0.05
    )
    
    standard = StandardNetwork(layer_sizes=layer_sizes)
    
    dlm_optimizer = torch.optim.Adam(dlm.parameters(), lr=lr)
    std_optimizer = torch.optim.Adam(standard.parameters(), lr=lr)
    
    # Training
    dlm_accs = []
    std_accs = []
    
    print("\nTraining...")
    for epoch in range(epochs):
        # Train DLM
        dlm_loss, dlm_train_acc = train_epoch(dlm, train_loader, dlm_optimizer, is_dlm=True)
        dlm_test_acc = evaluate(dlm, test_loader)
        dlm_accs.append(dlm_test_acc)
        
        # Train Standard
        std_loss, std_train_acc = train_epoch(standard, train_loader, std_optimizer, is_dlm=False)
        std_test_acc = evaluate(standard, test_loader)
        std_accs.append(std_test_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"DLM={dlm_test_acc:.4f}, Std={std_test_acc:.4f}, "
              f"Entropy={dlm.state.entropy_production:.4f}")
    
    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"DLM Test Accuracy:      {dlm_accs[-1]:.4f}")
    print(f"Standard Test Accuracy: {std_accs[-1]:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(dlm_accs, label='DLM', marker='o')
    plt.plot(std_accs, label='Standard', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('MNIST: DLM vs Standard Network')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(PROJECT_ROOT, 'results', 'mnist_comparison.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nFigure saved to {save_path}")


if __name__ == "__main__":
    main()
