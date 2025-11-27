"""
CIFAR-10 Experiment
===================

Scale up to a more challenging dataset.
Test if sparse coding benefits transfer to harder tasks.
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from thermodynamic_neural_network import ThermodynamicNeuralNetwork


class ConvSparseNet(nn.Module):
    """
    Convolutional network with sparse activations for CIFAR-10.
    Uses k-WTA in the fully connected layers.
    """
    
    def __init__(self, sparsity=0.05, num_classes=10):
        super().__init__()
        self.sparsity = sparsity
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected with sparse activation
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # For consolidation tracking
        self.register_buffer('consolidation', torch.zeros(256))
    
    def sparse_activation(self, x):
        """k-Winner-Take-All activation"""
        k = max(1, int(x.size(-1) * self.sparsity))
        topk_vals, topk_idx = torch.topk(x, k, dim=-1)
        
        if self.training:
            # Soft during training
            threshold = topk_vals[..., -1:]
            mask = torch.sigmoid((x - threshold) * 10)
            return x * mask
        else:
            # Hard during inference
            out = torch.zeros_like(x)
            out.scatter_(-1, topk_idx, topk_vals)
            return out
    
    def forward(self, x):
        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 64x8x8
        x = self.pool(F.relu(self.conv3(x)))  # 64x4x4
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Sparse FC
        x = F.relu(self.fc1(x))
        x = self.sparse_activation(x)
        x = self.fc2(x)
        
        return x


class StandardConvNet(nn.Module):
    """Standard ConvNet without sparse activation"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_split_cifar10(batch_size=64):
    """Load CIFAR-10 split into 5 tasks (2 classes each)"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    
    def get_indices(dataset, classes):
        return [i for i, (_, label) in enumerate(dataset) if label in classes]
    
    # 5 tasks: airplane/auto, bird/cat, deer/dog, frog/horse, ship/truck
    tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    task_names = [
        'airplane/automobile',
        'bird/cat', 
        'deer/dog',
        'frog/horse',
        'ship/truck'
    ]
    
    loaders = {}
    for i, classes in enumerate(tasks):
        train_idx = get_indices(train, classes)
        test_idx = get_indices(test, classes)
        loaders[f'task{i+1}_train'] = DataLoader(
            Subset(train, train_idx), batch_size=batch_size, shuffle=True
        )
        loaders[f'task{i+1}_test'] = DataLoader(
            Subset(test, test_idx), batch_size=batch_size
        )
    
    return loaders, tasks, task_names


def train_epoch(model, loader, optimizer):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        # Remap labels to 0,1 for binary classification
        y_binary = (y % 2).long()
        loss = F.cross_entropy(out[:, :2], y_binary)  # Only use 2 outputs
        loss.backward()
        optimizer.step()


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            y_binary = (y % 2).long()
            pred = out[:, :2].argmax(1)
            correct += (pred == y_binary).sum().item()
            total += y.size(0)
    return correct / total


def run_cifar_experiment():
    """Run continual learning experiment on CIFAR-10"""
    
    print("=" * 70)
    print("CIFAR-10 CONTINUAL LEARNING EXPERIMENT")
    print("=" * 70)
    
    epochs_per_task = 5
    n_tasks = 5
    
    loaders, tasks, task_names = load_split_cifar10()
    
    configurations = [
        ('Standard ConvNet', lambda: StandardConvNet(num_classes=2)),
        ('Sparse 5%', lambda: ConvSparseNet(sparsity=0.05, num_classes=2)),
        ('Sparse 2%', lambda: ConvSparseNet(sparsity=0.02, num_classes=2)),
        ('Sparse 1%', lambda: ConvSparseNet(sparsity=0.01, num_classes=2)),
    ]
    
    results = {name: {'accuracies': {f'task{i+1}': [] for i in range(n_tasks)}} 
               for name, _ in configurations}
    
    for name, create_model in configurations:
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"{'='*50}")
        
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for task_idx in range(n_tasks):
            task_name = f'task{task_idx + 1}'
            print(f"\n  Training on {task_name}: {task_names[task_idx]}")
            
            for epoch in range(epochs_per_task):
                train_epoch(model, loaders[f'{task_name}_train'], optimizer)
            
            # Evaluate all tasks
            print("  Accuracies: ", end="")
            for eval_idx in range(task_idx + 1):
                acc = evaluate(model, loaders[f'task{eval_idx+1}_test'])
                results[name]['accuracies'][f'task{eval_idx+1}'].append(acc)
                print(f"T{eval_idx+1}={acc:.3f} ", end="")
            print()
    
    return results, task_names


def compute_metrics(results, n_tasks=5):
    """Compute forgetting metrics"""
    metrics = {}
    
    for name, data in results.items():
        forgetting_sum = 0
        final_accs = []
        
        for task_idx in range(n_tasks - 1):
            accs = data['accuracies'][f'task{task_idx + 1}']
            if len(accs) > 1:
                peak = max(accs)
                final = accs[-1]
                forgetting_sum += (peak - final)
                final_accs.append(final)
        
        final_accs.append(data['accuracies'][f'task{n_tasks}'][-1])
        
        metrics[name] = {
            'avg_forgetting': forgetting_sum / (n_tasks - 1),
            'avg_accuracy': np.mean(final_accs),
            'final_accs': final_accs
        }
    
    return metrics


def visualize_results(results, metrics, task_names):
    """Visualize CIFAR-10 results"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = list(metrics.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    
    # 1. Forgetting comparison
    ax1 = axes[0]
    forgetting = [metrics[n]['avg_forgetting'] for n in names]
    bars = ax1.bar(range(len(names)), forgetting, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Average Forgetting')
    ax1.set_title('CIFAR-10: Catastrophic Forgetting\n(lower is better)')
    for bar, val in zip(bars, forgetting):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)
    
    # 2. Accuracy comparison
    ax2 = axes[1]
    accuracy = [metrics[n]['avg_accuracy'] for n in names]
    bars = ax2.bar(range(len(names)), accuracy, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Average Final Accuracy')
    ax2.set_title('CIFAR-10: Final Accuracy\n(higher is better)')
    for bar, val in zip(bars, accuracy):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)
    
    # 3. Task 1 retention over time
    ax3 = axes[2]
    for i, name in enumerate(names):
        accs = results[name]['accuracies']['task1']
        ax3.plot(range(1, len(accs)+1), accs, 'o-', label=name, color=colors[i], linewidth=2)
    ax3.set_xlabel('Tasks Trained')
    ax3.set_ylabel('Task 1 Accuracy')
    ax3.set_title('CIFAR-10: Task 1 Retention')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'cifar10_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")


def main():
    results, task_names = run_cifar_experiment()
    metrics = compute_metrics(results)
    
    print("\n" + "=" * 70)
    print("CIFAR-10 RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n{:<20} {:>15} {:>15}".format("Model", "Avg Forgetting", "Avg Accuracy"))
    print("-" * 50)
    
    sorted_names = sorted(metrics.keys(), key=lambda n: metrics[n]['avg_forgetting'])
    
    for name in sorted_names:
        m = metrics[name]
        print("{:<20} {:>15.4f} {:>15.4f}".format(name, m['avg_forgetting'], m['avg_accuracy']))
    
    # Best model
    best = sorted_names[0]
    std_forg = metrics['Standard ConvNet']['avg_forgetting']
    best_forg = metrics[best]['avg_forgetting']
    
    print(f"\n*** BEST MODEL: {best} ***")
    print(f"    Forgetting: {best_forg:.4f}")
    print(f"    Accuracy: {metrics[best]['avg_accuracy']:.4f}")
    
    if best != 'Standard ConvNet':
        improvement = (std_forg - best_forg) / std_forg * 100
        print(f"    Improvement over standard: {improvement:.1f}%")
    
    visualize_results(results, metrics, task_names)
    
    print("\n" + "=" * 70)
    print("CIFAR-10 EXPERIMENT COMPLETE")
    print("=" * 70)
    
    return results, metrics


if __name__ == "__main__":
    main()
