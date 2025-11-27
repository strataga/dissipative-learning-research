"""
Permuted MNIST Benchmark
========================

EXP-015: Standard continual learning benchmark

Permuted MNIST creates tasks by applying fixed random permutations to pixels.
This is harder than Split MNIST because all digits appear in each task.

Standard benchmark: 10 tasks with different permutations.
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
from torch.utils.data import DataLoader, TensorDataset

from thermodynamic_neural_network import ThermodynamicNeuralNetwork


class EWC:
    """Elastic Weight Consolidation"""
    
    def __init__(self, model: nn.Module, ewc_lambda: float = 2000):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.params = {}
        self.fisher = {}
        self.task_count = 0
    
    def compute_fisher(self, data_loader, num_samples: int = 200):
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        samples_used = 0
        for x, y in data_loader:
            if samples_used >= num_samples:
                break
            
            self.model.zero_grad()
            output = self.model(x)
            log_probs = F.log_softmax(output, dim=1)
            
            for i in range(min(x.size(0), num_samples - samples_used)):
                self.model.zero_grad()
                log_probs[i, y[i]].backward(retain_graph=True)
                
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.detach() ** 2
                samples_used += 1
        
        for n in fisher:
            fisher[n] /= max(samples_used, 1)
        return fisher
    
    def consolidate(self, data_loader):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.detach().clone()
        
        new_fisher = self.compute_fisher(data_loader)
        
        if self.task_count == 0:
            self.fisher = new_fisher
        else:
            for n in self.fisher:
                self.fisher[n] = (self.fisher[n] * self.task_count + new_fisher[n]) / (self.task_count + 1)
        
        self.task_count += 1
    
    def penalty(self):
        if self.task_count == 0:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for n, p in self.model.named_parameters():
            if n in self.fisher and n in self.params:
                loss = loss + (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.ewc_lambda * loss


def load_mnist():
    """Load full MNIST dataset"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, transform=transform)
    
    # Convert to tensors
    train_x = train.data.float().view(-1, 784) / 255.0
    train_y = train.targets
    test_x = test.data.float().view(-1, 784) / 255.0
    test_y = test.targets
    
    # Normalize
    train_x = (train_x - 0.1307) / 0.3081
    test_x = (test_x - 0.1307) / 0.3081
    
    return train_x, train_y, test_x, test_y


def create_permuted_tasks(train_x, train_y, test_x, test_y, n_tasks=10, seed=42):
    """Create permuted MNIST tasks"""
    np.random.seed(seed)
    
    tasks = []
    permutations = []
    
    for task_idx in range(n_tasks):
        if task_idx == 0:
            # First task: original MNIST (identity permutation)
            perm = np.arange(784)
        else:
            # Random permutation
            perm = np.random.permutation(784)
        
        permutations.append(perm)
        
        # Apply permutation
        train_x_perm = train_x[:, perm]
        test_x_perm = test_x[:, perm]
        
        train_loader = DataLoader(
            TensorDataset(train_x_perm, train_y),
            batch_size=64, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(test_x_perm, test_y),
            batch_size=64
        )
        
        tasks.append({
            'train': train_loader,
            'test': test_loader,
            'perm': perm
        })
    
    return tasks


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_epoch(model, loader, optimizer, ewc=None, use_tnn=False):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        
        if use_tnn:
            loss = model.compute_loss(out, y)
        else:
            loss = F.cross_entropy(out, y)
        
        if ewc is not None:
            loss = loss + ewc.penalty()
        
        loss.backward()
        
        if use_tnn:
            for layer in model.layers:
                layer.update_consolidation()
                layer.apply_consolidation_mask()
        
        optimizer.step()
        
        if use_tnn:
            for layer in model.layers:
                layer.update_homeostasis()


def run_permuted_mnist():
    print("=" * 70)
    print("PERMUTED MNIST BENCHMARK (10 tasks)")
    print("=" * 70)
    
    train_x, train_y, test_x, test_y = load_mnist()
    n_tasks = 5  # Reduced from 10 for memory
    tasks = create_permuted_tasks(train_x, train_y, test_x, test_y, n_tasks=n_tasks)
    
    layer_sizes = [784, 128, 10]  # Smaller for memory
    epochs_per_task = 2
    
    # Methods to compare
    methods = [
        ('Standard', False, 0, 0.0, 1.0),
        ('EWC only', False, 2000, 0.0, 1.0),
        ('Sparse 5%', True, 0, 0.05, 1.0),
        ('Sparse + EWC', True, 2000, 0.05, 1.0),
        ('Sparse + EWC + High T', True, 2000, 0.05, 2.0),
    ]
    
    results = {}
    
    for name, use_tnn, ewc_lambda, sparsity, temp in methods:
        print(f"\n{'='*50}")
        print(f"Method: {name}")
        print(f"{'='*50}")
        
        if use_tnn:
            model = ThermodynamicNeuralNetwork(
                layer_sizes, sparsity=sparsity, temperature=temp,
                consolidation_rate=0.005
            )
        else:
            model = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ewc = EWC(model, ewc_lambda=ewc_lambda) if ewc_lambda > 0 else None
        
        # Track accuracy on all tasks
        all_accuracies = {i: [] for i in range(n_tasks)}
        
        for task_idx in range(n_tasks):
            print(f"  Training Task {task_idx + 1}...", end=" ")
            
            # Train on current task
            for epoch in range(epochs_per_task):
                train_epoch(model, tasks[task_idx]['train'], optimizer, 
                          ewc=ewc, use_tnn=use_tnn)
            
            # Consolidate
            if ewc is not None:
                ewc.consolidate(tasks[task_idx]['train'])
            
            # Evaluate all tasks seen so far
            accs = []
            for eval_idx in range(task_idx + 1):
                acc = evaluate(model, tasks[eval_idx]['test'])
                all_accuracies[eval_idx].append(acc)
                accs.append(acc)
            
            avg_acc = np.mean(accs)
            print(f"Avg acc on tasks 1-{task_idx+1}: {avg_acc:.3f}")
        
        # Compute final metrics
        final_accuracies = [all_accuracies[i][-1] for i in range(n_tasks)]
        avg_final_acc = np.mean(final_accuracies)
        
        # Compute forgetting
        forgetting = []
        for i in range(n_tasks - 1):
            peak = max(all_accuracies[i])
            final = all_accuracies[i][-1]
            forgetting.append(peak - final)
        avg_forgetting = np.mean(forgetting)
        
        results[name] = {
            'final_accuracies': final_accuracies,
            'avg_final_acc': avg_final_acc,
            'avg_forgetting': avg_forgetting,
            'all_accuracies': all_accuracies,
        }
        
        print(f"  Final avg accuracy: {avg_final_acc:.4f}")
        print(f"  Avg forgetting: {avg_forgetting:.4f}")
    
    return results


def visualize_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    
    # 1. Final accuracy comparison
    ax1 = axes[0]
    accs = [results[n]['avg_final_acc'] for n in names]
    bars = ax1.bar(range(len(names)), accs, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Average Final Accuracy')
    ax1.set_title('Permuted MNIST: Final Accuracy\n(higher is better)')
    for bar, val in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=8)
    
    # 2. Forgetting comparison
    ax2 = axes[1]
    forg = [results[n]['avg_forgetting'] for n in names]
    bars = ax2.bar(range(len(names)), forg, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Average Forgetting')
    ax2.set_title('Permuted MNIST: Forgetting\n(lower is better)')
    for bar, val in zip(bars, forg):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=8)
    
    # 3. Task 1 accuracy over time
    ax3 = axes[2]
    for i, name in enumerate(names):
        task1_accs = results[name]['all_accuracies'][0]
        ax3.plot(range(1, len(task1_accs)+1), task1_accs, 'o-', 
                label=name, color=colors[i], linewidth=2)
    ax3.set_xlabel('Number of Tasks Trained')
    ax3.set_ylabel('Task 1 Accuracy')
    ax3.set_title('Task 1 Retention Over Time')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'permuted_mnist.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")


def main():
    results = run_permuted_mnist()
    
    print("\n" + "=" * 70)
    print("PERMUTED MNIST RESULTS")
    print("=" * 70)
    
    print("\n{:<25} {:>15} {:>15}".format("Method", "Final Accuracy", "Forgetting"))
    print("-" * 55)
    
    sorted_names = sorted(results.keys(), key=lambda n: -results[n]['avg_final_acc'])
    
    for name in sorted_names:
        r = results[name]
        print("{:<25} {:>15.4f} {:>15.4f}".format(
            name, r['avg_final_acc'], r['avg_forgetting']
        ))
    
    # Best method
    best_acc = max(results.keys(), key=lambda n: results[n]['avg_final_acc'])
    best_forg = min(results.keys(), key=lambda n: results[n]['avg_forgetting'])
    
    print(f"\n*** BEST ACCURACY: {best_acc} ***")
    print(f"    Accuracy: {results[best_acc]['avg_final_acc']:.4f}")
    
    print(f"\n*** LOWEST FORGETTING: {best_forg} ***")
    print(f"    Forgetting: {results[best_forg]['avg_forgetting']:.4f}")
    
    visualize_results(results)
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
