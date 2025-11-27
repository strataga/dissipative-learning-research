"""
CIFAR-10 Validation
===================

EXP-016: Validate findings on CIFAR-10 (harder benchmark)

Split CIFAR-10: 5 tasks of 2 classes each
- Task 1: airplane, automobile
- Task 2: bird, cat
- Task 3: deer, dog
- Task 4: frog, horse
- Task 5: ship, truck
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


def load_split_cifar10(batch_size=64):
    """Load CIFAR-10 split into 5 tasks"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 3072
    ])
    
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(data_dir, train=False, transform=transform)
    
    def get_indices(dataset, classes):
        return [i for i, (_, label) in enumerate(dataset) if label in classes]
    
    # 5 tasks of 2 classes each
    tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    # Classes: airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck
    
    loaders = {}
    for i, classes in enumerate(tasks):
        train_idx = get_indices(train, classes)
        test_idx = get_indices(test, classes)
        
        # Subsample for speed (use 2000 train, 500 test per task)
        train_idx = train_idx[:2000]
        test_idx = test_idx[:500]
        
        loaders[f'task{i+1}_train'] = DataLoader(
            Subset(train, train_idx), batch_size=batch_size, shuffle=True
        )
        loaders[f'task{i+1}_test'] = DataLoader(
            Subset(test, test_idx), batch_size=batch_size
        )
    
    return loaders, tasks


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


def run_cifar10_validation():
    print("=" * 70)
    print("CIFAR-10 VALIDATION (Split into 5 tasks)")
    print("=" * 70)
    
    loaders, tasks = load_split_cifar10()
    
    # CIFAR-10 is 32x32x3 = 3072 input dimensions
    layer_sizes = [3072, 256, 10]
    epochs_per_task = 5
    n_tasks = 5
    
    # Methods to test (reduced set for speed)
    methods = [
        ('Standard', False, 0, None, 1.0),
        ('EWC', False, 2000, None, 1.0),
        ('Sparse 5%', True, 0, 0.05, 1.0),
        ('Sparse + EWC', True, 2000, 0.05, 1.0),
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
                nn.Linear(3072, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ewc = EWC(model, ewc_lambda=ewc_lambda) if ewc_lambda > 0 else None
        
        task_accuracies = {f'task{i+1}': [] for i in range(n_tasks)}
        
        for task_idx in range(n_tasks):
            task_name = f'task{task_idx + 1}'
            print(f"  Training {task_name}...", end=" ", flush=True)
            
            for epoch in range(epochs_per_task):
                train_epoch(model, loaders[f'{task_name}_train'], optimizer,
                          ewc=ewc, use_tnn=use_tnn)
            
            if ewc is not None:
                ewc.consolidate(loaders[f'{task_name}_train'])
            
            # Evaluate all tasks
            accs = []
            for eval_idx in range(task_idx + 1):
                acc = evaluate(model, loaders[f'task{eval_idx+1}_test'])
                task_accuracies[f'task{eval_idx+1}'].append(acc)
                accs.append(acc)
            
            print(f"Avg: {np.mean(accs):.3f}")
        
        # Compute metrics
        forgetting = []
        final_accs = []
        for i in range(n_tasks - 1):
            accs = task_accuracies[f'task{i+1}']
            if len(accs) > 1:
                forgetting.append(max(accs) - accs[-1])
                final_accs.append(accs[-1])
        final_accs.append(task_accuracies[f'task{n_tasks}'][-1])
        
        avg_forgetting = np.mean(forgetting)
        avg_accuracy = np.mean(final_accs)
        
        results[name] = {
            'forgetting': avg_forgetting,
            'accuracy': avg_accuracy,
        }
        
        print(f"  Final - Forgetting: {avg_forgetting:.4f}, Accuracy: {avg_accuracy:.4f}")
    
    return results


def visualize_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = list(results.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    # Forgetting
    ax1 = axes[0]
    forgetting = [results[n]['forgetting'] for n in names]
    bars = ax1.bar(range(len(names)), forgetting, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.set_ylabel('Average Forgetting')
    ax1.set_title('CIFAR-10: Catastrophic Forgetting\n(lower is better)')
    for bar, val in zip(bars, forgetting):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10)
    
    # Accuracy
    ax2 = axes[1]
    accuracy = [results[n]['accuracy'] for n in names]
    bars = ax2.bar(range(len(names)), accuracy, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('Average Final Accuracy')
    ax2.set_title('CIFAR-10: Final Accuracy\n(higher is better)')
    for bar, val in zip(bars, accuracy):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'cifar10_validation.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nFigure saved to {save_path}")


def main():
    results = run_cifar10_validation()
    
    print("\n" + "=" * 70)
    print("CIFAR-10 RESULTS")
    print("=" * 70)
    
    print("\n{:<20} {:>15} {:>15}".format("Method", "Forgetting", "Accuracy"))
    print("-" * 50)
    
    for name in sorted(results.keys(), key=lambda n: results[n]['forgetting']):
        r = results[name]
        print("{:<20} {:>15.4f} {:>15.4f}".format(name, r['forgetting'], r['accuracy']))
    
    # Best method
    best = min(results.keys(), key=lambda n: results[n]['forgetting'])
    print(f"\n*** BEST: {best} ***")
    print(f"    Forgetting: {results[best]['forgetting']:.4f}")
    print(f"    Accuracy: {results[best]['accuracy']:.4f}")
    
    visualize_results(results)
    
    # Compare with MNIST results
    print("\n" + "=" * 70)
    print("COMPARISON: MNIST vs CIFAR-10")
    print("=" * 70)
    print("\nDoes sparse coding still help on harder benchmark?")
    
    mnist_sparse_ewc = 0.323
    cifar_sparse_ewc = results.get('Sparse + EWC', {}).get('forgetting', 1.0)
    
    print(f"  MNIST Sparse+EWC forgetting: {mnist_sparse_ewc:.3f}")
    print(f"  CIFAR Sparse+EWC forgetting: {cifar_sparse_ewc:.3f}")
    
    if cifar_sparse_ewc < 0.9:
        print("\n  CONCLUSION: Sparse coding generalizes to CIFAR-10!")
    else:
        print("\n  CONCLUSION: May need architecture tuning for CIFAR-10")
    
    return results


if __name__ == "__main__":
    main()
