"""
Ultra-Low Sparsity + Sparse-EWC Combination
============================================

Tests:
1. Ultra-low sparsity (1%, 2%, 3%) - can we go lower?
2. Sparse+EWC combination - best of both worlds?

Based on finding: sparsity creates orthogonal representations
Hypothesis: Even lower sparsity = even less interference
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
from copy import deepcopy

from thermodynamic_neural_network import ThermodynamicNeuralNetwork, evaluate_tnn


class EWC:
    """Elastic Weight Consolidation"""
    
    def __init__(self, model: nn.Module, ewc_lambda: float = 1000):
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
            
            for i in range(x.size(0)):
                if samples_used >= num_samples:
                    break
                self.model.zero_grad()
                log_probs[i, y[i]].backward(retain_graph=True)
                
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.detach() ** 2
                samples_used += 1
        
        for n in fisher:
            fisher[n] /= num_samples
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
            return 0.0
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher and n in self.params:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.ewc_lambda * loss


def load_split_mnist(batch_size=64):
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, transform=transform)
    
    def get_indices(dataset, digits):
        return [i for i, (_, label) in enumerate(dataset) if label in digits]
    
    tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    
    loaders = {}
    for i, digits in enumerate(tasks):
        train_idx = get_indices(train, digits)
        test_idx = get_indices(test, digits)
        loaders[f'task{i+1}_train'] = DataLoader(Subset(train, train_idx), batch_size=batch_size, shuffle=True)
        loaders[f'task{i+1}_test'] = DataLoader(Subset(test, test_idx), batch_size=batch_size)
    
    return loaders, tasks


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train_tnn_epoch(model, loader, optimizer):
    for x, y in loader:
        model.training_step(x, y, optimizer)


def train_sparse_ewc_epoch(model, loader, optimizer, ewc):
    """Train with both sparse activation AND EWC regularization"""
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = model.compute_loss(out, y) + ewc.penalty()
        loss.backward()
        
        # Apply consolidation mask from TNN
        for layer in model.layers:
            layer.apply_consolidation_mask()
        
        optimizer.step()
        
        # TNN dynamics
        for layer in model.layers:
            layer.update_homeostasis()


def experiment_ultra_low_sparsity():
    """Test sparsity levels from 1% to 10%"""
    
    print("=" * 70)
    print("EXPERIMENT 1: ULTRA-LOW SPARSITY")
    print("=" * 70)
    
    sparsity_levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
    layer_sizes = [784, 256, 10]
    epochs_per_task = 3
    n_tasks = 5
    
    loaders, tasks = load_split_mnist()
    
    results = {'sparsity': [], 'avg_forgetting': [], 'avg_accuracy': [], 'task_accs': []}
    
    for sparsity in sparsity_levels:
        print(f"\n--- Sparsity = {sparsity*100:.0f}% ---")
        
        model = ThermodynamicNeuralNetwork(
            layer_sizes, sparsity=sparsity, temperature=1.0, consolidation_rate=0.005
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        task_accuracies = {f'task{i+1}': [] for i in range(n_tasks)}
        
        for task_idx in range(n_tasks):
            task_name = f'task{task_idx + 1}'
            
            for _ in range(epochs_per_task):
                train_tnn_epoch(model, loaders[f'{task_name}_train'], optimizer)
            
            # Evaluate all tasks
            for eval_idx in range(task_idx + 1):
                acc = evaluate(model, loaders[f'task{eval_idx+1}_test'])
                task_accuracies[f'task{eval_idx+1}'].append(acc)
        
        # Compute metrics
        forgetting_sum = 0
        final_accs = []
        for task_idx in range(n_tasks - 1):
            accs = task_accuracies[f'task{task_idx+1}']
            if len(accs) > 1:
                forgetting_sum += max(accs) - accs[-1]
                final_accs.append(accs[-1])
        final_accs.append(task_accuracies[f'task{n_tasks}'][-1])
        
        avg_forgetting = forgetting_sum / (n_tasks - 1)
        avg_accuracy = np.mean(final_accs)
        
        print(f"  Avg Forgetting: {avg_forgetting:.4f}")
        print(f"  Avg Accuracy: {avg_accuracy:.4f}")
        print(f"  Final Task Accs: {[f'{a:.2f}' for a in final_accs]}")
        
        results['sparsity'].append(sparsity)
        results['avg_forgetting'].append(avg_forgetting)
        results['avg_accuracy'].append(avg_accuracy)
        results['task_accs'].append(final_accs)
    
    return results


def experiment_sparse_ewc():
    """Test combination of sparse TNN + EWC"""
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SPARSE + EWC COMBINATION")
    print("=" * 70)
    
    layer_sizes = [784, 256, 10]
    epochs_per_task = 3
    n_tasks = 5
    best_sparsity = 0.05  # From previous experiments
    
    loaders, tasks = load_split_mnist()
    
    configurations = [
        ('TNN only (5%)', best_sparsity, None),
        ('Sparse+EWC (λ=100)', best_sparsity, 100),
        ('Sparse+EWC (λ=500)', best_sparsity, 500),
        ('Sparse+EWC (λ=1000)', best_sparsity, 1000),
        ('Sparse+EWC (λ=2000)', best_sparsity, 2000),
    ]
    
    results = {'config': [], 'avg_forgetting': [], 'avg_accuracy': [], 'task_accs': []}
    
    for name, sparsity, ewc_lambda in configurations:
        print(f"\n--- {name} ---")
        
        model = ThermodynamicNeuralNetwork(
            layer_sizes, sparsity=sparsity, temperature=1.0, consolidation_rate=0.005
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        ewc = EWC(model, ewc_lambda=ewc_lambda) if ewc_lambda else None
        
        task_accuracies = {f'task{i+1}': [] for i in range(n_tasks)}
        
        for task_idx in range(n_tasks):
            task_name = f'task{task_idx + 1}'
            
            for _ in range(epochs_per_task):
                if ewc:
                    train_sparse_ewc_epoch(model, loaders[f'{task_name}_train'], optimizer, ewc)
                else:
                    train_tnn_epoch(model, loaders[f'{task_name}_train'], optimizer)
            
            # Consolidate after each task
            if ewc:
                ewc.consolidate(loaders[f'{task_name}_train'])
            
            # Evaluate all tasks
            for eval_idx in range(task_idx + 1):
                acc = evaluate(model, loaders[f'task{eval_idx+1}_test'])
                task_accuracies[f'task{eval_idx+1}'].append(acc)
        
        # Compute metrics
        forgetting_sum = 0
        final_accs = []
        for task_idx in range(n_tasks - 1):
            accs = task_accuracies[f'task{task_idx+1}']
            if len(accs) > 1:
                forgetting_sum += max(accs) - accs[-1]
                final_accs.append(accs[-1])
        final_accs.append(task_accuracies[f'task{n_tasks}'][-1])
        
        avg_forgetting = forgetting_sum / (n_tasks - 1)
        avg_accuracy = np.mean(final_accs)
        
        print(f"  Avg Forgetting: {avg_forgetting:.4f}")
        print(f"  Avg Accuracy: {avg_accuracy:.4f}")
        print(f"  Final Task Accs: {[f'{a:.2f}' for a in final_accs]}")
        
        results['config'].append(name)
        results['avg_forgetting'].append(avg_forgetting)
        results['avg_accuracy'].append(avg_accuracy)
        results['task_accs'].append(final_accs)
    
    return results


def visualize_results(ultra_sparse_results, sparse_ewc_results):
    """Create visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Ultra-low sparsity: Forgetting vs Sparsity
    ax1 = axes[0, 0]
    ax1.plot(ultra_sparse_results['sparsity'], ultra_sparse_results['avg_forgetting'], 
             'o-', color='#e74c3c', linewidth=2, markersize=10)
    ax1.set_xlabel('Sparsity (fraction active)')
    ax1.set_ylabel('Average Forgetting')
    ax1.set_title('Ultra-Low Sparsity: Forgetting vs Sparsity\n(lower is better)')
    ax1.grid(True, alpha=0.3)
    
    # Mark best
    best_idx = np.argmin(ultra_sparse_results['avg_forgetting'])
    ax1.scatter([ultra_sparse_results['sparsity'][best_idx]], 
                [ultra_sparse_results['avg_forgetting'][best_idx]],
                color='green', s=200, zorder=5, marker='*',
                label=f"Best: {ultra_sparse_results['sparsity'][best_idx]*100:.0f}%")
    ax1.legend()
    
    # 2. Ultra-low sparsity: Accuracy vs Sparsity
    ax2 = axes[0, 1]
    ax2.plot(ultra_sparse_results['sparsity'], ultra_sparse_results['avg_accuracy'],
             'o-', color='#3498db', linewidth=2, markersize=10)
    ax2.set_xlabel('Sparsity (fraction active)')
    ax2.set_ylabel('Average Final Accuracy')
    ax2.set_title('Ultra-Low Sparsity: Accuracy vs Sparsity\n(higher is better)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Sparse+EWC: Forgetting comparison
    ax3 = axes[1, 0]
    configs = sparse_ewc_results['config']
    forgetting = sparse_ewc_results['avg_forgetting']
    colors = ['#3498db'] + ['#2ecc71'] * (len(configs) - 1)
    bars = ax3.bar(range(len(configs)), forgetting, color=colors, edgecolor='black')
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels(configs, rotation=45, ha='right')
    ax3.set_ylabel('Average Forgetting')
    ax3.set_title('Sparse + EWC Combination: Forgetting\n(lower is better)')
    for bar, val in zip(bars, forgetting):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)
    
    # 4. Sparse+EWC: Accuracy comparison
    ax4 = axes[1, 1]
    accuracy = sparse_ewc_results['avg_accuracy']
    bars = ax4.bar(range(len(configs)), accuracy, color=colors, edgecolor='black')
    ax4.set_xticks(range(len(configs)))
    ax4.set_xticklabels(configs, rotation=45, ha='right')
    ax4.set_ylabel('Average Final Accuracy')
    ax4.set_title('Sparse + EWC Combination: Accuracy\n(higher is better)')
    for bar, val in zip(bars, accuracy):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'ultra_sparse_ewc.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")


def main():
    # Run experiments
    ultra_sparse_results = experiment_ultra_low_sparsity()
    sparse_ewc_results = experiment_sparse_ewc()
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\n1. ULTRA-LOW SPARSITY RESULTS:")
    print("-" * 40)
    best_idx = np.argmin(ultra_sparse_results['avg_forgetting'])
    print(f"   Best sparsity: {ultra_sparse_results['sparsity'][best_idx]*100:.0f}%")
    print(f"   Forgetting: {ultra_sparse_results['avg_forgetting'][best_idx]:.4f}")
    print(f"   Accuracy: {ultra_sparse_results['avg_accuracy'][best_idx]:.4f}")
    
    print("\n2. SPARSE + EWC COMBINATION RESULTS:")
    print("-" * 40)
    best_idx = np.argmin(sparse_ewc_results['avg_forgetting'])
    print(f"   Best config: {sparse_ewc_results['config'][best_idx]}")
    print(f"   Forgetting: {sparse_ewc_results['avg_forgetting'][best_idx]:.4f}")
    print(f"   Accuracy: {sparse_ewc_results['avg_accuracy'][best_idx]:.4f}")
    
    # Compare TNN only vs best Sparse+EWC
    tnn_only_idx = 0
    best_combo_idx = np.argmin(sparse_ewc_results['avg_forgetting'][1:]) + 1
    
    tnn_forg = sparse_ewc_results['avg_forgetting'][tnn_only_idx]
    combo_forg = sparse_ewc_results['avg_forgetting'][best_combo_idx]
    
    print(f"\n3. IMPROVEMENT FROM COMBINING:")
    print("-" * 40)
    print(f"   TNN only forgetting: {tnn_forg:.4f}")
    print(f"   Best Sparse+EWC forgetting: {combo_forg:.4f}")
    
    if combo_forg < tnn_forg:
        improvement = (tnn_forg - combo_forg) / tnn_forg * 100
        print(f"   IMPROVEMENT: {improvement:.1f}% less forgetting")
    else:
        print(f"   No improvement from adding EWC")
    
    visualize_results(ultra_sparse_results, sparse_ewc_results)
    
    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)
    
    return ultra_sparse_results, sparse_ewc_results


if __name__ == "__main__":
    main()
