"""
Triple Combination: Sparse + EWC + Thermodynamic
=================================================

EXP-014: Test if combining ALL THREE mechanisms achieves best results

Current best results:
- Sparse + EWC: 0.323 forgetting (68% reduction)
- Sparse + Entropy: 0.483 forgetting
- Sparse only: 0.389-0.678 forgetting

Hypothesis: Combining all three will achieve even better results.
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


def train_triple(model, loader, optimizer, ewc=None, entropy_weight=0.0, epochs=3):
    """Train with sparse (built-in) + EWC + entropy maximization"""
    
    for epoch in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            output = model(x)
            
            # Base TNN loss (includes sparsity regularization)
            base_loss = model.compute_loss(output, y)
            
            # Add EWC penalty if available
            if ewc is not None:
                ewc_penalty = ewc.penalty()
                base_loss = base_loss + ewc_penalty
            
            # Compute gradients for entropy calculation
            base_loss.backward(retain_graph=True)
            
            # Compute entropy and add to loss
            model._compute_entropy_production()
            entropy = model.state.entropy_production
            
            if entropy_weight > 0:
                # Scale entropy for numerical impact
                total_loss = base_loss - entropy_weight * entropy * 10000
            else:
                total_loss = base_loss
            
            # Full update
            optimizer.zero_grad()
            total_loss.backward()
            
            # TNN consolidation mechanisms
            for layer in model.layers:
                layer.update_consolidation()
                layer.apply_consolidation_mask()
            
            optimizer.step()
            
            # TNN homeostasis
            for layer in model.layers:
                layer.update_homeostasis()
                layer.inject_noise(scale=0.001 * model.temperature)


def run_experiment():
    print("=" * 70)
    print("TRIPLE COMBINATION: SPARSE + EWC + THERMODYNAMIC")
    print("=" * 70)
    
    layer_sizes = [784, 256, 10]
    epochs_per_task = 3
    n_tasks = 5
    
    loaders, tasks = load_split_mnist()
    
    # Configurations to test
    configs = [
        # (name, sparsity, ewc_lambda, entropy_weight, temperature)
        ('Baseline: Standard', None, 0, 0.0, 1.0),  # Standard network
        ('Sparse 5% only', 0.05, 0, 0.0, 1.0),
        ('Sparse + EWC', 0.05, 2000, 0.0, 1.0),
        ('Sparse + Thermo', 0.05, 0, 0.01, 1.0),
        ('Sparse + EWC + Thermo (α=0.001)', 0.05, 2000, 0.001, 1.0),
        ('Sparse + EWC + Thermo (α=0.01)', 0.05, 2000, 0.01, 1.0),
        ('Sparse + EWC + High T', 0.05, 2000, 0.0, 2.0),
        ('Full Triple (5%)', 0.05, 2000, 0.01, 2.0),
        ('Full Triple (1%)', 0.01, 2000, 0.01, 1.0),
    ]
    
    results = {}
    
    for name, sparsity, ewc_lambda, entropy_w, temp in configs:
        print(f"\n{'='*50}")
        print(f"Config: {name}")
        print(f"{'='*50}")
        
        # Create model
        if sparsity is None:
            # Standard MLP for baseline
            model = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
            use_tnn = False
        else:
            model = ThermodynamicNeuralNetwork(
                layer_sizes, sparsity=sparsity, temperature=temp,
                consolidation_rate=0.005
            )
            use_tnn = True
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ewc = EWC(model, ewc_lambda=ewc_lambda) if ewc_lambda > 0 else None
        
        task_accuracies = {f'task{i+1}': [] for i in range(n_tasks)}
        
        for task_idx in range(n_tasks):
            task_name = f'task{task_idx + 1}'
            
            if use_tnn:
                train_triple(model, loaders[f'{task_name}_train'], optimizer,
                           ewc=ewc, entropy_weight=entropy_w, epochs=epochs_per_task)
            else:
                # Standard training
                model.train()
                for _ in range(epochs_per_task):
                    for x, y in loaders[f'{task_name}_train']:
                        optimizer.zero_grad()
                        out = model(x)
                        loss = F.cross_entropy(out, y)
                        if ewc:
                            loss = loss + ewc.penalty()
                        loss.backward()
                        optimizer.step()
            
            # Consolidate after task
            if ewc:
                ewc.consolidate(loaders[f'{task_name}_train'])
            
            # Evaluate all tasks
            print(f"  After Task {task_idx+1}: ", end="")
            for eval_idx in range(task_idx + 1):
                acc = evaluate(model, loaders[f'task{eval_idx+1}_test'])
                task_accuracies[f'task{eval_idx+1}'].append(acc)
                print(f"T{eval_idx+1}={acc:.3f} ", end="")
            print()
        
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
        
        results[name] = {
            'forgetting': avg_forgetting,
            'accuracy': avg_accuracy,
            'task_accs': final_accs,
        }
        
        print(f"  Avg Forgetting: {avg_forgetting:.4f}")
        print(f"  Avg Accuracy: {avg_accuracy:.4f}")
    
    return results


def visualize_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    # 1. Forgetting comparison
    ax1 = axes[0]
    forgetting = [results[n]['forgetting'] for n in names]
    bars = ax1.barh(range(len(names)), forgetting, color=colors, edgecolor='black')
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Average Forgetting')
    ax1.set_title('Catastrophic Forgetting (lower is better)')
    ax1.invert_yaxis()
    
    for bar, val in zip(bars, forgetting):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)
    
    # 2. Accuracy comparison  
    ax2 = axes[1]
    accuracy = [results[n]['accuracy'] for n in names]
    bars = ax2.barh(range(len(names)), accuracy, color=colors, edgecolor='black')
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('Average Final Accuracy')
    ax2.set_title('Final Accuracy (higher is better)')
    ax2.invert_yaxis()
    
    for bar, val in zip(bars, accuracy):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'triple_combination.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")


def main():
    results = run_experiment()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS - SORTED BY FORGETTING")
    print("=" * 70)
    
    sorted_names = sorted(results.keys(), key=lambda n: results[n]['forgetting'])
    
    print("\n{:<40} {:>12} {:>12}".format("Configuration", "Forgetting", "Accuracy"))
    print("-" * 65)
    
    for name in sorted_names:
        r = results[name]
        print("{:<40} {:>12.4f} {:>12.4f}".format(name, r['forgetting'], r['accuracy']))
    
    # Analysis
    best = sorted_names[0]
    baseline_forg = results['Baseline: Standard']['forgetting']
    best_forg = results[best]['forgetting']
    
    print(f"\n*** BEST CONFIGURATION: {best} ***")
    print(f"    Forgetting: {best_forg:.4f}")
    print(f"    Accuracy: {results[best]['accuracy']:.4f}")
    print(f"    Reduction vs Standard: {(1 - best_forg/baseline_forg)*100:.1f}%")
    
    # Compare triple to double
    sparse_ewc_forg = results.get('Sparse + EWC', {}).get('forgetting', 1.0)
    triple_configs = [n for n in sorted_names if 'Triple' in n or ('EWC' in n and 'Thermo' in n)]
    
    if triple_configs:
        best_triple = min(triple_configs, key=lambda n: results[n]['forgetting'])
        triple_forg = results[best_triple]['forgetting']
        
        print(f"\n*** TRIPLE vs DOUBLE COMPARISON ***")
        print(f"    Sparse + EWC: {sparse_ewc_forg:.4f}")
        print(f"    Best Triple: {triple_forg:.4f} ({best_triple})")
        
        if triple_forg < sparse_ewc_forg:
            improvement = (sparse_ewc_forg - triple_forg) / sparse_ewc_forg * 100
            print(f"    Triple is {improvement:.1f}% BETTER")
        else:
            print(f"    Triple is NOT better than Sparse + EWC")
    
    visualize_results(results)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
