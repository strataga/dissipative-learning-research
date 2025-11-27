"""
EWC (Elastic Weight Consolidation) Comparison
==============================================

Research Roadmap Item 1.2: Compare against EWC baseline

EWC protects important weights by adding a quadratic penalty based on
Fisher information, preventing them from changing too much on new tasks.

Reference: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
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
from dissipative_learning_machine import DissipativeLearningMachine, StandardNetwork


class EWC:
    """
    Elastic Weight Consolidation implementation.
    
    After training on a task, computes Fisher information matrix (diagonal approximation)
    and stores optimal parameters. On subsequent tasks, adds regularization term
    to prevent important weights from changing.
    """
    
    def __init__(self, model: nn.Module, ewc_lambda: float = 1000):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.params = {}  # {name: param_tensor}
        self.fisher = {}  # {name: fisher_tensor}
        self.task_count = 0
    
    def compute_fisher(self, data_loader, num_samples: int = 200):
        """Compute diagonal Fisher information matrix using sampled data"""
        self.model.eval()
        
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        samples_used = 0
        for x, y in data_loader:
            if samples_used >= num_samples:
                break
            
            self.model.zero_grad()
            output = self.model(x)
            
            # Use log-likelihood of correct class
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
        
        # Average
        for n in fisher:
            fisher[n] /= num_samples
        
        return fisher
    
    def consolidate(self, data_loader):
        """Store current params and compute Fisher after completing a task"""
        # Store current optimal parameters
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.detach().clone()
        
        # Compute Fisher information
        new_fisher = self.compute_fisher(data_loader)
        
        # Accumulate Fisher (for multiple tasks)
        if self.task_count == 0:
            self.fisher = new_fisher
        else:
            for n in self.fisher:
                self.fisher[n] = (self.fisher[n] * self.task_count + new_fisher[n]) / (self.task_count + 1)
        
        self.task_count += 1
    
    def penalty(self):
        """Compute EWC penalty term"""
        if self.task_count == 0:
            return 0.0
        
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher and n in self.params:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        
        return self.ewc_lambda * loss


def load_split_mnist(batch_size=64):
    """Load MNIST split by digits"""
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


def train_standard(model, loader, optimizer, epochs=3):
    """Train standard network"""
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()


def train_ewc(model, loader, optimizer, ewc, epochs=3):
    """Train with EWC regularization"""
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y) + ewc.penalty()
            loss.backward()
            optimizer.step()


def train_tnn(model, loader, optimizer, epochs=3):
    """Train TNN"""
    for _ in range(epochs):
        for x, y in loader:
            model.training_step(x, y, optimizer)


def evaluate(model, loader):
    """Evaluate accuracy"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def run_comparison():
    """Run full comparison across all methods"""
    
    layer_sizes = [784, 256, 10]
    epochs_per_task = 3
    n_tasks = 5
    
    loaders, tasks = load_split_mnist()
    
    print("=" * 70)
    print("EWC COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"\nArchitecture: {layer_sizes}")
    print(f"Tasks: {n_tasks} (Split MNIST)")
    print(f"Epochs per task: {epochs_per_task}")
    
    methods = {
        'Standard': {
            'model': StandardNetwork(layer_sizes),
            'train_fn': lambda m, l, o, e: train_standard(m, l, o, e),
            'ewc': None
        },
        'EWC (λ=100)': {
            'model': StandardNetwork(layer_sizes),
            'train_fn': None,  # Will use EWC training
            'ewc': None  # Will be created
        },
        'EWC (λ=1000)': {
            'model': StandardNetwork(layer_sizes),
            'train_fn': None,
            'ewc': None
        },
        'EWC (λ=5000)': {
            'model': StandardNetwork(layer_sizes),
            'train_fn': None,
            'ewc': None
        },
        'TNN (sparse)': {
            'model': ThermodynamicNeuralNetwork(layer_sizes, sparsity=0.05, temperature=1.0),
            'train_fn': lambda m, l, o, e: train_tnn(m, l, o, e),
            'ewc': None
        },
        'DLM': {
            'model': DissipativeLearningMachine(layer_sizes, temperature=0.5, dissipation_rate=0.02),
            'train_fn': lambda m, l, o, e: [
                (o.zero_grad(), 
                 F.cross_entropy(m(x), y).backward(), 
                 o.step(), 
                 m.step_dynamics()) 
                for _ in range(e) for x, y in l
            ][-1],
            'ewc': None
        },
    }
    
    # Create EWC instances
    for ewc_lambda in [100, 1000, 5000]:
        name = f'EWC (λ={ewc_lambda})'
        methods[name]['ewc'] = EWC(methods[name]['model'], ewc_lambda=ewc_lambda)
    
    results = {name: {'accuracies': {f'task{i+1}': [] for i in range(n_tasks)}} for name in methods}
    
    for name, config in methods.items():
        print(f"\n{'='*50}")
        print(f"Method: {name}")
        print(f"{'='*50}")
        
        model = config['model']
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ewc = config['ewc']
        
        for task_idx in range(n_tasks):
            task_name = f'task{task_idx + 1}'
            print(f"\n  Training on {task_name} (digits {tasks[task_idx]})...")
            
            # Train
            if ewc is not None:
                train_ewc(model, loaders[f'{task_name}_train'], optimizer, ewc, epochs_per_task)
                ewc.consolidate(loaders[f'{task_name}_train'])
            elif config['train_fn'] is not None:
                config['train_fn'](model, loaders[f'{task_name}_train'], optimizer, epochs_per_task)
            
            # Evaluate on all tasks seen so far
            print("  Accuracies:", end=" ")
            for eval_idx in range(task_idx + 1):
                eval_task = f'task{eval_idx + 1}'
                acc = evaluate(model, loaders[f'{eval_task}_test'])
                results[name]['accuracies'][eval_task].append(acc)
                print(f"T{eval_idx+1}={acc:.3f}", end=" ")
            print()
    
    return results, tasks


def compute_metrics(results, n_tasks=5):
    """Compute forgetting and other metrics"""
    metrics = {}
    
    for name, data in results.items():
        forgetting_sum = 0
        final_accs = []
        
        for task_idx in range(n_tasks - 1):  # All but last task can be forgotten
            task_name = f'task{task_idx + 1}'
            accs = data['accuracies'][task_name]
            if len(accs) > 1:
                peak = max(accs)
                final = accs[-1]
                forgetting_sum += (peak - final)
                final_accs.append(final)
        
        # Last task
        last_task = f'task{n_tasks}'
        final_accs.append(data['accuracies'][last_task][-1] if data['accuracies'][last_task] else 0)
        
        avg_forgetting = forgetting_sum / (n_tasks - 1)
        avg_accuracy = np.mean(final_accs)
        
        metrics[name] = {
            'avg_forgetting': avg_forgetting,
            'avg_final_accuracy': avg_accuracy,
            'task_accuracies': final_accs
        }
    
    return metrics


def visualize_results(results, metrics, tasks):
    """Create comprehensive visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    # 1. Average Forgetting Comparison
    ax1 = axes[0, 0]
    forgetting = [metrics[m]['avg_forgetting'] for m in methods]
    bars = ax1.bar(range(len(methods)), forgetting, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Average Forgetting')
    ax1.set_title('Catastrophic Forgetting Comparison\n(lower is better)')
    for bar, val in zip(bars, forgetting):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=8)
    
    # 2. Average Final Accuracy
    ax2 = axes[0, 1]
    accuracy = [metrics[m]['avg_final_accuracy'] for m in methods]
    bars = ax2.bar(range(len(methods)), accuracy, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Average Final Accuracy')
    ax2.set_title('Final Accuracy Across All Tasks\n(higher is better)')
    for bar, val in zip(bars, accuracy):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=8)
    
    # 3. Task 1 Retention Over Time
    ax3 = axes[1, 0]
    for i, name in enumerate(methods):
        accs = results[name]['accuracies']['task1']
        ax3.plot(range(1, len(accs)+1), accs, 'o-', label=name, color=colors[i], linewidth=2)
    ax3.set_xlabel('Tasks Trained')
    ax3.set_ylabel('Task 1 Accuracy')
    ax3.set_title('Task 1 Retention During Sequential Learning')
    ax3.legend(fontsize=8, loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.05, 1.05)
    
    # 4. Final Accuracy per Task
    ax4 = axes[1, 1]
    x = np.arange(5)
    width = 0.12
    for i, name in enumerate(methods):
        final_accs = metrics[name]['task_accuracies']
        ax4.bar(x + i*width, final_accs, width, label=name, color=colors[i], edgecolor='black')
    ax4.set_xticks(x + width * (len(methods)-1) / 2)
    ax4.set_xticklabels([f'Task {i+1}' for i in range(5)])
    ax4.set_ylabel('Final Accuracy')
    ax4.set_title('Final Accuracy per Task')
    ax4.legend(fontsize=7, loc='lower right')
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'ewc_comparison.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")


def main():
    results, tasks = run_comparison()
    metrics = compute_metrics(results)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n{:<20} {:>15} {:>15}".format("Method", "Avg Forgetting", "Avg Accuracy"))
    print("-" * 50)
    
    sorted_methods = sorted(metrics.keys(), key=lambda m: metrics[m]['avg_forgetting'])
    
    for name in sorted_methods:
        m = metrics[name]
        print("{:<20} {:>15.4f} {:>15.4f}".format(name, m['avg_forgetting'], m['avg_final_accuracy']))
    
    # Best method
    best = sorted_methods[0]
    print(f"\n*** BEST METHOD: {best} ***")
    print(f"    Forgetting: {metrics[best]['avg_forgetting']:.4f}")
    print(f"    Accuracy: {metrics[best]['avg_final_accuracy']:.4f}")
    
    # Compare TNN to EWC
    tnn_forg = metrics['TNN (sparse)']['avg_forgetting']
    ewc_best = min([metrics[m]['avg_forgetting'] for m in metrics if 'EWC' in m])
    
    print(f"\n*** TNN vs BEST EWC ***")
    print(f"    TNN forgetting: {tnn_forg:.4f}")
    print(f"    Best EWC forgetting: {ewc_best:.4f}")
    if tnn_forg < ewc_best:
        improvement = (ewc_best - tnn_forg) / ewc_best * 100
        print(f"    TNN is {improvement:.1f}% better than EWC")
    else:
        improvement = (tnn_forg - ewc_best) / tnn_forg * 100
        print(f"    EWC is {improvement:.1f}% better than TNN")
    
    visualize_results(results, metrics, tasks)
    
    print("\n" + "=" * 70)
    print("EWC COMPARISON COMPLETE")
    print("=" * 70)
    
    return results, metrics


if __name__ == "__main__":
    main()
