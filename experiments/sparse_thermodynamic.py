"""
Sparse + Thermodynamic Combination
==================================

EXP-013: Test if thermodynamics helps WHEN COMBINED with sparsity

Previous findings:
- Sparsity alone: 0.39-0.61 forgetting (works!)
- Thermodynamic loss alone: 0.996 forgetting (doesn't work)

Hypothesis: Thermodynamic dynamics might help in the LOW-OVERLAP regime
created by sparse coding, even if they don't help in dense networks.
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

from thermodynamic_neural_network import ThermodynamicNeuralNetwork, evaluate_tnn


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


def train_sparse_thermo(model, loader, optimizer, entropy_weight=0.0, epochs=3):
    """Train TNN with optional entropy maximization"""
    entropy_history = []
    
    for epoch in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            output = model(x)
            
            # Base loss from TNN (includes sparsity)
            base_loss = model.compute_loss(output, y)
            
            # Compute gradients for entropy calculation
            base_loss.backward(retain_graph=True)
            
            # Compute entropy
            model._compute_entropy_production()
            entropy = model.state.entropy_production
            entropy_history.append(entropy)
            
            # Add entropy term if weight > 0
            if entropy_weight > 0:
                # Scale entropy for numerical stability
                scaled_entropy = entropy * 10000
                total_loss = base_loss - entropy_weight * scaled_entropy
            else:
                total_loss = base_loss
            
            # Full backward and step
            optimizer.zero_grad()
            total_loss.backward()
            
            # Apply TNN mechanisms
            for layer in model.layers:
                layer.update_consolidation()
                layer.apply_consolidation_mask()
            
            optimizer.step()
            
            for layer in model.layers:
                layer.update_homeostasis()
                layer.inject_noise(scale=0.001 * model.temperature)
    
    return entropy_history


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


def run_experiment():
    print("=" * 70)
    print("SPARSE + THERMODYNAMIC COMBINATION")
    print("=" * 70)
    
    layer_sizes = [784, 256, 10]
    epochs_per_task = 3
    n_tasks = 5
    
    loaders, tasks = load_split_mnist()
    
    # Test configurations: (name, sparsity, entropy_weight, temperature)
    configs = [
        ('Sparse 5% only', 0.05, 0.0, 1.0),
        ('Sparse 5% + Entropy (α=0.001)', 0.05, 0.001, 1.0),
        ('Sparse 5% + Entropy (α=0.01)', 0.05, 0.01, 1.0),
        ('Sparse 5% + High T', 0.05, 0.0, 2.0),
        ('Sparse 5% + Entropy + High T', 0.05, 0.01, 2.0),
        ('Sparse 1% only', 0.01, 0.0, 1.0),
        ('Sparse 1% + Entropy', 0.01, 0.01, 1.0),
    ]
    
    results = {}
    
    for name, sparsity, entropy_w, temp in configs:
        print(f"\n{'='*50}")
        print(f"Config: {name}")
        print(f"  Sparsity: {sparsity}, Entropy weight: {entropy_w}, Temp: {temp}")
        print(f"{'='*50}")
        
        model = ThermodynamicNeuralNetwork(
            layer_sizes, sparsity=sparsity, temperature=temp, 
            consolidation_rate=0.005
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        task_accuracies = {f'task{i+1}': [] for i in range(n_tasks)}
        all_entropy = []
        
        for task_idx in range(n_tasks):
            task_name = f'task{task_idx + 1}'
            
            entropy_hist = train_sparse_thermo(
                model, loaders[f'{task_name}_train'], optimizer,
                entropy_weight=entropy_w, epochs=epochs_per_task
            )
            all_entropy.extend(entropy_hist)
            
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
        avg_entropy = np.mean(all_entropy) if all_entropy else 0
        
        results[name] = {
            'forgetting': avg_forgetting,
            'accuracy': avg_accuracy,
            'entropy': avg_entropy,
            'task_accs': final_accs,
        }
        
        print(f"  Avg Forgetting: {avg_forgetting:.4f}")
        print(f"  Avg Accuracy: {avg_accuracy:.4f}")
        print(f"  Avg Entropy: {avg_entropy:.8f}")
    
    return results


def visualize_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    
    # 1. Forgetting
    ax1 = axes[0]
    forgetting = [results[n]['forgetting'] for n in names]
    bars = ax1.bar(range(len(names)), forgetting, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Average Forgetting')
    ax1.set_title('Sparse + Thermodynamic: Forgetting\n(lower is better)')
    for bar, val in zip(bars, forgetting):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=7)
    
    # 2. Accuracy
    ax2 = axes[1]
    accuracy = [results[n]['accuracy'] for n in names]
    bars = ax2.bar(range(len(names)), accuracy, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Average Accuracy')
    ax2.set_title('Sparse + Thermodynamic: Accuracy\n(higher is better)')
    for bar, val in zip(bars, accuracy):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=7)
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'sparse_thermodynamic.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")


def main():
    results = run_experiment()
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n{:<35} {:>12} {:>12}".format("Configuration", "Forgetting", "Accuracy"))
    print("-" * 60)
    
    sorted_names = sorted(results.keys(), key=lambda n: results[n]['forgetting'])
    
    for name in sorted_names:
        r = results[name]
        print("{:<35} {:>12.4f} {:>12.4f}".format(name, r['forgetting'], r['accuracy']))
    
    # Analysis
    baseline = 'Sparse 5% only'
    baseline_forg = results[baseline]['forgetting']
    
    print(f"\n*** ANALYSIS ***")
    print(f"Baseline (Sparse 5% only): {baseline_forg:.4f}")
    
    for name in sorted_names:
        if name != baseline and 'Entropy' in name or 'High T' in name:
            forg = results[name]['forgetting']
            diff = forg - baseline_forg
            print(f"  {name}: {forg:.4f} ({'+' if diff > 0 else ''}{diff:.4f})")
    
    visualize_results(results)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
