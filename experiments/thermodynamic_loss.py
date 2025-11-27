"""
Thermodynamic Loss Functions
============================

EXP-012: Test if adding thermodynamic terms to the loss function helps

Hypothesis: Maximizing entropy production during training will:
1. Keep network in exploratory state
2. Prevent settling into task-specific minima
3. Reduce catastrophic forgetting

Loss variants:
1. Standard: L = CrossEntropy
2. Entropy Max: L = CrossEntropy - α × EntropyProduction
3. Energy Regularized: L = CrossEntropy + β × Energy
4. Full Thermodynamic: L = CrossEntropy - α × Entropy + β × Energy
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


class ThermodynamicMLP(nn.Module):
    """MLP with thermodynamic state tracking for loss computation"""
    
    def __init__(self, layer_sizes, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        # Activity tracking for information current
        self.layer_activities = []
        
    def forward(self, x):
        self.layer_activities = [x.abs().mean().item()]
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
            self.layer_activities.append(x.abs().mean().item())
        
        return x
    
    def compute_information_current(self):
        """Compute total information flow through network"""
        if len(self.layer_activities) < 2:
            return 0.0
        
        total_current = 0.0
        for i in range(len(self.layer_activities) - 1):
            current = abs(self.layer_activities[i+1] - self.layer_activities[i])
            total_current += current
        return total_current
    
    def compute_entropy_production(self):
        """Compute entropy production from gradients and currents"""
        total_force = 0.0
        for layer in self.layers:
            if layer.weight.grad is not None:
                total_force += layer.weight.grad.abs().mean().item()
        
        current = self.compute_information_current()
        entropy = current * total_force / (self.temperature + 1e-8)
        
        # Scale up for numerical stability
        return entropy * 1000  # Scale factor
    
    def compute_energy(self):
        """Compute total network energy (weight magnitude)"""
        energy = 0.0
        for layer in self.layers:
            energy += (layer.weight ** 2).sum()
        return energy


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


def train_with_loss(model, loader, optimizer, loss_type='standard', 
                    entropy_weight=0.1, energy_weight=0.001, epochs=3):
    """Train with different loss functions"""
    model.train()
    entropy_history = []
    
    for epoch in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            output = model(x)
            
            # Standard loss
            ce_loss = F.cross_entropy(output, y)
            
            # Compute gradients first for entropy calculation
            ce_loss.backward(retain_graph=True)
            
            if loss_type == 'standard':
                loss = ce_loss
                
            elif loss_type == 'entropy_max':
                # Maximize entropy production
                entropy = model.compute_entropy_production()
                loss = ce_loss - entropy_weight * entropy
                entropy_history.append(entropy)
                
            elif loss_type == 'energy_reg':
                # Regularize energy
                energy = model.compute_energy()
                loss = ce_loss + energy_weight * energy
                
            elif loss_type == 'full_thermo':
                # Full thermodynamic loss
                entropy = model.compute_entropy_production()
                energy = model.compute_energy()
                loss = ce_loss - entropy_weight * entropy + energy_weight * energy
                entropy_history.append(entropy)
            
            else:
                loss = ce_loss
            
            # Recompute gradients for actual loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return entropy_history


def run_experiment():
    """Compare different loss functions on continual learning"""
    
    print("=" * 70)
    print("THERMODYNAMIC LOSS FUNCTIONS EXPERIMENT")
    print("=" * 70)
    
    layer_sizes = [784, 256, 10]
    epochs_per_task = 3
    n_tasks = 5
    
    loaders, tasks = load_split_mnist()
    
    loss_configs = [
        ('Standard', 'standard', 0.0, 0.0),
        ('Entropy Max (α=0.01)', 'entropy_max', 0.01, 0.0),
        ('Entropy Max (α=0.1)', 'entropy_max', 0.1, 0.0),
        ('Entropy Max (α=1.0)', 'entropy_max', 1.0, 0.0),
        ('Energy Reg (β=0.001)', 'energy_reg', 0.0, 0.001),
        ('Full Thermo', 'full_thermo', 0.1, 0.001),
    ]
    
    results = {}
    
    for name, loss_type, entropy_w, energy_w in loss_configs:
        print(f"\n{'='*50}")
        print(f"Loss: {name}")
        print(f"{'='*50}")
        
        model = ThermodynamicMLP(layer_sizes, temperature=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        task_accuracies = {f'task{i+1}': [] for i in range(n_tasks)}
        all_entropy = []
        
        for task_idx in range(n_tasks):
            task_name = f'task{task_idx + 1}'
            
            entropy_hist = train_with_loss(
                model, loaders[f'{task_name}_train'], optimizer,
                loss_type=loss_type, entropy_weight=entropy_w, 
                energy_weight=energy_w, epochs=epochs_per_task
            )
            all_entropy.extend(entropy_hist)
            
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
        avg_entropy = np.mean(all_entropy) if all_entropy else 0
        
        results[name] = {
            'forgetting': avg_forgetting,
            'accuracy': avg_accuracy,
            'entropy': avg_entropy,
            'task_accs': final_accs,
        }
        
        print(f"  Avg Forgetting: {avg_forgetting:.4f}")
        print(f"  Avg Accuracy: {avg_accuracy:.4f}")
        print(f"  Avg Entropy: {avg_entropy:.6f}")
    
    return results


def visualize_results(results):
    """Visualize comparison"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    
    # 1. Forgetting
    ax1 = axes[0]
    forgetting = [results[n]['forgetting'] for n in names]
    bars = ax1.bar(range(len(names)), forgetting, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Average Forgetting')
    ax1.set_title('Forgetting (lower is better)')
    for bar, val in zip(bars, forgetting):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=8)
    
    # 2. Accuracy
    ax2 = axes[1]
    accuracy = [results[n]['accuracy'] for n in names]
    bars = ax2.bar(range(len(names)), accuracy, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Average Accuracy')
    ax2.set_title('Accuracy (higher is better)')
    for bar, val in zip(bars, accuracy):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=8)
    
    # 3. Entropy
    ax3 = axes[2]
    entropy = [results[n]['entropy'] for n in names]
    bars = ax3.bar(range(len(names)), entropy, color=colors, edgecolor='black')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.set_ylabel('Average Entropy Production')
    ax3.set_title('Entropy (thermodynamic activity)')
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'thermodynamic_loss.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")


def main():
    results = run_experiment()
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n{:<25} {:>12} {:>12} {:>12}".format(
        "Loss Function", "Forgetting", "Accuracy", "Entropy"
    ))
    print("-" * 65)
    
    sorted_names = sorted(results.keys(), key=lambda n: results[n]['forgetting'])
    
    for name in sorted_names:
        r = results[name]
        print("{:<25} {:>12.4f} {:>12.4f} {:>12.6f}".format(
            name, r['forgetting'], r['accuracy'], r['entropy']
        ))
    
    # Best result
    best = sorted_names[0]
    std_forg = results['Standard']['forgetting']
    best_forg = results[best]['forgetting']
    
    print(f"\n*** BEST: {best} ***")
    print(f"    Forgetting: {best_forg:.4f}")
    print(f"    vs Standard: {std_forg:.4f}")
    
    if best != 'Standard' and best_forg < std_forg:
        improvement = (std_forg - best_forg) / std_forg * 100
        print(f"    Improvement: {improvement:.1f}%")
    else:
        print("    No improvement from thermodynamic loss")
    
    visualize_results(results)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
