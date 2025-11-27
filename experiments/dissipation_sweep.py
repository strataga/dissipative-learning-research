"""
Dissipation Rate Parameter Sweep
================================

Research Roadmap Item 1.3: Hyperparameter Sensitivity Analysis
- Dissipation rate (γ): 0.001 -> 0.5

Tests how dissipation affects:
1. Learning stability
2. Catastrophic forgetting
3. Final accuracy
4. Thermodynamic state (entropy production)
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from dissipative_learning_machine import DissipativeLearningMachine, StandardNetwork


def load_split_mnist(batch_size=64):
    """Load MNIST split by digits for continual learning"""
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


def train_dlm_epoch(model, loader, optimizer):
    """Train DLM for one epoch"""
    model.train()
    total_entropy = 0
    count = 0
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        model.step_dynamics()
        total_entropy += model.state.entropy_production
        count += 1
    return total_entropy / max(count, 1)


def evaluate(model, loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def run_dissipation_sweep():
    """Sweep dissipation rate and measure effects"""
    
    dissipation_rates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    layer_sizes = [784, 256, 10]
    epochs_per_task = 3
    
    results = {
        'gamma': [],
        'task1_initial': [],
        'task1_final': [],
        'task2_final': [],
        'forgetting': [],
        'entropy_task1': [],
        'entropy_task2': [],
        'training_stable': [],
    }
    
    loaders, tasks = load_split_mnist()
    
    print("=" * 70)
    print("DISSIPATION RATE (γ) SWEEP EXPERIMENT")
    print("=" * 70)
    print(f"\nTesting γ: {dissipation_rates}")
    print(f"Architecture: {layer_sizes}")
    print(f"Epochs per task: {epochs_per_task}\n")
    
    for gamma in dissipation_rates:
        print(f"\n{'='*50}")
        print(f"Dissipation Rate γ = {gamma}")
        print(f"{'='*50}")
        
        try:
            model = DissipativeLearningMachine(
                layer_sizes=layer_sizes,
                temperature=0.5,
                dissipation_rate=gamma,
                energy_injection_rate=0.05
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Train on Task 1
            entropy_t1 = []
            for _ in range(epochs_per_task):
                ent = train_dlm_epoch(model, loaders['task1_train'], optimizer)
                entropy_t1.append(ent)
            
            task1_initial = evaluate(model, loaders['task1_test'])
            avg_entropy_t1 = np.mean(entropy_t1)
            print(f"  Task 1 accuracy: {task1_initial:.4f}")
            print(f"  Avg entropy production: {avg_entropy_t1:.4f}")
            
            # Train on Task 2
            entropy_t2 = []
            for _ in range(epochs_per_task):
                ent = train_dlm_epoch(model, loaders['task2_train'], optimizer)
                entropy_t2.append(ent)
            
            task1_final = evaluate(model, loaders['task1_test'])
            task2_final = evaluate(model, loaders['task2_test'])
            avg_entropy_t2 = np.mean(entropy_t2)
            
            forgetting = task1_initial - task1_final
            
            print(f"  Task 1 after Task 2: {task1_final:.4f}")
            print(f"  Task 2 accuracy: {task2_final:.4f}")
            print(f"  Forgetting: {forgetting:.4f}")
            print(f"  Avg entropy (Task 2): {avg_entropy_t2:.4f}")
            
            results['gamma'].append(gamma)
            results['task1_initial'].append(task1_initial)
            results['task1_final'].append(task1_final)
            results['task2_final'].append(task2_final)
            results['forgetting'].append(forgetting)
            results['entropy_task1'].append(avg_entropy_t1)
            results['entropy_task2'].append(avg_entropy_t2)
            results['training_stable'].append(True)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results['gamma'].append(gamma)
            results['task1_initial'].append(np.nan)
            results['task1_final'].append(np.nan)
            results['task2_final'].append(np.nan)
            results['forgetting'].append(np.nan)
            results['entropy_task1'].append(np.nan)
            results['entropy_task2'].append(np.nan)
            results['training_stable'].append(False)
    
    return results


def visualize_results(results):
    """Create visualization of dissipation sweep results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    gammas = results['gamma']
    
    # 1. Forgetting vs Dissipation
    ax1 = axes[0, 0]
    ax1.semilogx(gammas, results['forgetting'], 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax1.set_xlabel('Dissipation Rate (γ)')
    ax1.set_ylabel('Forgetting')
    ax1.set_title('Catastrophic Forgetting vs Dissipation\n(lower is better)')
    ax1.grid(True, alpha=0.3)
    
    # Mark optimal
    best_idx = np.nanargmin(results['forgetting'])
    ax1.scatter([gammas[best_idx]], [results['forgetting'][best_idx]], 
                color='green', s=150, zorder=5, label=f'Best: γ={gammas[best_idx]}')
    ax1.legend()
    
    # 2. Task Accuracies
    ax2 = axes[0, 1]
    ax2.semilogx(gammas, results['task1_final'], 'o-', label='Task 1 (retained)', linewidth=2)
    ax2.semilogx(gammas, results['task2_final'], 's-', label='Task 2 (new)', linewidth=2)
    ax2.semilogx(gammas, results['task1_initial'], '--', label='Task 1 (initial)', alpha=0.5)
    ax2.set_xlabel('Dissipation Rate (γ)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Task Accuracies vs Dissipation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Entropy Production
    ax3 = axes[1, 0]
    ax3.semilogx(gammas, results['entropy_task1'], 'o-', label='Task 1', linewidth=2)
    ax3.semilogx(gammas, results['entropy_task2'], 's-', label='Task 2', linewidth=2)
    ax3.set_xlabel('Dissipation Rate (γ)')
    ax3.set_ylabel('Entropy Production')
    ax3.set_title('Entropy Production vs Dissipation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Forgetting vs Entropy correlation
    ax4 = axes[1, 1]
    entropy_avg = [(e1 + e2) / 2 for e1, e2 in zip(results['entropy_task1'], results['entropy_task2'])]
    ax4.scatter(entropy_avg, results['forgetting'], c=gammas, cmap='viridis', s=100)
    ax4.set_xlabel('Average Entropy Production')
    ax4.set_ylabel('Forgetting')
    ax4.set_title('Forgetting vs Entropy Production\n(color = γ)')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('γ')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'dissipation_sweep.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")


def main():
    results = run_dissipation_sweep()
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n{:<10} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "γ", "Task1 Init", "Task1 Ret", "Task2 Acc", "Forgetting", "Entropy"
    ))
    print("-" * 70)
    
    for i, gamma in enumerate(results['gamma']):
        entropy_avg = (results['entropy_task1'][i] + results['entropy_task2'][i]) / 2
        print("{:<10.3f} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            gamma,
            results['task1_initial'][i],
            results['task1_final'][i],
            results['task2_final'][i],
            results['forgetting'][i],
            entropy_avg
        ))
    
    # Find optimal
    forgetting = np.array(results['forgetting'])
    valid_mask = ~np.isnan(forgetting)
    if valid_mask.any():
        best_idx = np.nanargmin(forgetting)
        best_gamma = results['gamma'][best_idx]
        
        print(f"\n*** OPTIMAL DISSIPATION RATE: γ = {best_gamma} ***")
        print(f"    Forgetting: {results['forgetting'][best_idx]:.4f}")
        print(f"    Task 1 retention: {results['task1_final'][best_idx]:.4f}")
        print(f"    Task 2 accuracy: {results['task2_final'][best_idx]:.4f}")
    
    # Analyze correlation between entropy and forgetting
    entropy_avg = [(e1 + e2) / 2 for e1, e2 in zip(results['entropy_task1'], results['entropy_task2'])]
    valid = [i for i in range(len(forgetting)) if not np.isnan(forgetting[i])]
    if len(valid) > 2:
        from scipy import stats
        corr, p_val = stats.pearsonr(
            [entropy_avg[i] for i in valid],
            [forgetting[i] for i in valid]
        )
        print(f"\n*** ENTROPY-FORGETTING CORRELATION ***")
        print(f"    Pearson r: {corr:.4f}")
        print(f"    p-value: {p_val:.4f}")
        if p_val < 0.05:
            if corr > 0:
                print("    Interpretation: Higher entropy → MORE forgetting (significant)")
            else:
                print("    Interpretation: Higher entropy → LESS forgetting (significant)")
        else:
            print("    Interpretation: No significant correlation")
    
    visualize_results(results)
    
    print("\n" + "=" * 70)
    print("DISSIPATION SWEEP COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
