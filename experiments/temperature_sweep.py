"""
Temperature Parameter Sweep Experiment
======================================

Research Roadmap Item 1.3: Hyperparameter Sensitivity Analysis
- Temperature (T): 0.01 -> 10.0

Tests how temperature affects:
1. Final accuracy
2. Catastrophic forgetting
3. Noise spectrum slope
4. Training stability
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
from scipy import stats
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

from thermodynamic_neural_network import ThermodynamicNeuralNetwork, evaluate_tnn
from dissipative_learning_machine import DissipativeLearningMachine


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


def train_tnn_epoch(model, loader, optimizer):
    """Train TNN for one epoch"""
    for x, y in loader:
        model.training_step(x, y, optimizer)


def measure_noise_slope(model, num_samples=1500):
    """Measure power spectrum slope"""
    model.eval()
    activations = []
    x = torch.randn(1, 784)
    
    with torch.no_grad():
        for _ in range(num_samples):
            out = model(x)
            activations.append(out.mean().item())
            if hasattr(model, 'layers'):
                for layer in model.layers:
                    if hasattr(layer, 'inject_noise'):
                        layer.inject_noise(0.01)
            x = x + torch.randn_like(x) * 0.005
    
    activations = np.array(activations) - np.mean(activations)
    fft = np.fft.fft(activations)
    power = np.abs(fft) ** 2
    freqs = np.fft.fftfreq(len(activations))
    
    pos_mask = (freqs > 0.001) & (freqs < 0.4)
    log_freqs = np.log10(freqs[pos_mask])
    log_power = np.log10(power[pos_mask] + 1e-10)
    
    slope, _, r_value, _, _ = stats.linregress(log_freqs, log_power)
    return slope, r_value ** 2


def run_temperature_sweep():
    """Sweep temperature parameter and measure effects"""
    
    temperatures = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    layer_sizes = [784, 256, 10]
    epochs_per_task = 3
    
    results = {
        'temperature': [],
        'task1_final': [],
        'task2_final': [],
        'forgetting': [],
        'noise_slope': [],
        'noise_r2': [],
        'training_stable': [],
    }
    
    loaders, tasks = load_split_mnist()
    
    print("=" * 70)
    print("TEMPERATURE SWEEP EXPERIMENT")
    print("=" * 70)
    print(f"\nTesting temperatures: {temperatures}")
    print(f"Architecture: {layer_sizes}")
    print(f"Epochs per task: {epochs_per_task}\n")
    
    for temp in temperatures:
        print(f"\n{'='*50}")
        print(f"Temperature = {temp}")
        print(f"{'='*50}")
        
        try:
            model = ThermodynamicNeuralNetwork(
                layer_sizes,
                sparsity=0.15,
                temperature=temp,
                consolidation_rate=0.005,
                sleep_frequency=100
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Train on Task 1
            for _ in range(epochs_per_task):
                train_tnn_epoch(model, loaders['task1_train'], optimizer)
            task1_before = evaluate_tnn(model, loaders['task1_test'])
            print(f"  Task 1 accuracy: {task1_before:.4f}")
            
            # Train on Task 2
            for _ in range(epochs_per_task):
                train_tnn_epoch(model, loaders['task2_train'], optimizer)
            task1_after = evaluate_tnn(model, loaders['task1_test'])
            task2_acc = evaluate_tnn(model, loaders['task2_test'])
            
            forgetting = task1_before - task1_after
            print(f"  Task 1 after Task 2: {task1_after:.4f}")
            print(f"  Task 2 accuracy: {task2_acc:.4f}")
            print(f"  Forgetting: {forgetting:.4f}")
            
            # Measure noise spectrum
            slope, r2 = measure_noise_slope(model)
            print(f"  Noise slope: {slope:.3f} (R²={r2:.3f})")
            
            results['temperature'].append(temp)
            results['task1_final'].append(task1_after)
            results['task2_final'].append(task2_acc)
            results['forgetting'].append(forgetting)
            results['noise_slope'].append(slope)
            results['noise_r2'].append(r2)
            results['training_stable'].append(True)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results['temperature'].append(temp)
            results['task1_final'].append(np.nan)
            results['task2_final'].append(np.nan)
            results['forgetting'].append(np.nan)
            results['noise_slope'].append(np.nan)
            results['noise_r2'].append(np.nan)
            results['training_stable'].append(False)
    
    return results


def visualize_results(results):
    """Create visualization of temperature sweep results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    temps = results['temperature']
    
    # 1. Forgetting vs Temperature
    ax1 = axes[0, 0]
    ax1.semilogx(temps, results['forgetting'], 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Forgetting')
    ax1.set_title('Catastrophic Forgetting vs Temperature\n(lower is better)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=min(results['forgetting']), color='green', linestyle='--', alpha=0.5)
    
    # 2. Task Accuracies
    ax2 = axes[0, 1]
    ax2.semilogx(temps, results['task1_final'], 'o-', label='Task 1 (retained)', linewidth=2)
    ax2.semilogx(temps, results['task2_final'], 's-', label='Task 2 (new)', linewidth=2)
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Task Accuracies vs Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Noise Spectrum Slope
    ax3 = axes[1, 0]
    ax3.semilogx(temps, results['noise_slope'], 'o-', color='#9b59b6', linewidth=2, markersize=8)
    ax3.axhline(y=-1.0, color='green', linestyle='--', label='Ideal pink noise (slope=-1)', alpha=0.7)
    ax3.set_xlabel('Temperature')
    ax3.set_ylabel('Power Spectrum Slope')
    ax3.set_title('Noise Characteristics vs Temperature')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary heatmap
    ax4 = axes[1, 1]
    
    # Normalize metrics for heatmap
    forgetting_norm = np.array(results['forgetting'])
    forgetting_norm = 1 - (forgetting_norm - np.nanmin(forgetting_norm)) / (np.nanmax(forgetting_norm) - np.nanmin(forgetting_norm) + 1e-8)
    
    task1_norm = np.array(results['task1_final'])
    task1_norm = (task1_norm - np.nanmin(task1_norm)) / (np.nanmax(task1_norm) - np.nanmin(task1_norm) + 1e-8)
    
    noise_ideal = np.abs(np.array(results['noise_slope']) + 1.0)  # Distance from -1
    noise_norm = 1 - (noise_ideal - np.nanmin(noise_ideal)) / (np.nanmax(noise_ideal) - np.nanmin(noise_ideal) + 1e-8)
    
    # Combined score
    combined = (forgetting_norm + task1_norm + noise_norm) / 3
    
    ax4.bar(range(len(temps)), combined, color='#3498db', edgecolor='black')
    ax4.set_xticks(range(len(temps)))
    ax4.set_xticklabels([f'{t}' for t in temps], rotation=45)
    ax4.set_xlabel('Temperature')
    ax4.set_ylabel('Combined Score (higher is better)')
    ax4.set_title('Overall Performance Score')
    
    best_idx = np.nanargmax(combined)
    ax4.bar(best_idx, combined[best_idx], color='#2ecc71', edgecolor='black', label=f'Best: T={temps[best_idx]}')
    ax4.legend()
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'temperature_sweep.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")
    
    return temps[best_idx]


def main():
    results = run_temperature_sweep()
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n{:<12} {:>12} {:>12} {:>12} {:>12}".format(
        "Temp", "Task1 Ret", "Task2 Acc", "Forgetting", "Noise Slope"
    ))
    print("-" * 60)
    
    for i, temp in enumerate(results['temperature']):
        print("{:<12.2f} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.3f}".format(
            temp,
            results['task1_final'][i],
            results['task2_final'][i],
            results['forgetting'][i],
            results['noise_slope'][i]
        ))
    
    # Find optimal temperature
    forgetting = np.array(results['forgetting'])
    valid_mask = ~np.isnan(forgetting)
    if valid_mask.any():
        best_idx = np.nanargmin(forgetting)
        best_temp = results['temperature'][best_idx]
        print(f"\n*** OPTIMAL TEMPERATURE: {best_temp} ***")
        print(f"    Forgetting: {results['forgetting'][best_idx]:.4f}")
        print(f"    Task 1 retention: {results['task1_final'][best_idx]:.4f}")
        print(f"    Task 2 accuracy: {results['task2_final'][best_idx]:.4f}")
        print(f"    Noise slope: {results['noise_slope'][best_idx]:.3f}")
    
    best_temp = visualize_results(results)
    
    print("\n" + "=" * 70)
    print("TEMPERATURE SWEEP COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
