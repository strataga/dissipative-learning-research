"""
Theory Validation Script for Dissipative Learning Machine
==========================================================

Tests the three core claims:
1. Reduced catastrophic forgetting (2.4x better retention)
2. Pink noise (1/f) spectrum in activations
3. Comparable accuracy to standard networks

Also fixes the entropy production calculation bug.
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

from dissipative_learning_machine import (
    DissipativeLearningMachine,
    StandardNetwork
)


def load_mnist_split(digits_task1=(0,1,2,3,4), digits_task2=(5,6,7,8,9), batch_size=64):
    """Load MNIST split into two tasks by digit groups"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    # Split by digit
    def get_indices(dataset, digits):
        return [i for i, (_, label) in enumerate(dataset) if label in digits]
    
    task1_train_idx = get_indices(train_dataset, digits_task1)
    task1_test_idx = get_indices(test_dataset, digits_task1)
    task2_train_idx = get_indices(train_dataset, digits_task2)
    task2_test_idx = get_indices(test_dataset, digits_task2)
    
    return {
        'task1_train': DataLoader(Subset(train_dataset, task1_train_idx), batch_size=batch_size, shuffle=True),
        'task1_test': DataLoader(Subset(test_dataset, task1_test_idx), batch_size=batch_size),
        'task2_train': DataLoader(Subset(train_dataset, task2_train_idx), batch_size=batch_size, shuffle=True),
        'task2_test': DataLoader(Subset(test_dataset, task2_test_idx), batch_size=batch_size),
    }


def train_epoch(model, loader, optimizer, is_dlm=False):
    """Train for one epoch, tracking entropy production correctly"""
    model.train()
    total_entropy = 0.0
    n_batches = 0
    
    for X, y in loader:
        optimizer.zero_grad()
        output = model(X)
        
        if is_dlm:
            # Standard loss only for backward pass
            loss = F.cross_entropy(output, y)
        else:
            loss = F.cross_entropy(output, y)
        
        loss.backward()
        
        # Compute entropy production AFTER backward (when gradients exist)
        if is_dlm:
            entropy = model.compute_total_entropy_production()
            total_entropy += entropy
            n_batches += 1
        
        optimizer.step()
        
        if is_dlm:
            model.step_dynamics()
    
    return total_entropy / max(n_batches, 1)


def evaluate(model, loader):
    """Evaluate accuracy"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def test_catastrophic_forgetting(epochs_per_task=5):
    """
    Test 1: Catastrophic Forgetting
    Train on Task 1 (digits 0-4), then Task 2 (digits 5-9)
    Measure retention on Task 1 after learning Task 2
    """
    print("\n" + "="*60)
    print("TEST 1: CATASTROPHIC FORGETTING")
    print("="*60)
    
    layer_sizes = [784, 256, 128, 10]
    data = load_mnist_split()
    
    results = {}
    
    for name, ModelClass, is_dlm in [
        ("DLM", DissipativeLearningMachine, True),
        ("Standard", StandardNetwork, False)
    ]:
        print(f"\n--- {name} Network ---")
        
        if is_dlm:
            model = ModelClass(layer_sizes, temperature=0.5, dissipation_rate=0.02, energy_injection_rate=0.05)
        else:
            model = ModelClass(layer_sizes)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train on Task 1
        print("Training on Task 1 (digits 0-4)...")
        for epoch in range(epochs_per_task):
            train_epoch(model, data['task1_train'], optimizer, is_dlm)
        
        task1_before = evaluate(model, data['task1_test'])
        print(f"  Task 1 accuracy after Task 1 training: {task1_before:.4f}")
        
        # Train on Task 2
        print("Training on Task 2 (digits 5-9)...")
        for epoch in range(epochs_per_task):
            train_epoch(model, data['task2_train'], optimizer, is_dlm)
        
        task1_after = evaluate(model, data['task1_test'])
        task2_acc = evaluate(model, data['task2_test'])
        
        forgetting = task1_before - task1_after
        retention = task1_after / task1_before if task1_before > 0 else 0
        
        print(f"  Task 1 accuracy after Task 2 training: {task1_after:.4f}")
        print(f"  Task 2 accuracy: {task2_acc:.4f}")
        print(f"  Forgetting (drop): {forgetting:.4f}")
        print(f"  Retention rate: {retention:.2%}")
        
        results[name] = {
            'task1_before': task1_before,
            'task1_after': task1_after,
            'task2_acc': task2_acc,
            'forgetting': forgetting,
            'retention': retention
        }
    
    # Compare
    print("\n--- COMPARISON ---")
    dlm_forget = results['DLM']['forgetting']
    std_forget = results['Standard']['forgetting']
    if std_forget > 0:
        improvement = std_forget / dlm_forget if dlm_forget > 0 else float('inf')
        print(f"DLM forgetting: {dlm_forget:.4f}")
        print(f"Standard forgetting: {std_forget:.4f}")
        print(f"Improvement ratio: {improvement:.2f}x less forgetting")
    
    return results


def test_noise_spectrum(num_samples=2000):
    """
    Test 2: Noise Spectrum Analysis
    DLM should show pink noise (1/f, slope ~ -1)
    Standard should show white noise (slope ~ 0)
    """
    print("\n" + "="*60)
    print("TEST 2: NOISE SPECTRUM ANALYSIS")
    print("="*60)
    
    layer_sizes = [784, 256, 128, 10]
    
    results = {}
    
    for name, ModelClass, is_dlm in [
        ("DLM", DissipativeLearningMachine, True),
        ("Standard", StandardNetwork, False)
    ]:
        print(f"\n--- {name} Network ---")
        
        if is_dlm:
            model = ModelClass(layer_sizes, temperature=0.5, dissipation_rate=0.02, energy_injection_rate=0.05)
        else:
            model = ModelClass(layer_sizes)
        
        model.eval()
        activations = []
        x = torch.randn(1, 784)
        
        with torch.no_grad():
            for _ in range(num_samples):
                out = model(x)
                activations.append(out.mean().item())
                if is_dlm:
                    model.step_dynamics(0.01)
                # Add small input variation for standard network
                x = x + torch.randn_like(x) * 0.01
        
        # Compute power spectrum
        activations = np.array(activations)
        activations = activations - activations.mean()  # Detrend
        
        fft = np.fft.fft(activations)
        power = np.abs(fft) ** 2
        freqs = np.fft.fftfreq(len(activations))
        
        # Fit slope in log-log space (positive frequencies only)
        pos_mask = (freqs > 0.001) & (freqs < 0.4)  # Avoid DC and Nyquist
        log_freqs = np.log10(freqs[pos_mask])
        log_power = np.log10(power[pos_mask] + 1e-10)
        
        slope, intercept, r_value, _, _ = stats.linregress(log_freqs, log_power)
        
        print(f"  Power spectrum slope: {slope:.3f}")
        print(f"  R-squared: {r_value**2:.3f}")
        
        if slope < -0.5:
            print(f"  -> Shows PINK NOISE characteristics (biological signature)")
        elif slope > -0.3:
            print(f"  -> Shows WHITE NOISE characteristics")
        else:
            print(f"  -> Shows intermediate characteristics")
        
        results[name] = {
            'slope': slope,
            'r_squared': r_value**2,
            'freqs': freqs[pos_mask],
            'power': power[pos_mask]
        }
    
    return results


def test_performance_parity(epochs=5):
    """
    Test 3: Performance Comparison
    DLM should achieve comparable accuracy to standard networks
    """
    print("\n" + "="*60)
    print("TEST 3: PERFORMANCE PARITY")
    print("="*60)
    
    layer_sizes = [784, 256, 128, 10]
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    results = {}
    
    for name, ModelClass, is_dlm in [
        ("DLM", DissipativeLearningMachine, True),
        ("Standard", StandardNetwork, False)
    ]:
        print(f"\n--- {name} Network ---")
        
        if is_dlm:
            model = ModelClass(layer_sizes, temperature=0.5, dissipation_rate=0.02, energy_injection_rate=0.05)
        else:
            model = ModelClass(layer_sizes)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        accuracies = []
        entropies = []
        
        for epoch in range(epochs):
            entropy = train_epoch(model, train_loader, optimizer, is_dlm)
            acc = evaluate(model, test_loader)
            accuracies.append(acc)
            if is_dlm:
                entropies.append(entropy)
            print(f"  Epoch {epoch+1}/{epochs}: Accuracy={acc:.4f}" + 
                  (f", Entropy={entropy:.6f}" if is_dlm else ""))
        
        results[name] = {
            'final_accuracy': accuracies[-1],
            'accuracies': accuracies,
            'entropies': entropies if is_dlm else []
        }
    
    print("\n--- COMPARISON ---")
    dlm_acc = results['DLM']['final_accuracy']
    std_acc = results['Standard']['final_accuracy']
    diff = abs(dlm_acc - std_acc)
    
    print(f"DLM final accuracy: {dlm_acc:.4f}")
    print(f"Standard final accuracy: {std_acc:.4f}")
    print(f"Difference: {diff:.4f}")
    
    if diff < 0.02:
        print("-> PASS: Performance is comparable (within 2%)")
    else:
        print(f"-> Performance gap of {diff:.1%}")
    
    return results


def visualize_results(forgetting_results, noise_results, perf_results):
    """Create visualization of all test results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Catastrophic forgetting comparison
    ax = axes[0, 0]
    names = ['DLM', 'Standard']
    forgetting = [forgetting_results['DLM']['forgetting'], forgetting_results['Standard']['forgetting']]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(names, forgetting, color=colors, edgecolor='black')
    ax.set_ylabel('Forgetting (accuracy drop)')
    ax.set_title('Catastrophic Forgetting Comparison\n(lower is better)')
    for bar, val in zip(bars, forgetting):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Retention comparison
    ax = axes[0, 1]
    retention = [forgetting_results['DLM']['retention'] * 100, 
                 forgetting_results['Standard']['retention'] * 100]
    bars = ax.bar(names, retention, color=colors, edgecolor='black')
    ax.set_ylabel('Task 1 Retention (%)')
    ax.set_title('Knowledge Retention After Learning Task 2\n(higher is better)')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, retention):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Noise spectrum
    ax = axes[1, 0]
    for name, color in [('DLM', '#2ecc71'), ('Standard', '#e74c3c')]:
        freqs = noise_results[name]['freqs']
        power = noise_results[name]['power']
        ax.loglog(freqs, power, label=f"{name} (slope={noise_results[name]['slope']:.2f})", 
                  alpha=0.7, color=color)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_title('Power Spectrum Analysis\n(slope ~ -1 indicates pink/1/f noise)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Training accuracy
    ax = axes[1, 1]
    epochs = range(1, len(perf_results['DLM']['accuracies']) + 1)
    ax.plot(epochs, perf_results['DLM']['accuracies'], 'o-', label='DLM', color='#2ecc71')
    ax.plot(epochs, perf_results['Standard']['accuracies'], 's-', label='Standard', color='#e74c3c')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Training Progress Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'theory_validation.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")
    
    return fig


def main():
    print("="*60)
    print("DISSIPATIVE LEARNING MACHINE - THEORY VALIDATION")
    print("="*60)
    
    # Run all tests
    forgetting_results = test_catastrophic_forgetting(epochs_per_task=5)
    noise_results = test_noise_spectrum(num_samples=2000)
    perf_results = test_performance_parity(epochs=5)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    # Claim 1: Catastrophic forgetting
    dlm_forget = forgetting_results['DLM']['forgetting']
    std_forget = forgetting_results['Standard']['forgetting']
    improvement = std_forget / dlm_forget if dlm_forget > 0 else float('inf')
    
    print(f"\n1. CATASTROPHIC FORGETTING:")
    print(f"   Claimed: 2.4x better retention")
    print(f"   Measured: {improvement:.2f}x improvement")
    print(f"   Status: {'VALIDATED' if improvement > 1.5 else 'NEEDS MORE TESTING'}")
    
    # Claim 2: Pink noise
    dlm_slope = noise_results['DLM']['slope']
    std_slope = noise_results['Standard']['slope']
    
    print(f"\n2. PINK NOISE (1/f) SPECTRUM:")
    print(f"   Claimed: DLM shows 1/f noise (slope ~ -1), Standard shows white noise (slope ~ 0)")
    print(f"   DLM slope: {dlm_slope:.3f}")
    print(f"   Standard slope: {std_slope:.3f}")
    print(f"   Status: {'VALIDATED' if dlm_slope < -0.5 and std_slope > -0.5 else 'MIXED RESULTS'}")
    
    # Claim 3: Performance parity
    dlm_acc = perf_results['DLM']['final_accuracy']
    std_acc = perf_results['Standard']['final_accuracy']
    
    print(f"\n3. PERFORMANCE PARITY:")
    print(f"   Claimed: No performance loss")
    print(f"   DLM accuracy: {dlm_acc:.4f}")
    print(f"   Standard accuracy: {std_acc:.4f}")
    print(f"   Status: {'VALIDATED' if abs(dlm_acc - std_acc) < 0.02 else 'GAP DETECTED'}")
    
    # Entropy production check
    if perf_results['DLM']['entropies']:
        final_entropy = perf_results['DLM']['entropies'][-1]
        print(f"\n4. ENTROPY PRODUCTION (diagnostic):")
        print(f"   Final epoch entropy: {final_entropy:.6f}")
        print(f"   Status: {'WORKING' if final_entropy > 0 else 'STILL ZERO - CHECK IMPLEMENTATION'}")
    
    # Create visualization
    visualize_results(forgetting_results, noise_results, perf_results)
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
