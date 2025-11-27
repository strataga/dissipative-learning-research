"""
Comprehensive Validation of Thermodynamic Neural Network (TNN)
===============================================================

Tests the key claims:
1. Reduced catastrophic forgetting via consolidation + sparsity
2. Biological noise characteristics  
3. Energy efficiency
4. Sleep cycle benefits

Compares: TNN vs Standard vs Original DLM
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

from thermodynamic_neural_network import (
    ThermodynamicNeuralNetwork,
    train_tnn,
    evaluate_tnn,
)
from dissipative_learning_machine import (
    DissipativeLearningMachine,
    StandardNetwork,
)


def load_mnist(batch_size=64):
    """Load full MNIST dataset"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, transform=transform)
    
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size)
    )


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
    
    # Split into 5 tasks: 0-1, 2-3, 4-5, 6-7, 8-9
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


def train_standard(model, loader, optimizer):
    """Standard training step"""
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()


def train_dlm(model, loader, optimizer):
    """DLM training step"""
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)  # Use standard loss, not compute_loss
        loss.backward()
        optimizer.step()
        model.step_dynamics()


def evaluate(model, loader):
    """Evaluate any model"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def experiment_catastrophic_forgetting(epochs_per_task=3):
    """
    EXPERIMENT 1: Catastrophic Forgetting on 5-Task MNIST
    
    Train on tasks sequentially, measure retention on all previous tasks.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: CATASTROPHIC FORGETTING (5-Task Sequential MNIST)")
    print("="*70)
    
    layer_sizes = [784, 256, 256, 10]
    loaders, tasks = load_split_mnist()
    
    results = {}
    
    # Test each architecture
    for name, create_model, train_fn in [
        ("TNN", 
         lambda: ThermodynamicNeuralNetwork(
             layer_sizes, sparsity=0.15, temperature=1.0,
             consolidation_rate=0.005, sleep_frequency=50
         ),
         lambda m, l, o: train_tnn_epoch(m, l, o)),
        ("DLM",
         lambda: DissipativeLearningMachine(
             layer_sizes, temperature=0.5, dissipation_rate=0.02
         ),
         train_dlm),
        ("Standard",
         lambda: StandardNetwork(layer_sizes),
         train_standard),
    ]:
        print(f"\n{'='*40}")
        print(f"Training: {name}")
        print(f"{'='*40}")
        
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Track accuracy on all tasks after each task
        task_accuracies = defaultdict(list)
        
        for task_idx in range(5):
            task_name = f'task{task_idx+1}'
            print(f"\n--- Training on Task {task_idx+1} (digits {tasks[task_idx]}) ---")
            
            # Train on current task
            for epoch in range(epochs_per_task):
                train_fn(model, loaders[f'{task_name}_train'], optimizer)
            
            # Evaluate on ALL tasks
            print("  Accuracies after this task:")
            for eval_idx in range(task_idx + 1):
                eval_task = f'task{eval_idx+1}'
                acc = evaluate(model, loaders[f'{eval_task}_test'])
                task_accuracies[eval_task].append(acc)
                print(f"    Task {eval_idx+1}: {acc:.4f}")
        
        results[name] = dict(task_accuracies)
    
    # Compute forgetting metrics
    print("\n" + "="*70)
    print("FORGETTING ANALYSIS")
    print("="*70)
    
    forgetting_scores = {}
    for name in results:
        total_forgetting = 0
        for task in results[name]:
            accs = results[name][task]
            if len(accs) > 1:
                # Forgetting = peak accuracy - final accuracy
                forgetting = max(accs) - accs[-1]
                total_forgetting += forgetting
        avg_forgetting = total_forgetting / 4  # 4 tasks can be forgotten
        forgetting_scores[name] = avg_forgetting
        print(f"{name}: Average forgetting = {avg_forgetting:.4f}")
    
    return results, forgetting_scores


def train_tnn_epoch(model, loader, optimizer):
    """Train TNN for one epoch"""
    for x, y in loader:
        model.training_step(x, y, optimizer)


def experiment_noise_spectrum(num_samples=3000):
    """
    EXPERIMENT 2: Noise Spectrum Analysis
    
    Measure power spectrum of network activations over time.
    Pink noise (1/f) is a signature of biological neural systems.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: NOISE SPECTRUM ANALYSIS")
    print("="*70)
    
    layer_sizes = [784, 256, 256, 10]
    results = {}
    
    for name, create_model, has_dynamics in [
        ("TNN", 
         lambda: ThermodynamicNeuralNetwork(layer_sizes, sparsity=0.15, temperature=1.0),
         True),
        ("DLM",
         lambda: DissipativeLearningMachine(layer_sizes, temperature=0.5),
         True),
        ("Standard",
         lambda: StandardNetwork(layer_sizes),
         False),
    ]:
        print(f"\n--- {name} ---")
        
        model = create_model()
        model.eval()
        
        activations = []
        x = torch.randn(1, 784)
        
        with torch.no_grad():
            for _ in range(num_samples):
                out = model(x)
                activations.append(out.mean().item())
                
                # Update dynamics if applicable
                if has_dynamics:
                    if hasattr(model, 'step_dynamics'):
                        model.step_dynamics(0.01)
                    elif hasattr(model, 'layers'):
                        for layer in model.layers:
                            if hasattr(layer, 'inject_noise'):
                                layer.inject_noise(0.01)
                
                # Vary input slightly
                x = x + torch.randn_like(x) * 0.005
        
        # Analyze spectrum
        activations = np.array(activations)
        activations = activations - activations.mean()
        
        fft = np.fft.fft(activations)
        power = np.abs(fft) ** 2
        freqs = np.fft.fftfreq(len(activations))
        
        # Fit slope in log-log space
        pos_mask = (freqs > 0.001) & (freqs < 0.4)
        log_freqs = np.log10(freqs[pos_mask])
        log_power = np.log10(power[pos_mask] + 1e-10)
        
        slope, intercept, r_value, _, _ = stats.linregress(log_freqs, log_power)
        
        print(f"  Power spectrum slope: {slope:.3f}")
        print(f"  R-squared: {r_value**2:.3f}")
        
        noise_type = "PINK (1/f)" if slope < -0.7 else "WHITE" if slope > -0.3 else "MIXED"
        print(f"  Classification: {noise_type}")
        
        results[name] = {
            'slope': slope,
            'r_squared': r_value**2,
            'freqs': freqs[pos_mask],
            'power': power[pos_mask],
            'noise_type': noise_type
        }
    
    return results


def experiment_consolidation_dynamics():
    """
    EXPERIMENT 3: Consolidation Dynamics
    
    Track how weights become consolidated over training.
    Show that important weights get protected.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: CONSOLIDATION DYNAMICS")
    print("="*70)
    
    layer_sizes = [784, 128, 10]
    train_loader, test_loader = load_mnist()
    
    model = ThermodynamicNeuralNetwork(
        layer_sizes, sparsity=0.2, temperature=1.0,
        consolidation_rate=0.01, sleep_frequency=100
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    consolidation_history = []
    accuracy_history = []
    
    print("\nTraining with consolidation tracking...")
    
    # Train for several epochs
    for epoch in range(5):
        # Train
        for batch_idx, (x, y) in enumerate(train_loader):
            model.training_step(x, y, optimizer)
            
            # Record consolidation every 100 batches
            if batch_idx % 100 == 0:
                consol = model.layers[0].consolidation.mean().item()
                consolidation_history.append(consol)
        
        # Evaluate
        acc = evaluate_tnn(model, test_loader)
        accuracy_history.append(acc)
        print(f"  Epoch {epoch+1}: Accuracy={acc:.4f}, "
              f"Consolidation={consolidation_history[-1]:.4f}")
    
    # Analyze consolidation distribution
    final_consolidation = model.layers[0].consolidation.detach().cpu().numpy()
    
    print(f"\nFinal consolidation statistics:")
    print(f"  Mean: {final_consolidation.mean():.4f}")
    print(f"  Std:  {final_consolidation.std():.4f}")
    print(f"  Max:  {final_consolidation.max():.4f}")
    print(f"  % highly consolidated (>0.5): {(final_consolidation > 0.5).mean()*100:.1f}%")
    
    return {
        'consolidation_history': consolidation_history,
        'accuracy_history': accuracy_history,
        'final_consolidation': final_consolidation
    }


def experiment_sleep_ablation():
    """
    EXPERIMENT 4: Sleep Cycle Ablation
    
    Compare TNN with and without sleep phases.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: SLEEP CYCLE ABLATION")
    print("="*70)
    
    layer_sizes = [784, 256, 10]
    loaders, tasks = load_split_mnist()
    
    results = {}
    
    for sleep_freq, name in [(50, "With Sleep"), (100000, "No Sleep")]:
        print(f"\n--- {name} ---")
        
        model = ThermodynamicNeuralNetwork(
            layer_sizes, sparsity=0.15, temperature=1.0,
            consolidation_rate=0.005, sleep_frequency=sleep_freq
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train on task 1, then task 2
        for epoch in range(3):
            train_tnn_epoch(model, loaders['task1_train'], optimizer)
        
        task1_before = evaluate_tnn(model, loaders['task1_test'])
        print(f"  After Task 1: {task1_before:.4f}")
        
        for epoch in range(3):
            train_tnn_epoch(model, loaders['task2_train'], optimizer)
        
        task1_after = evaluate_tnn(model, loaders['task1_test'])
        task2_acc = evaluate_tnn(model, loaders['task2_test'])
        
        forgetting = task1_before - task1_after
        print(f"  After Task 2: Task1={task1_after:.4f}, Task2={task2_acc:.4f}")
        print(f"  Forgetting: {forgetting:.4f}")
        
        results[name] = {
            'task1_before': task1_before,
            'task1_after': task1_after,
            'task2': task2_acc,
            'forgetting': forgetting
        }
    
    return results


def experiment_sparsity_ablation():
    """
    EXPERIMENT 5: Sparsity Ablation
    
    Test effect of different sparsity levels.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: SPARSITY ABLATION")
    print("="*70)
    
    layer_sizes = [784, 256, 10]
    loaders, tasks = load_split_mnist()
    
    results = {}
    
    for sparsity in [0.05, 0.15, 0.30, 0.50, 1.0]:
        print(f"\n--- Sparsity = {sparsity} ({int(sparsity*100)}% active) ---")
        
        model = ThermodynamicNeuralNetwork(
            layer_sizes, sparsity=sparsity, temperature=1.0,
            consolidation_rate=0.005
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train on task 1
        for _ in range(3):
            train_tnn_epoch(model, loaders['task1_train'], optimizer)
        task1_before = evaluate_tnn(model, loaders['task1_test'])
        
        # Train on task 2
        for _ in range(3):
            train_tnn_epoch(model, loaders['task2_train'], optimizer)
        task1_after = evaluate_tnn(model, loaders['task1_test'])
        
        forgetting = task1_before - task1_after
        
        results[sparsity] = {
            'task1_before': task1_before,
            'task1_after': task1_after,
            'forgetting': forgetting
        }
        
        print(f"  Task 1: {task1_before:.4f} -> {task1_after:.4f}")
        print(f"  Forgetting: {forgetting:.4f}")
    
    return results


def visualize_all_results(
    forgetting_results, 
    noise_results,
    consolidation_results,
    sleep_results,
    sparsity_results
):
    """Create comprehensive visualization"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Forgetting comparison
    ax1 = fig.add_subplot(2, 3, 1)
    names = list(forgetting_results[1].keys())
    scores = [forgetting_results[1][n] for n in names]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax1.bar(names, scores, color=colors, edgecolor='black')
    ax1.set_ylabel('Average Forgetting')
    ax1.set_title('Catastrophic Forgetting\n(lower is better)')
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', fontweight='bold')
    
    # 2. Task retention over time
    ax2 = fig.add_subplot(2, 3, 2)
    for name, color in [('TNN', '#2ecc71'), ('Standard', '#e74c3c')]:
        task1_accs = forgetting_results[0][name].get('task1', [])
        if task1_accs:
            ax2.plot(range(1, len(task1_accs)+1), task1_accs, 'o-', 
                    label=name, color=color, linewidth=2)
    ax2.set_xlabel('Tasks Trained')
    ax2.set_ylabel('Task 1 Accuracy')
    ax2.set_title('Task 1 Retention During Sequential Learning')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Noise spectrum
    ax3 = fig.add_subplot(2, 3, 3)
    for name, color in [('TNN', '#2ecc71'), ('DLM', '#3498db'), ('Standard', '#e74c3c')]:
        if name in noise_results:
            slope = noise_results[name]['slope']
            ax3.loglog(noise_results[name]['freqs'], noise_results[name]['power'],
                      label=f"{name} (slope={slope:.2f})", alpha=0.7, color=color)
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Power')
    ax3.set_title('Noise Spectrum Analysis\n(slope~-1 = pink noise)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Consolidation histogram
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(consolidation_results['final_consolidation'].flatten(), bins=50, 
             color='#3498db', edgecolor='black', alpha=0.7)
    ax4.axvline(x=0.5, color='red', linestyle='--', label='High consolidation threshold')
    ax4.set_xlabel('Consolidation Level')
    ax4.set_ylabel('Count')
    ax4.set_title('Weight Consolidation Distribution')
    ax4.legend()
    
    # 5. Sleep ablation
    ax5 = fig.add_subplot(2, 3, 5)
    sleep_names = list(sleep_results.keys())
    sleep_forgetting = [sleep_results[n]['forgetting'] for n in sleep_names]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax5.bar(sleep_names, sleep_forgetting, color=colors, edgecolor='black')
    ax5.set_ylabel('Forgetting')
    ax5.set_title('Effect of Sleep Cycles\n(lower is better)')
    for bar, score in zip(bars, sleep_forgetting):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', fontweight='bold')
    
    # 6. Sparsity effect
    ax6 = fig.add_subplot(2, 3, 6)
    sparsities = sorted(sparsity_results.keys())
    forgetting = [sparsity_results[s]['forgetting'] for s in sparsities]
    ax6.plot(sparsities, forgetting, 'o-', color='#3498db', linewidth=2, markersize=8)
    ax6.set_xlabel('Sparsity (fraction active)')
    ax6.set_ylabel('Forgetting')
    ax6.set_title('Effect of Sparsity Level')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'tnn_validation.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")


def main():
    print("="*70)
    print("THERMODYNAMIC NEURAL NETWORK - COMPREHENSIVE VALIDATION")
    print("="*70)
    
    # Run all experiments
    forgetting_results = experiment_catastrophic_forgetting(epochs_per_task=3)
    noise_results = experiment_noise_spectrum(num_samples=2000)
    consolidation_results = experiment_consolidation_dynamics()
    sleep_results = experiment_sleep_ablation()
    sparsity_results = experiment_sparsity_ablation()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\n1. CATASTROPHIC FORGETTING:")
    for name, score in forgetting_results[1].items():
        print(f"   {name}: {score:.4f}")
    
    best = min(forgetting_results[1], key=forgetting_results[1].get)
    worst = max(forgetting_results[1], key=forgetting_results[1].get)
    improvement = forgetting_results[1][worst] / (forgetting_results[1][best] + 1e-8)
    print(f"   Best ({best}) is {improvement:.1f}x better than worst ({worst})")
    
    print("\n2. NOISE CHARACTERISTICS:")
    for name, data in noise_results.items():
        print(f"   {name}: slope={data['slope']:.3f} ({data['noise_type']})")
    
    print("\n3. CONSOLIDATION:")
    fc = consolidation_results['final_consolidation']
    print(f"   Highly consolidated weights: {(fc > 0.5).mean()*100:.1f}%")
    
    print("\n4. SLEEP EFFECT:")
    sleep_improvement = sleep_results['No Sleep']['forgetting'] - sleep_results['With Sleep']['forgetting']
    print(f"   Sleep reduces forgetting by: {sleep_improvement:.4f}")
    
    print("\n5. OPTIMAL SPARSITY:")
    best_sparsity = min(sparsity_results, key=lambda s: sparsity_results[s]['forgetting'])
    print(f"   Best sparsity: {best_sparsity} ({int(best_sparsity*100)}% active)")
    
    # Visualize
    visualize_all_results(
        forgetting_results, noise_results, consolidation_results,
        sleep_results, sparsity_results
    )
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
