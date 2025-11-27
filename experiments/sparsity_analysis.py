"""
Sparsity Analysis: Why Does It Work?
====================================

Hypothesis: Sparse representations create orthogonal task embeddings
that don't interfere with each other.

Tests:
1. Representation overlap between tasks
2. Weight overlap analysis
3. Activation pattern similarity
4. Effective capacity analysis
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
from scipy.spatial.distance import cosine

from thermodynamic_neural_network import ThermodynamicNeuralNetwork


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
    
    tasks = [(0, 1), (2, 3), (4, 5)]
    
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


def get_activations(model, loader, num_samples=500):
    """Get hidden layer activations for samples"""
    model.eval()
    activations = {i: [] for i in range(len(model.layers) - 1)}
    
    samples = 0
    with torch.no_grad():
        for x, _ in loader:
            if samples >= num_samples:
                break
            
            # Forward through layers, collecting activations
            h = x
            for i, layer in enumerate(model.layers[:-1]):
                h, _ = layer(h)
                h = F.gelu(h)
                activations[i].append(h.cpu().numpy())
            
            samples += x.size(0)
    
    return {k: np.vstack(v)[:num_samples] for k, v in activations.items()}


def compute_active_neurons(activations, threshold=0.01):
    """Compute which neurons are active (above threshold)"""
    return (np.abs(activations) > threshold).astype(float)


def compute_overlap(active1, active2):
    """Compute Jaccard overlap between two sets of active neurons"""
    # Average over samples
    avg1 = active1.mean(axis=0) > 0.1  # Neuron active in >10% of samples
    avg2 = active2.mean(axis=0) > 0.1
    
    intersection = (avg1 & avg2).sum()
    union = (avg1 | avg2).sum()
    
    if union == 0:
        return 0.0
    return intersection / union


def compute_representation_similarity(act1, act2):
    """Compute average cosine similarity between representations"""
    similarities = []
    for i in range(min(len(act1), len(act2))):
        sim = 1 - cosine(act1[i], act2[i])
        if not np.isnan(sim):
            similarities.append(sim)
    return np.mean(similarities) if similarities else 0.0


def analyze_sparsity_levels():
    """Analyze how different sparsity levels affect representation overlap"""
    
    sparsity_levels = [0.05, 0.10, 0.15, 0.25, 0.50, 1.0]
    layer_sizes = [784, 256, 10]
    
    loaders, tasks = load_split_mnist()
    
    results = {
        'sparsity': [],
        'overlap_t1_t2': [],
        'overlap_t1_t3': [],
        'overlap_t2_t3': [],
        'similarity_t1_t2': [],
        'active_neurons_t1': [],
        'active_neurons_t2': [],
        'forgetting': [],
    }
    
    print("=" * 70)
    print("SPARSITY ANALYSIS: WHY DOES IT WORK?")
    print("=" * 70)
    
    for sparsity in sparsity_levels:
        print(f"\n{'='*50}")
        print(f"Sparsity = {sparsity} ({int(sparsity*100)}% active)")
        print(f"{'='*50}")
        
        # Create and train model on Task 1
        model = ThermodynamicNeuralNetwork(
            layer_sizes, sparsity=sparsity, temperature=1.0,
            consolidation_rate=0.005
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train on Task 1
        for _ in range(3):
            for x, y in loaders['task1_train']:
                model.training_step(x, y, optimizer)
        
        # Get Task 1 activations
        act_t1 = get_activations(model, loaders['task1_test'])
        active_t1 = compute_active_neurons(act_t1[0])
        
        # Evaluate Task 1
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loaders['task1_test']:
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        task1_before = correct / total
        
        # Train on Task 2
        for _ in range(3):
            for x, y in loaders['task2_train']:
                model.training_step(x, y, optimizer)
        
        # Get Task 2 activations (using task 2 data)
        act_t2 = get_activations(model, loaders['task2_test'])
        active_t2 = compute_active_neurons(act_t2[0])
        
        # Also get activations when showing Task 1 data to trained model
        act_t1_after = get_activations(model, loaders['task1_test'])
        active_t1_after = compute_active_neurons(act_t1_after[0])
        
        # Evaluate Task 1 after Task 2
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loaders['task1_test']:
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        task1_after = correct / total
        
        # Train on Task 3
        for _ in range(3):
            for x, y in loaders['task3_train']:
                model.training_step(x, y, optimizer)
        
        act_t3 = get_activations(model, loaders['task3_test'])
        active_t3 = compute_active_neurons(act_t3[0])
        
        # Compute metrics
        overlap_12 = compute_overlap(active_t1, active_t2)
        overlap_13 = compute_overlap(active_t1, active_t3)
        overlap_23 = compute_overlap(active_t2, active_t3)
        
        similarity_12 = compute_representation_similarity(act_t1[0], act_t2[0])
        
        n_active_t1 = (active_t1.mean(axis=0) > 0.1).sum()
        n_active_t2 = (active_t2.mean(axis=0) > 0.1).sum()
        
        forgetting = task1_before - task1_after
        
        print(f"  Active neurons T1: {n_active_t1}")
        print(f"  Active neurons T2: {n_active_t2}")
        print(f"  Overlap T1-T2: {overlap_12:.3f}")
        print(f"  Overlap T1-T3: {overlap_13:.3f}")
        print(f"  Overlap T2-T3: {overlap_23:.3f}")
        print(f"  Representation similarity T1-T2: {similarity_12:.3f}")
        print(f"  Task 1 accuracy: {task1_before:.4f} → {task1_after:.4f}")
        print(f"  Forgetting: {forgetting:.4f}")
        
        results['sparsity'].append(sparsity)
        results['overlap_t1_t2'].append(overlap_12)
        results['overlap_t1_t3'].append(overlap_13)
        results['overlap_t2_t3'].append(overlap_23)
        results['similarity_t1_t2'].append(similarity_12)
        results['active_neurons_t1'].append(n_active_t1)
        results['active_neurons_t2'].append(n_active_t2)
        results['forgetting'].append(forgetting)
    
    return results


def visualize_analysis(results):
    """Visualize the analysis results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sparsity = results['sparsity']
    
    # 1. Overlap vs Sparsity
    ax1 = axes[0, 0]
    ax1.plot(sparsity, results['overlap_t1_t2'], 'o-', label='T1-T2', linewidth=2)
    ax1.plot(sparsity, results['overlap_t1_t3'], 's-', label='T1-T3', linewidth=2)
    ax1.plot(sparsity, results['overlap_t2_t3'], '^-', label='T2-T3', linewidth=2)
    ax1.set_xlabel('Sparsity (fraction active)')
    ax1.set_ylabel('Neuron Overlap (Jaccard)')
    ax1.set_title('Task Representation Overlap vs Sparsity\n(lower = more orthogonal)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Forgetting vs Overlap
    ax2 = axes[0, 1]
    ax2.scatter(results['overlap_t1_t2'], results['forgetting'], s=100, c=sparsity, cmap='viridis')
    ax2.set_xlabel('T1-T2 Overlap')
    ax2.set_ylabel('Forgetting')
    ax2.set_title('Forgetting vs Representation Overlap\n(color = sparsity)')
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Sparsity')
    
    # Add correlation line
    from scipy import stats
    slope, intercept, r, _, _ = stats.linregress(results['overlap_t1_t2'], results['forgetting'])
    x_line = np.linspace(min(results['overlap_t1_t2']), max(results['overlap_t1_t2']), 100)
    ax2.plot(x_line, slope * x_line + intercept, 'r--', label=f'r={r:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Active Neurons vs Sparsity
    ax3 = axes[1, 0]
    ax3.plot(sparsity, results['active_neurons_t1'], 'o-', label='Task 1', linewidth=2)
    ax3.plot(sparsity, results['active_neurons_t2'], 's-', label='Task 2', linewidth=2)
    ax3.set_xlabel('Sparsity (fraction active)')
    ax3.set_ylabel('Number of Active Neurons')
    ax3.set_title('Active Neurons per Task vs Sparsity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Combined: Overlap and Forgetting
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(sparsity, results['overlap_t1_t2'], 'b-o', label='Overlap', linewidth=2)
    line2 = ax4_twin.plot(sparsity, results['forgetting'], 'r-s', label='Forgetting', linewidth=2)
    
    ax4.set_xlabel('Sparsity (fraction active)')
    ax4.set_ylabel('Overlap (Jaccard)', color='blue')
    ax4_twin.set_ylabel('Forgetting', color='red')
    ax4.set_title('Overlap and Forgetting vs Sparsity')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'results', 'sparsity_analysis.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")


def main():
    results = analyze_sparsity_levels()
    
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\n{:<10} {:>10} {:>12} {:>12} {:>12}".format(
        "Sparsity", "Active N", "Overlap", "Similarity", "Forgetting"
    ))
    print("-" * 60)
    
    for i, s in enumerate(results['sparsity']):
        print("{:<10.2f} {:>10.0f} {:>12.3f} {:>12.3f} {:>12.4f}".format(
            s,
            results['active_neurons_t1'][i],
            results['overlap_t1_t2'][i],
            results['similarity_t1_t2'][i],
            results['forgetting'][i]
        ))
    
    # Correlation analysis
    from scipy import stats
    
    overlap = np.array(results['overlap_t1_t2'])
    forgetting = np.array(results['forgetting'])
    
    corr, p_val = stats.pearsonr(overlap, forgetting)
    
    print(f"\n*** CORRELATION: Overlap vs Forgetting ***")
    print(f"    Pearson r: {corr:.4f}")
    print(f"    p-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print(f"    SIGNIFICANT: Higher overlap → {'MORE' if corr > 0 else 'LESS'} forgetting")
    
    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    
    if corr > 0.5 and p_val < 0.1:
        print("""
    CONFIRMED: Sparse representations reduce forgetting by creating
    more ORTHOGONAL task representations.
    
    - Low sparsity (5%): Few neurons per task → Low overlap → Less interference
    - High sparsity (100%): Many neurons per task → High overlap → More interference
    
    This is consistent with the "sparse coding" hypothesis in neuroscience:
    the brain uses sparse, distributed representations to maximize storage
    capacity and minimize interference between memories.
        """)
    else:
        print("""
    The relationship between overlap and forgetting needs further investigation.
    Other factors may be at play.
        """)
    
    visualize_analysis(results)
    
    return results


if __name__ == "__main__":
    main()
