"""
Debug Entropy Calculation
=========================

EXP-011: Diagnose why entropy production is always 0

Hypothesis: Either gradients or information currents are not being computed correctly.
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from thermodynamic_neural_network import ThermodynamicNeuralNetwork
from dissipative_learning_machine import DissipativeLearningMachine


def load_mnist_sample(batch_size=32):
    """Load a small MNIST sample for debugging"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    return DataLoader(train, batch_size=batch_size, shuffle=True)


def debug_tnn():
    """Debug TNN entropy calculation"""
    print("=" * 70)
    print("DEBUGGING TNN ENTROPY CALCULATION")
    print("=" * 70)
    
    model = ThermodynamicNeuralNetwork([784, 256, 10], sparsity=0.15, temperature=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = load_mnist_sample()
    
    x, y = next(iter(loader))
    
    print("\n1. BEFORE TRAINING STEP")
    print("-" * 40)
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}:")
        print(f"  weight.grad: {layer.weight.grad}")
        print(f"  input_activity mean: {layer.input_activity.mean().item():.6f}")
        print(f"  output_activity mean: {layer.output_activity.mean().item():.6f}")
        print(f"  information_current: {layer.compute_information_current():.6f}")
    
    print("\n2. AFTER FORWARD PASS")
    print("-" * 40)
    optimizer.zero_grad()
    output = model(x)
    
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}:")
        print(f"  weight.grad: {layer.weight.grad}")
        print(f"  input_activity mean: {layer.input_activity.mean().item():.6f}")
        print(f"  output_activity mean: {layer.output_activity.mean().item():.6f}")
        print(f"  information_current: {layer.compute_information_current():.6f}")
    
    print("\n3. AFTER BACKWARD PASS")
    print("-" * 40)
    loss = F.cross_entropy(output, y)
    loss.backward()
    
    for i, layer in enumerate(model.layers):
        grad_mean = layer.weight.grad.abs().mean().item() if layer.weight.grad is not None else 0
        print(f"Layer {i}:")
        print(f"  weight.grad exists: {layer.weight.grad is not None}")
        print(f"  weight.grad mean abs: {grad_mean:.6f}")
        print(f"  information_current: {layer.compute_information_current():.6f}")
    
    print("\n4. ENTROPY CALCULATION BREAKDOWN")
    print("-" * 40)
    total_entropy = 0.0
    for i, layer in enumerate(model.layers):
        if layer.weight.grad is not None:
            force = layer.weight.grad.abs().mean().item()
            current = abs(layer.compute_information_current())
            entropy = current * force / (model.temperature + 1e-8)
            print(f"Layer {i}:")
            print(f"  force (grad magnitude): {force:.6f}")
            print(f"  current (info flow): {current:.6f}")
            print(f"  entropy = current * force / T: {entropy:.6f}")
            total_entropy += entropy
        else:
            print(f"Layer {i}: NO GRADIENT")
    
    print(f"\nTOTAL ENTROPY: {total_entropy:.6f}")
    
    print("\n5. USING TRAINING_STEP METHOD")
    print("-" * 40)
    
    # Reset model
    model = ThermodynamicNeuralNetwork([784, 256, 10], sparsity=0.15, temperature=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    metrics = model.training_step(x, y, optimizer)
    print(f"  Returned entropy: {metrics['entropy']:.6f}")
    print(f"  State entropy_production: {model.state.entropy_production:.6f}")
    
    return total_entropy


def debug_dlm():
    """Debug DLM entropy calculation"""
    print("\n" + "=" * 70)
    print("DEBUGGING DLM ENTROPY CALCULATION")
    print("=" * 70)
    
    model = DissipativeLearningMachine([784, 256, 10], temperature=0.5, dissipation_rate=0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = load_mnist_sample()
    
    x, y = next(iter(loader))
    
    print("\n1. BEFORE TRAINING")
    print("-" * 40)
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}:")
        print(f"  input_current mean: {layer.input_current.mean().item():.6f}")
        print(f"  output_current mean: {layer.output_current.mean().item():.6f}")
        print(f"  weight_velocity norm: {layer.weight_velocity.norm().item():.6f}")
    
    print("\n2. AFTER FORWARD PASS")
    print("-" * 40)
    optimizer.zero_grad()
    output = model(x)
    
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}:")
        print(f"  input_current mean: {layer.input_current.mean().item():.6f}")
        print(f"  output_current mean: {layer.output_current.mean().item():.6f}")
    
    print("\n3. AFTER BACKWARD")
    print("-" * 40)
    loss = F.cross_entropy(output, y)
    loss.backward()
    
    for i, layer in enumerate(model.layers):
        grad_mean = layer.weight.grad.abs().mean().item() if layer.weight.grad is not None else 0
        print(f"Layer {i}:")
        print(f"  weight.grad mean abs: {grad_mean:.6f}")
    
    print("\n4. ENTROPY CALCULATION")
    print("-" * 40)
    total_entropy = 0.0
    for i, layer in enumerate(model.layers):
        entropy = layer.compute_entropy_production()
        print(f"Layer {i}: entropy = {entropy:.6f}")
        total_entropy += entropy
    
    print(f"\nTOTAL ENTROPY: {total_entropy:.6f}")
    
    print("\n5. AFTER step_dynamics()")
    print("-" * 40)
    model.step_dynamics()
    print(f"  State entropy_production: {model.state.entropy_production:.6f}")
    
    return total_entropy


def identify_bug():
    """Identify the root cause of entropy = 0"""
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)
    
    tnn_entropy = debug_tnn()
    dlm_entropy = debug_dlm()
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if tnn_entropy == 0 and dlm_entropy == 0:
        print("\nBOTH TNN and DLM have zero entropy!")
        print("\nPossible causes:")
        print("1. Information current is always ~0 (activities not updating)")
        print("2. Gradients are too small")
        print("3. Calculation order issue")
    elif tnn_entropy == 0:
        print("\nTNN has zero entropy, DLM doesn't")
        print("Issue is specific to TNN implementation")
    elif dlm_entropy == 0:
        print("\nDLM has zero entropy, TNN doesn't")
        print("Issue is specific to DLM implementation")
    else:
        print(f"\nBoth have non-zero entropy!")
        print(f"TNN: {tnn_entropy:.6f}")
        print(f"DLM: {dlm_entropy:.6f}")
        print("The bug might be in how experiments call these functions")


if __name__ == "__main__":
    identify_bug()
