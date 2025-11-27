"""
Dissipative Learning Machine (DLM)
==================================

A novel neural network architecture inspired by non-equilibrium thermodynamics
and Prigogine's dissipative structures.

Core Principles:
1. Weights have DYNAMICS - they evolve continuously, not just during training
2. Information FLOWS through the network (currents, not static storage)
3. Learning maximizes entropy production under constraints
4. The system operates far from equilibrium

Author: Experimental / Research Code
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math


@dataclass
class ThermodynamicState:
    """Tracks the thermodynamic state of the network"""
    energy: float = 0.0
    entropy_production: float = 0.0
    information_current: float = 0.0
    dissipation: float = 0.0
    temperature: float = 1.0


class DissipativeLayer(nn.Module):
    """
    A neural network layer with dissipative dynamics.
    
    Unlike standard layers where weights are static between updates,
    this layer has:
    - Dynamic weights that evolve according to a flow equation
    - Information currents that measure data throughput
    - Explicit energy dissipation
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        temperature: float = 1.0,
        dissipation_rate: float = 0.01,
        current_strength: float = 0.1
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.dissipation_rate = dissipation_rate
        self.current_strength = current_strength
        
        # Static weight matrix (learned)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Dynamic weight perturbations (evolving)
        self.register_buffer('weight_velocity', torch.zeros_like(self.weight))
        self.register_buffer('weight_momentum', torch.zeros_like(self.weight))
        
        # Information current tracking
        self.register_buffer('input_current', torch.zeros(in_features))
        self.register_buffer('output_current', torch.zeros(out_features))
        
        # Entropy production accumulator
        self.register_buffer('local_entropy_production', torch.tensor(0.0))
        
    def compute_effective_weight(self) -> torch.Tensor:
        """
        Compute the effective weight including dynamic perturbations.
        This is the key innovation - weights are not static!
        """
        # Add thermal fluctuations scaled by temperature
        noise = torch.randn_like(self.weight) * math.sqrt(self.temperature) * 0.01
        
        # Effective weight = static + velocity perturbation + noise
        return self.weight + self.weight_velocity * self.current_strength + noise
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with information current tracking.
        
        Returns:
            output: The layer output
            current: The information current (measure of information flow)
        """
        # Track input current (exponential moving average of input activity)
        with torch.no_grad():
            input_activity = x.abs().mean(dim=0) if x.dim() > 1 else x.abs()
            self.input_current = 0.9 * self.input_current + 0.1 * input_activity
        
        # Compute with effective (dynamic) weights
        effective_weight = self.compute_effective_weight()
        output = F.linear(x, effective_weight, self.bias)
        
        # Track output current
        with torch.no_grad():
            output_activity = output.abs().mean(dim=0) if output.dim() > 1 else output.abs()
            self.output_current = 0.9 * self.output_current + 0.1 * output_activity
        
        # Compute information current (flux through the layer)
        # This is analogous to particle current in non-equilibrium stat mech
        current = self.output_current.mean() - self.input_current.mean()
        
        return output, current
    
    def dissipate(self) -> float:
        """
        Apply dissipation to weight dynamics.
        Models energy loss to environment (like friction/viscosity).
        
        Returns:
            dissipated_energy: Amount of energy dissipated
        """
        # Dissipation reduces velocity (like friction)
        dissipated_energy = (self.weight_velocity ** 2).sum().item() * self.dissipation_rate
        self.weight_velocity *= (1 - self.dissipation_rate)
        
        return dissipated_energy
    
    def inject_energy(self, energy: float):
        """
        Inject energy into the layer's dynamics.
        This drives the system away from equilibrium.
        """
        # Energy injection creates random velocity perturbations
        direction = torch.randn_like(self.weight_velocity)
        direction = direction / (direction.norm() + 1e-8)
        self.weight_velocity += direction * math.sqrt(energy)
    
    def compute_entropy_production(self) -> float:
        """
        Compute local entropy production rate.
        
        Based on non-equilibrium thermodynamics:
        σ = J · F / T
        where J is the current and F is the thermodynamic force
        """
        # Thermodynamic force is the gradient of the "potential" (weights)
        force = self.weight.grad.abs().mean().item() if self.weight.grad is not None else 0.0
        
        # Current is the information flow
        current = (self.output_current.mean() - self.input_current.mean()).abs().item()
        
        # Entropy production = current × force / temperature
        entropy_prod = current * force / (self.temperature + 1e-8)
        
        self.local_entropy_production = torch.tensor(entropy_prod)
        return entropy_prod
    
    def update_dynamics(self, dt: float = 0.01):
        """
        Evolve the weight dynamics forward in time.
        
        This is the core equation:
        dw/dt = -∇L + η(T) + J
        
        where:
        - ∇L is the loss gradient (from backprop)
        - η(T) is thermal noise
        - J is the information current contribution
        """
        with torch.no_grad():
            # Add gradient-based update if available
            if self.weight.grad is not None:
                self.weight_velocity -= self.weight.grad * dt
            
            # Add momentum (inertia in the dynamics)
            self.weight_momentum = 0.9 * self.weight_momentum + 0.1 * self.weight_velocity
            
            # Thermal fluctuations (Langevin dynamics)
            noise = torch.randn_like(self.weight) * math.sqrt(2 * self.temperature * dt)
            self.weight_velocity += noise
            
            # Information current feedback (the novel part)
            # Weights that carry more current get reinforced
            current_feedback = torch.outer(self.output_current, self.input_current)
            current_feedback = current_feedback / (current_feedback.norm() + 1e-8)
            self.weight_velocity += current_feedback * self.current_strength * dt


class DissipativeLearningMachine(nn.Module):
    """
    A complete Dissipative Learning Machine.
    
    This network maintains a non-equilibrium steady state by:
    1. Continuously injecting energy
    2. Dissipating energy through dynamics
    3. Learning by maximizing entropy production
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        temperature: float = 1.0,
        dissipation_rate: float = 0.01,
        energy_injection_rate: float = 0.1
    ):
        super().__init__()
        
        self.temperature = temperature
        self.energy_injection_rate = energy_injection_rate
        
        # Build dissipative layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(DissipativeLayer(
                layer_sizes[i],
                layer_sizes[i + 1],
                temperature=temperature,
                dissipation_rate=dissipation_rate
            ))
        
        # Thermodynamic state tracking
        self.state = ThermodynamicState(temperature=temperature)
        
        # History for analysis
        self.entropy_history = []
        self.energy_history = []
        self.current_history = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all dissipative layers"""
        total_current = 0.0
        
        for i, layer in enumerate(self.layers):
            x, current = layer(x)
            total_current += current.item() if isinstance(current, torch.Tensor) else current
            
            # Apply nonlinearity (except last layer)
            if i < len(self.layers) - 1:
                x = F.gelu(x)  # Smooth activation for gradient flow
        
        self.state.information_current = total_current
        return x
    
    def compute_total_entropy_production(self) -> float:
        """Compute total entropy production across all layers"""
        total = 0.0
        for layer in self.layers:
            total += layer.compute_entropy_production()
        self.state.entropy_production = total
        return total
    
    def step_dynamics(self, dt: float = 0.01):
        """
        Advance the non-equilibrium dynamics by one timestep.
        
        This is where the magic happens - the network is never at rest!
        """
        total_dissipation = 0.0
        
        for layer in self.layers:
            # Inject energy to maintain non-equilibrium
            layer.inject_energy(self.energy_injection_rate * dt)
            
            # Dissipate energy (approach to steady state)
            total_dissipation += layer.dissipate()
            
            # Update weight dynamics
            layer.update_dynamics(dt)
        
        self.state.dissipation = total_dissipation
        self.state.energy += self.energy_injection_rate * dt - total_dissipation
        
        # Record history
        self.entropy_history.append(self.state.entropy_production)
        self.energy_history.append(self.state.energy)
        self.current_history.append(self.state.information_current)
    
    def compute_loss(
        self, 
        output: torch.Tensor, 
        target: torch.Tensor,
        alpha: float = 0.1  # Weight of entropy production term
    ) -> torch.Tensor:
        """
        Compute the dissipative loss function.
        
        Loss = Standard Loss - α × Entropy Production
        
        We SUBTRACT entropy production because we want to MAXIMIZE it!
        """
        # Standard cross-entropy or MSE loss
        if target.dim() == 1 and target.dtype == torch.long:
            standard_loss = F.cross_entropy(output, target)
        else:
            standard_loss = F.mse_loss(output, target)
        
        # Entropy production term (we want to maximize this)
        entropy_prod = self.compute_total_entropy_production()
        
        # Combined loss: minimize standard loss, maximize entropy production
        total_loss = standard_loss - alpha * entropy_prod
        
        return total_loss
    
    def analyze_noise_spectrum(self, num_samples: int = 1000) -> np.ndarray:
        """
        Analyze the noise spectrum of activations.
        
        Biological neural networks show 1/f (pink) noise.
        Standard networks show white noise.
        A successful DLM should show pink noise!
        """
        activations = []
        x = torch.randn(1, self.layers[0].in_features)
        
        for _ in range(num_samples):
            with torch.no_grad():
                out = self.forward(x)
                activations.append(out.mean().item())
            self.step_dynamics(0.01)
        
        # Compute power spectrum
        fft = np.fft.fft(activations)
        power = np.abs(fft) ** 2
        freqs = np.fft.fftfreq(len(activations))
        
        # Return positive frequencies only
        positive = freqs > 0
        return freqs[positive], power[positive]


class StandardNetwork(nn.Module):
    """Standard equilibrium network for comparison"""
    
    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.gelu(x)
        return x


def train_comparison(
    dlm: DissipativeLearningMachine,
    standard: StandardNetwork,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 100,
    lr: float = 0.01
) -> dict:
    """
    Train both networks and compare their behavior.
    """
    X, y = train_data
    
    dlm_optimizer = torch.optim.Adam(dlm.parameters(), lr=lr)
    std_optimizer = torch.optim.Adam(standard.parameters(), lr=lr)
    
    dlm_losses = []
    std_losses = []
    dlm_accuracies = []
    std_accuracies = []
    
    for epoch in range(epochs):
        # Train DLM
        dlm_optimizer.zero_grad()
        dlm_out = dlm(X)
        dlm_loss = dlm.compute_loss(dlm_out, y, alpha=0.1)
        dlm_loss.backward()
        dlm_optimizer.step()
        dlm.step_dynamics()
        
        # Train standard network
        std_optimizer.zero_grad()
        std_out = standard(X)
        std_loss = F.cross_entropy(std_out, y)
        std_loss.backward()
        std_optimizer.step()
        
        # Record metrics
        dlm_losses.append(dlm_loss.item())
        std_losses.append(std_loss.item())
        
        with torch.no_grad():
            dlm_acc = (dlm_out.argmax(dim=1) == y).float().mean().item()
            std_acc = (std_out.argmax(dim=1) == y).float().mean().item()
            dlm_accuracies.append(dlm_acc)
            std_accuracies.append(std_acc)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: DLM acc={dlm_acc:.3f}, Std acc={std_acc:.3f}, "
                  f"Entropy prod={dlm.state.entropy_production:.4f}")
    
    return {
        'dlm_losses': dlm_losses,
        'std_losses': std_losses,
        'dlm_accuracies': dlm_accuracies,
        'std_accuracies': std_accuracies,
        'entropy_history': dlm.entropy_history,
        'energy_history': dlm.energy_history
    }


def test_continual_learning(
    dlm: DissipativeLearningMachine,
    standard: StandardNetwork,
    task1_data: Tuple[torch.Tensor, torch.Tensor],
    task2_data: Tuple[torch.Tensor, torch.Tensor],
    epochs_per_task: int = 50,
    lr: float = 0.01
) -> dict:
    """
    Test catastrophic forgetting by training on two tasks sequentially.
    
    The DLM hypothesis: Non-equilibrium dynamics should reduce forgetting
    because the system never "settles" into a task-specific minimum.
    """
    X1, y1 = task1_data
    X2, y2 = task2_data
    
    dlm_optimizer = torch.optim.Adam(dlm.parameters(), lr=lr)
    std_optimizer = torch.optim.Adam(standard.parameters(), lr=lr)
    
    results = {
        'dlm_task1_before': [], 'dlm_task1_after': [],
        'std_task1_before': [], 'std_task1_after': [],
    }
    
    # Train on task 1
    print("Training on Task 1...")
    for epoch in range(epochs_per_task):
        dlm_optimizer.zero_grad()
        dlm_loss = dlm.compute_loss(dlm(X1), y1)
        dlm_loss.backward()
        dlm_optimizer.step()
        dlm.step_dynamics()
        
        std_optimizer.zero_grad()
        std_loss = F.cross_entropy(standard(X1), y1)
        std_loss.backward()
        std_optimizer.step()
    
    # Record task 1 accuracy before training on task 2
    with torch.no_grad():
        dlm_t1_before = (dlm(X1).argmax(dim=1) == y1).float().mean().item()
        std_t1_before = (standard(X1).argmax(dim=1) == y1).float().mean().item()
    
    print(f"Task 1 accuracy - DLM: {dlm_t1_before:.3f}, Standard: {std_t1_before:.3f}")
    
    # Train on task 2
    print("\nTraining on Task 2...")
    for epoch in range(epochs_per_task):
        dlm_optimizer.zero_grad()
        dlm_loss = dlm.compute_loss(dlm(X2), y2)
        dlm_loss.backward()
        dlm_optimizer.step()
        dlm.step_dynamics()
        
        std_optimizer.zero_grad()
        std_loss = F.cross_entropy(standard(X2), y2)
        std_loss.backward()
        std_optimizer.step()
    
    # Record task 1 accuracy AFTER training on task 2 (forgetting test)
    with torch.no_grad():
        dlm_t1_after = (dlm(X1).argmax(dim=1) == y1).float().mean().item()
        std_t1_after = (standard(X1).argmax(dim=1) == y1).float().mean().item()
        dlm_t2 = (dlm(X2).argmax(dim=1) == y2).float().mean().item()
        std_t2 = (standard(X2).argmax(dim=1) == y2).float().mean().item()
    
    print(f"\nAfter Task 2 training:")
    print(f"  Task 1 retention - DLM: {dlm_t1_after:.3f}, Standard: {std_t1_after:.3f}")
    print(f"  Task 2 accuracy  - DLM: {dlm_t2:.3f}, Standard: {std_t2:.3f}")
    print(f"\nForgetting (Task 1 drop):")
    print(f"  DLM: {dlm_t1_before - dlm_t1_after:.3f}")
    print(f"  Standard: {std_t1_before - std_t1_after:.3f}")
    
    return {
        'dlm_task1_before': dlm_t1_before,
        'dlm_task1_after': dlm_t1_after,
        'std_task1_before': std_t1_before,
        'std_task1_after': std_t1_after,
        'dlm_forgetting': dlm_t1_before - dlm_t1_after,
        'std_forgetting': std_t1_before - std_t1_after
    }


def generate_synthetic_data(
    n_samples: int = 500,
    n_features: int = 20,
    n_classes: int = 5,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic classification data"""
    torch.manual_seed(seed)
    
    X = torch.randn(n_samples, n_features)
    # Create clusters
    centers = torch.randn(n_classes, n_features) * 2
    y = torch.randint(0, n_classes, (n_samples,))
    
    for i in range(n_classes):
        mask = y == i
        X[mask] += centers[i]
    
    return X, y


def visualize_results(results: dict, save_path: Optional[str] = None):
    """Visualize training results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    axes[0, 0].plot(results['dlm_losses'], label='DLM', alpha=0.7)
    axes[0, 0].plot(results['std_losses'], label='Standard', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    
    # Accuracy curves
    axes[0, 1].plot(results['dlm_accuracies'], label='DLM', alpha=0.7)
    axes[0, 1].plot(results['std_accuracies'], label='Standard', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy Comparison')
    axes[0, 1].legend()
    
    # Entropy production
    axes[1, 0].plot(results['entropy_history'], color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Entropy Production')
    axes[1, 0].set_title('DLM Entropy Production Over Time')
    
    # Energy
    axes[1, 1].plot(results['energy_history'], color='blue', alpha=0.7)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Energy')
    axes[1, 1].set_title('DLM System Energy Over Time')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def main():
    """
    Main demonstration of the Dissipative Learning Machine.
    """
    print("=" * 60)
    print("DISSIPATIVE LEARNING MACHINE - Experimental Implementation")
    print("=" * 60)
    print()
    
    # Network architecture
    layer_sizes = [20, 64, 32, 5]
    
    # Create networks
    dlm = DissipativeLearningMachine(
        layer_sizes=layer_sizes,
        temperature=0.5,
        dissipation_rate=0.02,
        energy_injection_rate=0.05
    )
    
    standard = StandardNetwork(layer_sizes=layer_sizes)
    
    print(f"Network architecture: {layer_sizes}")
    print(f"DLM Parameters: T={dlm.temperature}, γ={dlm.layers[0].dissipation_rate}")
    print()
    
    # Generate training data
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=500, n_features=20, n_classes=5)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print()
    
    # Train and compare
    print("=" * 40)
    print("EXPERIMENT 1: Standard Training Comparison")
    print("=" * 40)
    results = train_comparison(dlm, standard, (X, y), epochs=100, lr=0.01)
    
    print()
    print("=" * 40)
    print("EXPERIMENT 2: Continual Learning (Catastrophic Forgetting Test)")
    print("=" * 40)
    
    # Create fresh networks for continual learning test
    dlm2 = DissipativeLearningMachine(
        layer_sizes=layer_sizes,
        temperature=0.5,
        dissipation_rate=0.02,
        energy_injection_rate=0.05
    )
    standard2 = StandardNetwork(layer_sizes=layer_sizes)
    
    # Generate two different tasks
    task1_data = generate_synthetic_data(n_samples=300, n_features=20, n_classes=5, seed=42)
    task2_data = generate_synthetic_data(n_samples=300, n_features=20, n_classes=5, seed=123)
    
    forgetting_results = test_continual_learning(
        dlm2, standard2, task1_data, task2_data, 
        epochs_per_task=50, lr=0.01
    )
    
    print()
    print("=" * 40)
    print("EXPERIMENT 3: Noise Spectrum Analysis")
    print("=" * 40)
    
    freqs, power = dlm.analyze_noise_spectrum(num_samples=500)
    
    # Check for 1/f characteristic
    # In log-log space, 1/f noise should have slope ≈ -1
    log_freqs = np.log10(freqs[1:50])  # Avoid zero frequency
    log_power = np.log10(power[1:50] + 1e-10)
    slope = np.polyfit(log_freqs, log_power, 1)[0]
    
    print(f"Power spectrum slope: {slope:.2f}")
    print(f"(1/f noise would have slope ≈ -1, white noise ≈ 0)")
    
    if slope < -0.5:
        print("→ Network shows pink/1/f-like noise characteristics!")
    else:
        print("→ Network shows more white noise characteristics")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final DLM accuracy: {results['dlm_accuracies'][-1]:.3f}")
    print(f"Final Standard accuracy: {results['std_accuracies'][-1]:.3f}")
    print(f"DLM forgetting: {forgetting_results['dlm_forgetting']:.3f}")
    print(f"Standard forgetting: {forgetting_results['std_forgetting']:.3f}")
    print(f"Noise spectrum slope: {slope:.2f}")
    
    # Save visualization
    import os
    os.makedirs('/Users/jason/projects/wellos2/experiments', exist_ok=True)
    visualize_results(results, save_path='/Users/jason/projects/wellos2/experiments/dlm_results.png')
    
    print()
    print("Visualization saved to experiments/dlm_results.png")


if __name__ == "__main__":
    main()
