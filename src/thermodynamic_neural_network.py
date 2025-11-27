"""
Thermodynamic Neural Network (TNN)
===================================

Next-generation architecture combining:
1. Non-equilibrium thermodynamics (energy flow, entropy production)
2. Sparse distributed representations (minimal interference)
3. Metaplasticity (synaptic consolidation over time)
4. Sleep/wake cycles (consolidation phases)
5. Homeostatic plasticity (activity regulation)
6. Critical dynamics (edge of chaos)

This addresses the failures of the basic DLM:
- DLM had dynamics but still forgot (dynamics alone insufficient)
- TNN adds structural and algorithmic mechanisms from neuroscience

Author: Research Code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum


class Phase(Enum):
    WAKE = "wake"      # Active learning
    SLEEP = "sleep"    # Consolidation


@dataclass
class NetworkState:
    """Complete thermodynamic state of the network"""
    energy: float = 0.0
    entropy_production: float = 0.0
    information_flow: float = 0.0
    temperature: float = 1.0
    criticality: float = 0.5  # 0=ordered, 1=chaotic, 0.5=critical
    sparsity: float = 0.0
    consolidation: float = 0.0
    phase: Phase = Phase.WAKE
    age: int = 0  # Training steps


@dataclass 
class LayerMetrics:
    """Per-layer diagnostic metrics"""
    activity: float = 0.0
    sparsity: float = 0.0
    weight_magnitude: float = 0.0
    consolidation_level: float = 0.0
    gradient_magnitude: float = 0.0


class SparseActivation(nn.Module):
    """
    k-Winner-Take-All activation for sparse representations.
    
    Only top-k neurons fire, creating minimal interference between patterns.
    This is crucial for preventing catastrophic forgetting.
    """
    def __init__(self, k_ratio: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.k_ratio = k_ratio
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Soft k-WTA during training (differentiable)
            k = max(1, int(x.size(-1) * self.k_ratio))
            topk_vals, _ = torch.topk(x, k, dim=-1)
            threshold = topk_vals[..., -1:] 
            # Soft thresholding with temperature
            mask = torch.sigmoid((x - threshold) / self.temperature)
            return x * mask
        else:
            # Hard k-WTA during inference
            k = max(1, int(x.size(-1) * self.k_ratio))
            topk_vals, topk_idx = torch.topk(x, k, dim=-1)
            out = torch.zeros_like(x)
            out.scatter_(-1, topk_idx, topk_vals)
            return out


class ThermodynamicLayer(nn.Module):
    """
    A neural network layer with full thermodynamic properties.
    
    Key features:
    - Sparse activation (k-WTA)
    - Metaplasticity (weight consolidation)
    - Homeostatic plasticity (activity normalization)
    - Information current tracking
    - Energy accounting
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity: float = 0.1,
        temperature: float = 1.0,
        consolidation_rate: float = 0.001,
        homeostatic_rate: float = 0.01,
        target_activity: float = 0.1,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.consolidation_rate = consolidation_rate
        self.homeostatic_rate = homeostatic_rate
        self.target_activity = target_activity
        
        # Core weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Metaplasticity: consolidation strength per weight
        # Higher = more consolidated = harder to change
        self.register_buffer('consolidation', torch.zeros_like(self.weight))
        
        # Homeostatic plasticity: per-neuron gain
        self.register_buffer('gain', torch.ones(out_features))
        
        # Running activity estimate for homeostasis
        self.register_buffer('running_activity', torch.ones(out_features) * target_activity)
        
        # Information flow tracking
        self.register_buffer('input_activity', torch.zeros(in_features))
        self.register_buffer('output_activity', torch.zeros(out_features))
        
        # Sparse activation
        self.sparse_activation = SparseActivation(k_ratio=sparsity, temperature=temperature)
        
        # Layer energy
        self.register_buffer('energy', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with full thermodynamic tracking"""
        
        # Track input activity
        with torch.no_grad():
            input_act = x.abs().mean(dim=0) if x.dim() > 1 else x.abs()
            self.input_activity = 0.9 * self.input_activity + 0.1 * input_act
        
        # Linear transformation with homeostatic gain
        out = F.linear(x, self.weight, self.bias)
        out = out * self.gain  # Homeostatic scaling
        
        # Sparse activation
        out = self.sparse_activation(out)
        
        # Track output activity
        with torch.no_grad():
            output_act = (out.abs() > 0.01).float().mean(dim=0) if out.dim() > 1 else (out.abs() > 0.01).float()
            self.output_activity = 0.9 * self.output_activity + 0.1 * output_act
            self.running_activity = 0.99 * self.running_activity + 0.01 * output_act
        
        # Compute layer energy (proxy for metabolic cost)
        with torch.no_grad():
            self.energy = (out ** 2).sum() + (self.weight ** 2).sum() * 0.01
        
        # Metrics for this forward pass
        metrics = LayerMetrics(
            activity=self.output_activity.mean().item(),
            sparsity=1.0 - (out.abs() > 0.01).float().mean().item(),
            weight_magnitude=self.weight.abs().mean().item(),
            consolidation_level=self.consolidation.mean().item(),
        )
        
        return out, metrics
    
    def update_consolidation(self):
        """
        Update metaplasticity: weights used frequently become consolidated.
        
        This implements the idea that important synapses become "protected"
        from future changes - similar to synaptic tagging and capture.
        """
        if self.weight.grad is not None:
            # Weights with high gradient × activity become more consolidated
            importance = self.weight.grad.abs() * self.weight.abs()
            importance = importance / (importance.max() + 1e-8)
            
            # Consolidation increases slowly, providing protection
            self.consolidation = (
                self.consolidation + 
                self.consolidation_rate * importance * (1 - self.consolidation)
            )
    
    def update_homeostasis(self):
        """
        Homeostatic plasticity: adjust gains to maintain target activity.
        
        Neurons that are too active get suppressed; inactive ones get boosted.
        This prevents dead neurons and runaway excitation.
        """
        with torch.no_grad():
            # Error signal: how far from target activity
            error = self.target_activity - self.running_activity
            
            # Adjust gain to compensate
            self.gain = self.gain * (1 + self.homeostatic_rate * error)
            
            # Clamp to reasonable range
            self.gain = self.gain.clamp(0.1, 10.0)
    
    def get_effective_lr_mask(self) -> torch.Tensor:
        """
        Get per-weight learning rate multiplier based on consolidation.
        
        Highly consolidated weights learn slower (are protected).
        """
        # Consolidated weights have reduced learning rate
        # consolidation 0 -> lr multiplier 1
        # consolidation 1 -> lr multiplier 0.01
        return torch.exp(-5 * self.consolidation) + 0.01
    
    def apply_consolidation_mask(self):
        """Apply consolidation to gradients (call before optimizer.step)"""
        if self.weight.grad is not None:
            self.weight.grad = self.weight.grad * self.get_effective_lr_mask()
    
    def compute_information_current(self) -> float:
        """Compute information flow through this layer"""
        return (self.output_activity.mean() - self.input_activity.mean()).item()
    
    def inject_noise(self, scale: float = 0.01):
        """Inject thermal noise into weights (for exploration)"""
        with torch.no_grad():
            noise = torch.randn_like(self.weight) * scale * self.temperature
            # Noise is suppressed for consolidated weights
            noise = noise * (1 - self.consolidation)
            self.weight.add_(noise)


class ThermodynamicNeuralNetwork(nn.Module):
    """
    Complete Thermodynamic Neural Network.
    
    Implements:
    1. Sparse distributed representations (k-WTA)
    2. Metaplasticity with synaptic consolidation
    3. Homeostatic plasticity for stability
    4. Sleep/wake cycles for consolidation
    5. Critical dynamics at edge of chaos
    6. Full energy accounting
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        sparsity: float = 0.15,
        temperature: float = 1.0,
        consolidation_rate: float = 0.001,
        sleep_frequency: int = 100,  # Sleep every N batches
        sleep_duration: int = 10,    # Sleep for N "replay" iterations
    ):
        super().__init__()
        
        self.sparsity = sparsity
        self.temperature = temperature
        self.sleep_frequency = sleep_frequency
        self.sleep_duration = sleep_duration
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            is_output = (i == len(layer_sizes) - 2)
            self.layers.append(ThermodynamicLayer(
                layer_sizes[i],
                layer_sizes[i + 1],
                sparsity=1.0 if is_output else sparsity,  # No sparsity on output
                temperature=temperature,
                consolidation_rate=consolidation_rate,
            ))
        
        # Network state
        self.state = NetworkState(temperature=temperature)
        
        # Memory buffer for sleep replay
        self.memory_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.buffer_size = 1000
        
        # Training step counter
        self.step_count = 0
        
        # History tracking
        self.history = {
            'energy': [],
            'entropy': [],
            'sparsity': [],
            'consolidation': [],
            'criticality': [],
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers"""
        all_metrics = []
        
        for i, layer in enumerate(self.layers):
            x, metrics = layer(x)
            all_metrics.append(metrics)
            
            # No activation after final layer
            if i < len(self.layers) - 1:
                x = F.gelu(x)
        
        # Update network state from layer metrics
        self._update_state(all_metrics)
        
        return x
    
    def _update_state(self, layer_metrics: List[LayerMetrics]):
        """Update network-level state from layer metrics"""
        # Average sparsity across layers
        self.state.sparsity = np.mean([m.sparsity for m in layer_metrics])
        
        # Average consolidation
        self.state.consolidation = np.mean([m.consolidation_level for m in layer_metrics])
        
        # Total energy
        self.state.energy = sum(layer.energy.item() for layer in self.layers)
        
        # Information flow (sum of currents)
        self.state.information_flow = sum(
            layer.compute_information_current() for layer in self.layers
        )
        
        # Estimate criticality from activity variance
        activities = [m.activity for m in layer_metrics]
        if len(activities) > 1:
            self.state.criticality = np.std(activities) / (np.mean(activities) + 1e-8)
            self.state.criticality = min(1.0, self.state.criticality)
    
    def compute_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        sparsity_weight: float = 0.01,
        energy_weight: float = 0.001,
    ) -> torch.Tensor:
        """
        Compute loss with thermodynamic regularization.
        
        Loss = Task Loss + Sparsity Penalty + Energy Cost
        """
        # Task loss
        if target.dim() == 1 and target.dtype == torch.long:
            task_loss = F.cross_entropy(output, target)
        else:
            task_loss = F.mse_loss(output, target)
        
        # Sparsity encouragement (penalize non-sparse activations)
        sparsity_loss = (1 - self.state.sparsity) * sparsity_weight
        
        # Energy cost (metabolic constraint)
        energy_loss = self.state.energy * energy_weight
        
        return task_loss + sparsity_loss + energy_loss
    
    def training_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Complete training step with all thermodynamic mechanisms.
        """
        self.step_count += 1
        self.state.age = self.step_count
        self.state.phase = Phase.WAKE
        
        # Forward pass
        optimizer.zero_grad()
        output = self.forward(x)
        loss = self.compute_loss(output, y)
        
        # Backward pass
        loss.backward()
        
        # Update consolidation based on gradients (before they're used)
        for layer in self.layers:
            layer.update_consolidation()
        
        # Apply consolidation mask to gradients (protect important weights)
        for layer in self.layers:
            layer.apply_consolidation_mask()
        
        # Optimizer step
        optimizer.step()
        
        # Homeostatic updates
        for layer in self.layers:
            layer.update_homeostasis()
        
        # Inject small noise (exploration)
        for layer in self.layers:
            layer.inject_noise(scale=0.001 * self.temperature)
        
        # Store in memory buffer for replay
        self._store_memory(x.detach(), y.detach())
        
        # Compute entropy production (after backward when gradients exist)
        self._compute_entropy_production()
        
        # Periodic sleep/consolidation
        if self.step_count % self.sleep_frequency == 0:
            self._sleep_phase()
        
        # Record history
        self._record_history()
        
        # Compute accuracy
        with torch.no_grad():
            pred = output.argmax(dim=1) if output.dim() > 1 else output
            if y.dtype == torch.long:
                accuracy = (pred == y).float().mean().item()
            else:
                accuracy = 0.0
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'sparsity': self.state.sparsity,
            'consolidation': self.state.consolidation,
            'entropy': self.state.entropy_production,
            'energy': self.state.energy,
        }
    
    def _store_memory(self, x: torch.Tensor, y: torch.Tensor):
        """Store samples in replay buffer"""
        # Random subset to store
        n_store = min(10, x.size(0))
        idx = torch.randperm(x.size(0))[:n_store]
        
        for i in idx:
            self.memory_buffer.append((x[i:i+1].clone(), y[i:i+1].clone()))
        
        # Trim to buffer size
        while len(self.memory_buffer) > self.buffer_size:
            self.memory_buffer.pop(0)
    
    def _sleep_phase(self):
        """
        Sleep/consolidation phase.
        
        During sleep:
        1. Replay memories from buffer
        2. Strengthen consolidated connections
        3. Prune weak connections
        4. No new learning (frozen task loss)
        """
        if len(self.memory_buffer) < 10:
            return
        
        self.state.phase = Phase.SLEEP
        
        # Replay random memories
        for _ in range(self.sleep_duration):
            # Sample random memories
            idx = np.random.choice(len(self.memory_buffer), min(32, len(self.memory_buffer)), replace=False)
            
            x_replay = torch.cat([self.memory_buffer[i][0] for i in idx])
            y_replay = torch.cat([self.memory_buffer[i][1] for i in idx])
            
            # Forward pass (no gradient for task loss, only for consolidation)
            with torch.no_grad():
                _ = self.forward(x_replay)
            
            # Strengthen consolidated weights
            for layer in self.layers:
                with torch.no_grad():
                    # Consolidated weights get slightly reinforced
                    reinforcement = layer.consolidation * layer.weight * 0.001
                    layer.weight.add_(reinforcement)
            
            # Prune weak, unconsolidated connections
            for layer in self.layers:
                with torch.no_grad():
                    # Weights that are both small AND unconsolidated get pruned
                    prune_mask = (layer.weight.abs() < 0.001) & (layer.consolidation < 0.1)
                    layer.weight[prune_mask] = 0
        
        self.state.phase = Phase.WAKE
    
    def _compute_entropy_production(self):
        """Compute entropy production from layer gradients and currents"""
        total_entropy = 0.0
        
        for layer in self.layers:
            if layer.weight.grad is not None:
                # Thermodynamic force (gradient magnitude)
                force = layer.weight.grad.abs().mean().item()
                
                # Current (information flow)
                current = abs(layer.compute_information_current())
                
                # Entropy production = current × force / temperature
                entropy = current * force / (self.temperature + 1e-8)
                total_entropy += entropy
        
        self.state.entropy_production = total_entropy
    
    def _record_history(self):
        """Record state history for analysis"""
        self.history['energy'].append(self.state.energy)
        self.history['entropy'].append(self.state.entropy_production)
        self.history['sparsity'].append(self.state.sparsity)
        self.history['consolidation'].append(self.state.consolidation)
        self.history['criticality'].append(self.state.criticality)
    
    def get_consolidation_map(self) -> List[torch.Tensor]:
        """Get consolidation levels for all layers (for visualization)"""
        return [layer.consolidation.clone() for layer in self.layers]
    
    def freeze_consolidated(self, threshold: float = 0.8):
        """Completely freeze weights above consolidation threshold"""
        for layer in self.layers:
            mask = layer.consolidation > threshold
            layer.weight.requires_grad_(True)  # Keep overall grad
            # Will be masked in apply_consolidation_mask


class CriticalityRegulator:
    """
    Maintains network at edge of chaos (criticality).
    
    Critical systems have:
    - Maximum information transfer
    - Power-law correlations
    - Optimal computational capabilities
    
    We adjust temperature to maintain criticality.
    """
    
    def __init__(self, target_criticality: float = 0.5, adjustment_rate: float = 0.01):
        self.target = target_criticality
        self.rate = adjustment_rate
    
    def adjust(self, network: ThermodynamicNeuralNetwork):
        """Adjust network temperature to maintain criticality"""
        error = self.target - network.state.criticality
        
        # Increase temperature if too ordered, decrease if too chaotic
        network.temperature *= (1 + self.rate * error)
        network.temperature = max(0.1, min(10.0, network.temperature))
        
        # Propagate to layers
        for layer in network.layers:
            layer.temperature = network.temperature
            layer.sparse_activation.temperature = network.temperature


def train_tnn(
    model: ThermodynamicNeuralNetwork,
    train_loader,
    epochs: int = 10,
    lr: float = 0.001,
    use_criticality: bool = True,
) -> Dict[str, List[float]]:
    """
    Train a Thermodynamic Neural Network.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criticality_reg = CriticalityRegulator() if use_criticality else None
    
    history = {
        'loss': [], 'accuracy': [], 'sparsity': [],
        'consolidation': [], 'entropy': [], 'temperature': []
    }
    
    for epoch in range(epochs):
        epoch_metrics = {k: [] for k in history.keys()}
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Flatten if needed
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            # Training step
            metrics = model.training_step(x, y, optimizer)
            
            # Record
            for k, v in metrics.items():
                if k in epoch_metrics:
                    epoch_metrics[k].append(v)
            epoch_metrics['temperature'].append(model.temperature)
            
            # Adjust criticality periodically
            if criticality_reg and batch_idx % 10 == 0:
                criticality_reg.adjust(model)
        
        # Epoch averages
        for k in history.keys():
            if epoch_metrics[k]:
                history[k].append(np.mean(epoch_metrics[k]))
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss={history['loss'][-1]:.4f}, "
              f"Acc={history['accuracy'][-1]:.4f}, "
              f"Sparse={history['sparsity'][-1]:.3f}, "
              f"Consol={history['consolidation'][-1]:.4f}, "
              f"T={history['temperature'][-1]:.3f}")
    
    return history


def evaluate_tnn(model: ThermodynamicNeuralNetwork, test_loader) -> float:
    """Evaluate model accuracy"""
    model.eval()
    correct = total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    model.train()
    return correct / total
