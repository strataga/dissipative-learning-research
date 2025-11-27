# Research Roadmap: Dissipative Learning Machines

## Phase 1: Validation (Current → 1 month)

### 1.1 Scale to Real Datasets
- [x] MNIST classification (see results/mnist_comparison.png)
- [ ] CIFAR-10 classification
- [ ] Compare training dynamics, not just final accuracy
- [ ] Measure wall-clock time and compute cost

### 1.2 Rigorous Continual Learning Benchmarks
- [x] Split MNIST (5 tasks, 2 classes each) - see results/tnn_validation.png
  - **Finding**: TNN reduces forgetting by 1.4x vs Standard (0.73 vs 0.997)
  - **Finding**: Best sparsity is 5% active neurons (0.77 forgetting vs ~1.0)
  - **Finding**: Sleep cycles had minimal effect
- [ ] Permuted MNIST (different permutations as tasks)
- [ ] Split CIFAR-100 (10 or 20 tasks)
- [x] Compare against EWC - see results/ewc_comparison.png
  - **MAJOR FINDING**: TNN (5% sparsity) is 35.6% better than best EWC
  - TNN: 0.61 avg forgetting, 48.8% avg accuracy
  - Best EWC (λ=5000): 0.95 avg forgetting, 23.8% avg accuracy
  - Standard/DLM: ~0.998 forgetting (catastrophic)
  - **KEY INSIGHT**: Sparse representations are the primary mechanism, not thermodynamics
- [ ] Compare against:
  - SI (Synaptic Intelligence)
  - PackNet
  - Progressive Neural Networks

### 1.3 Hyperparameter Sensitivity Analysis
- [x] Temperature (T): 0.01 → 10.0 - see results/temperature_sweep.png
  - **CRITICAL FINDING**: Sharp phase transition at T≈0.25-0.5
  - T ≤ 0.25: Frozen network, no forgetting but CAN'T LEARN NEW TASKS
  - T ≥ 0.5: Plastic network, learns new tasks but CATASTROPHIC FORGETTING
  - **Implication**: Classic stability-plasticity dilemma not solved by temperature alone
- [x] Dissipation rate (γ): 0.001 → 0.5 - see results/dissipation_sweep.png
  - **FINDING**: NO EFFECT on forgetting. All γ values show ~100% forgetting
  - Entropy production = 0 across all settings (implementation issue?)
  - DLM dynamics alone don't prevent catastrophic forgetting
- [ ] Energy injection rate: 0.01 → 1.0 (deprioritized - dissipation had no effect)
- [ ] Current strength (α): 0.01 → 1.0
- [x] Document stable operating regimes - T=0.05-0.25 stable but frozen

### 1.4 Noise Spectrum Deep Dive
- [ ] Measure at multiple layers
- [ ] Compare to biological recordings
- [ ] Analyze during training vs inference
- [ ] Correlate with forgetting resistance

---

## Phase 2: Theory Development (1-3 months)

### 2.1 Mathematical Framework
- [ ] Define proper entropy production functional
- [ ] Prove bounds on forgetting rate under DLM dynamics
- [ ] Characterize steady-state distributions
- [ ] Connection to optimal transport theory

### 2.2 Information-Theoretic Analysis
- [ ] Mutual information between layers
- [ ] Information bottleneck under non-equilibrium
- [ ] Compression-accuracy tradeoffs

### 2.3 Dynamical Systems Analysis
- [ ] Characterize attractors and limit cycles
- [ ] Lyapunov exponents (edge of chaos?)
- [ ] Bifurcation analysis as T varies
- [ ] Connection to criticality in biological networks

### 2.4 Write Theoretical Paper
- [ ] "Non-Equilibrium Thermodynamics of Neural Network Learning"
- [ ] Target: NeurIPS, ICML, or Physical Review X

---

## Phase 3: Architecture Innovations (3-6 months)

### 3.1 Dissipative Transformers
- [ ] Apply DLM principles to attention mechanism
- [ ] Information currents through attention heads
- [ ] Temperature-modulated attention
- [ ] Test on language modeling

### 3.2 Dissipative Convolutional Networks
- [ ] Spatial information currents
- [ ] Local vs global dissipation
- [ ] Feature map energy landscapes

### 3.3 Hierarchical Energy Flows
- [ ] Multi-scale energy injection
- [ ] Cross-layer currents
- [ ] Renormalization group perspective

### 3.4 Recurrent Dissipative Networks
- [ ] Natural fit for temporal dynamics
- [ ] Memory as sustained currents
- [ ] Comparison to LSTMs/GRUs

---

## Phase 4: Efficiency & Hardware (6-12 months)

### 4.1 Energy Efficiency Analysis
- [ ] Joules per inference
- [ ] Compare to standard networks at iso-accuracy
- [ ] Theoretical minimum (Landauer limit analysis)

### 4.2 Sparse Dissipative Networks
- [ ] Pruning in non-equilibrium context
- [ ] Current-based importance scores
- [ ] Dynamic sparsity

### 4.3 Neuromorphic Implementation
- [ ] Map to spiking neural networks
- [ ] Loihi / BrainScaleS compatibility
- [ ] Event-driven dissipation

### 4.4 Analog Hardware Design
- [ ] Resistive elements for dissipation
- [ ] Noise injection circuits
- [ ] Current measurement

---

## Phase 5: Applications (12+ months)

### 5.1 Lifelong Learning Systems
- [ ] Robot learning with continuous adaptation
- [ ] Personalized AI that doesn't forget
- [ ] Online learning from streams

### 5.2 Scientific Discovery
- [ ] Physics-informed DLM
- [ ] Molecular dynamics integration
- [ ] Climate modeling

### 5.3 Neuroscience Collaboration
- [ ] Compare DLM dynamics to neural recordings
- [ ] Predict experimental signatures
- [ ] Joint theory-experiment paper

---

## Key Experiments Queue

### Immediate (This Week)
1. MNIST experiment with current implementation
2. Sweep temperature parameter
3. Longer continual learning sequence (5+ tasks)

### Short-term (This Month)
1. CIFAR-10 implementation
2. EWC comparison implementation
3. Multi-layer noise spectrum analysis

### Medium-term (Next Quarter)
1. Transformer integration
2. Theoretical bounds derivation
3. First paper draft

---

## Success Metrics

| Milestone | Target | Status |
|-----------|--------|--------|
| Synthetic data PoC | ✓ Working | ✅ Done |
| MNIST ≥95% accuracy | Match standard | ✅ Done (97%+) |
| Forgetting <50% of standard | Measured | ✅ Done (Sparse+EWC: 68% reduction) |
| Pink noise signature | Slope < -0.5 | ✅ Done (-1.1 to -1.8) |
| Temperature sweep | Document regimes | ✅ Done (phase transition at T≈0.3) |
| Ultra-low sparsity | Test 1-3% | ✅ Done (1% optimal: 0.39 forgetting) |
| Sparse+EWC combo | Combine methods | ✅ Done (43% improvement, 0.32 forgetting) |
| CIFAR-10 | Scale up | ⚠️ Partial (benefit less dramatic) |
| First paper submission | Q1 2025 | Pending |
| Neuromorphic prototype | Q3 2025 | Pending |

---

## Open Questions

1. **Is there an optimal "temperature" for learning?** ✅ ANSWERED: No single optimal T exists. Sharp phase transition at T≈0.25-0.5 creates stability-plasticity tradeoff. Need dynamic/annealing approach.
2. **Can information currents be made learnable?** (Meta-learning the dynamics)
3. **What's the connection to attention?** (Attention as routing current)
4. **Does the brain actually maximize entropy production?** (Testable!)
5. **Can we derive backprop as a special case?** (Equilibrium limit)
6. **NEW: How to achieve plasticity AND stability?** Temperature alone insufficient. Consider:
   - Task-aware temperature scheduling (anneal during each task)
   - Separate consolidation pathway
   - Gating mechanisms for new vs consolidated knowledge
7. **ANSWERED: Why does sparsity work?** - see results/sparsity_analysis.png
   - Sparse representations create ORTHOGONAL task embeddings
   - Correlation: overlap vs forgetting r=0.89, p=0.017 (SIGNIFICANT)
   - 5% sparsity: 21 neurons/task, 13% overlap, 0.85 forgetting
   - 100% active: 256 neurons/task, 100% overlap, 1.0 forgetting
   - **KEY INSIGHT**: Sparse coding is the primary mechanism, not thermodynamics
   - This aligns with neuroscience: brain uses sparse distributed representations

---

## Resources Needed

- GPU access for scaled experiments (A100 or better)
- Neuromorphic hardware access (Intel Loihi, IBM TrueNorth)
- Collaboration with:
  - Statistical physics group
  - Computational neuroscience lab
  - ML theory researchers

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Doesn't scale | Medium | Multiple architecture variants |
| No theoretical grounding | Low | Strong physics foundation |
| Scooped | Medium | Move fast, publish incrementally |
| Hardware limitations | High | Start with software, plan hardware |

---

## Timeline Summary

```
Month 1-3:   Validation + Theory foundations
Month 3-6:   Architecture innovations + First paper
Month 6-12:  Efficiency + Hardware exploration
Month 12+:   Applications + Impact
```

The goal: **Establish non-equilibrium thermodynamics as a principled framework for neural network design.**
