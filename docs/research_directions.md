# Research Directions: Dissipative Learning Machines

**Date**: November 2024  
**Status**: Phase 1 Complete, Planning Next Steps

---

## Current State Summary

### What We Know
1. **Sparse coding works** - 1% sparsity reduces forgetting by 61% (0.39 vs 1.0)
2. **Sparse + EWC is best** - 68% reduction in forgetting (0.32)
3. **Thermodynamic dynamics show no effect** - Dissipation rate sweep was flat
4. **Entropy calculation broken** - Always returns 0
5. **Temperature has phase transition** - Not a tunable parameter

### Open Questions
1. Is the thermodynamic hypothesis wrong, or just our implementation?
2. Could thermodynamics help with properties OTHER than forgetting?
3. What's the theoretical basis for why sparsity works?

---

## Direction A: Pivot to Sparse Coding

**Thesis**: Abandon thermodynamic framing. Focus on sparse distributed representations as the mechanism for continual learning.

### Rationale
- Sparse coding is well-established in neuroscience
- Clear theoretical basis (orthogonal representations)
- Strong empirical results (68% forgetting reduction)
- Simpler to explain and implement

### Research Plan

#### A.1 Theoretical Foundation
| Task | Description | Timeline |
|------|-------------|----------|
| Information theory analysis | Prove capacity bounds for sparse representations | 2 weeks |
| Interference analysis | Formalize relationship between overlap and forgetting | 1 week |
| Optimal sparsity derivation | Derive optimal k for k-WTA given task complexity | 2 weeks |

#### A.2 Empirical Validation
| Task | Description | Timeline |
|------|-------------|----------|
| Permuted MNIST | Standard benchmark (10+ tasks) | 3 days |
| Split CIFAR-100 | 20-task benchmark | 1 week |
| Sequence learning | Test on sequential prediction tasks | 1 week |
| Comparison suite | SI, PackNet, Progressive Nets, A-GEM | 2 weeks |

#### A.3 Architecture Development
| Task | Description | Timeline |
|------|-------------|----------|
| Sparse Transformers | k-WTA in attention mechanism | 2 weeks |
| Adaptive sparsity | Learn optimal sparsity per layer/task | 2 weeks |
| Sparse convolutions | Extend to CNN architectures properly | 1 week |

#### A.4 Paper Writing
| Target | Title | Timeline |
|--------|-------|----------|
| NeurIPS/ICML | "Sparse Distributed Representations for Continual Learning" | 6 weeks |
| ICLR | "Orthogonal Task Embeddings via Extreme Sparsity" | 8 weeks |

### Expected Outcomes
- Novel continual learning method competitive with SOTA
- Clear theoretical understanding
- Practical implementation guidelines

### Risks
- May be incremental (sparse coding is known)
- Competition from other sparse coding papers
- Loses the "thermodynamic" novelty angle

---

## Direction B: Debug & Salvage Thermodynamics

**Thesis**: The thermodynamic hypothesis is correct but our implementation is flawed. Fix bugs and properly test.

### Rationale
- Entropy calculation returns 0 (clear bug)
- Never properly tested thermodynamic loss terms
- Biological plausibility of non-equilibrium dynamics
- Unique research angle vs. standard ML approaches

### Research Plan

#### B.1 Implementation Debugging
| Task | Description | Priority |
|------|-------------|----------|
| Fix entropy calculation | Debug `_compute_entropy_production()` | HIGH |
| Verify weight dynamics | Ensure `step_dynamics()` modifies weights | HIGH |
| Energy tracking | Implement proper energy accounting | HIGH |
| Visualization | Plot thermodynamic quantities during training | MEDIUM |

#### B.2 Thermodynamic Loss Functions
| Experiment | Description | Hypothesis |
|------------|-------------|------------|
| Entropy regularization | Add entropy production to loss | More exploration |
| Free energy minimization | Minimize F = E - TS | Better representations |
| Dissipation penalty | Penalize high dissipation | More efficient learning |
| Information current loss | Maximize information flow | Better gradient flow |

#### B.3 Proper Thermodynamic Tests
| Experiment | Description | Expected Result |
|------------|-------------|-----------------|
| Entropy vs forgetting | Correlate entropy production with retention | Negative correlation |
| Energy landscape | Visualize loss landscape with/without dynamics | Flatter minima |
| Fluctuation analysis | Measure weight fluctuations over training | Power-law spectrum |
| Steady-state analysis | Characterize long-term dynamics | Non-equilibrium steady state |

#### B.4 Timescale Experiments
| Experiment | Description | Rationale |
|------------|-------------|-----------|
| Long training | 100x epochs, measure thermodynamic effects | Effects may be slow |
| Continuous learning | Online learning over 1000+ tasks | Real continual scenario |
| Annealing schedules | Vary T during training | Simulated annealing connection |

### Expected Outcomes
- Working thermodynamic neural network
- Clear evidence for/against hypothesis
- Novel training dynamics

### Risks
- May confirm thermodynamics doesn't help
- Time-intensive debugging
- Complex to explain if it works

---

## Direction C: Alternative Thermodynamic Frameworks

**Thesis**: Our specific thermodynamic formulation is wrong, but other thermodynamic principles may work.

### C.1 Free Energy Principle (Friston)

**Concept**: Biological systems minimize variational free energy (surprise).

| Component | Implementation |
|-----------|----------------|
| Generative model | Decoder network predicting inputs |
| Recognition model | Encoder inferring latents |
| Free energy | Reconstruction + KL divergence |
| Active inference | Actions that reduce expected surprise |

**Experiments**:
- Implement variational autoencoder with free energy objective
- Test continual learning with free energy minimization
- Compare to standard VAE + EWC

**Timeline**: 3 weeks

---

### C.2 Maximum Entropy Production (MEP)

**Concept**: Far-from-equilibrium systems maximize entropy production rate.

| Component | Implementation |
|-----------|----------------|
| Entropy production | σ = Σ Jᵢ Xᵢ (currents × forces) |
| Currents | Information flow between layers |
| Forces | Gradient magnitudes |
| MEP loss | Maximize σ subject to task performance |

**Experiments**:
- Implement proper entropy production calculation
- Use MEP as auxiliary training objective
- Test if MEP correlates with generalization

**Timeline**: 2 weeks

---

### C.3 Stochastic Thermodynamics

**Concept**: Apply fluctuation theorems to learning trajectories.

| Component | Implementation |
|-----------|----------------|
| Trajectory entropy | S[θ(t)] over training path |
| Jarzynski equality | ⟨e^(-W/kT)⟩ = e^(-ΔF/kT) |
| Work | W = ∫ (∂L/∂θ) dθ |
| Free energy difference | ΔF between task distributions |

**Experiments**:
- Track learning trajectories
- Verify Jarzynski equality holds
- Use fluctuation theorems for task transfer

**Timeline**: 4 weeks

---

### C.4 Hopfield Energy Networks

**Concept**: Modern continuous Hopfield networks with energy-based dynamics.

| Component | Implementation |
|-----------|----------------|
| Energy function | E(x) = -½ Σ ξᵢ·softmax(βξᵢ·x) |
| Dynamics | dx/dt = -∂E/∂x |
| Associative memory | Patterns as attractors |
| Continual learning | Add patterns without catastrophic forgetting |

**Experiments**:
- Implement modern Hopfield layer
- Test memory capacity scaling
- Compare to transformer attention (known connection)

**Timeline**: 3 weeks

---

### C.5 Information Thermodynamics

**Concept**: Apply Landauer's principle to neural computation.

| Component | Implementation |
|-----------|----------------|
| Bit erasure | Forgetting = entropy increase |
| Landauer cost | E ≥ kT ln(2) per bit erased |
| Reversible computing | Minimize information loss |
| Memory cost | Track bits stored vs forgotten |

**Experiments**:
- Measure information content of weights
- Track "bit erasure" during learning
- Design reversible neural operations

**Timeline**: 3 weeks

---

### C.6 Critical Dynamics

**Concept**: Optimal computation occurs at edge of chaos (criticality).

| Component | Implementation |
|-----------|----------------|
| Order parameter | Activity correlation length |
| Criticality measure | Power-law correlations, χ ~ L^γ |
| Branching ratio | σ = 1 at criticality |
| Self-organized criticality | Dynamics that tune to critical point |

**Experiments**:
- Measure branching ratio during training
- Test if critical networks generalize better
- Implement SOC mechanisms

**Timeline**: 3 weeks

---

## Direction D: Hybrid Approaches

**Thesis**: Combine sparse coding with thermodynamic principles.

### D.1 Thermodynamic Sparsity

**Concept**: Use thermodynamics to LEARN optimal sparsity.

| Approach | Description |
|----------|-------------|
| Temperature-dependent sparsity | k(T) varies with temperature |
| Energy-based pruning | Prune based on energy contribution |
| Entropy-maximizing sparsity | Choose k to maximize entropy |

---

### D.2 Free Energy Sparse Coding

**Concept**: Sparse coding as free energy minimization.

| Component | Description |
|-----------|-------------|
| Energy | Reconstruction error |
| Entropy | Sparsity prior (-Σ p log p) |
| Free energy | F = E - TS |
| Optimization | Minimize F, not just E |

---

### D.3 Non-Equilibrium Sparse Representations

**Concept**: Maintain sparsity through active dynamics, not constraints.

| Approach | Description |
|----------|-------------|
| Competitive dynamics | Neurons compete via lateral inhibition |
| Energy injection | Prevent collapse to trivial solutions |
| Dissipative selection | Weak neurons decay, strong persist |

---

## Recommended Research Timeline

### Phase 1: Immediate (Weeks 1-2)
| Priority | Task | Direction |
|----------|------|-----------|
| 1 | Fix entropy calculation bug | B |
| 2 | Re-run thermodynamic experiments | B |
| 3 | Permuted MNIST benchmark | A |
| 4 | Document negative results properly | All |

### Phase 2: Short-term (Weeks 3-6)
| Priority | Task | Direction |
|----------|------|-----------|
| 1 | Thermodynamic loss functions | B |
| 2 | Free Energy Principle implementation | C.1 |
| 3 | Theoretical sparsity analysis | A |
| 4 | Split CIFAR-100 benchmark | A |

### Phase 3: Medium-term (Weeks 7-12)
| Priority | Task | Direction |
|----------|------|-----------|
| 1 | Paper draft (best results) | A or B |
| 2 | Modern Hopfield implementation | C.4 |
| 3 | Hybrid sparse-thermodynamic | D |
| 4 | Critical dynamics analysis | C.6 |

### Phase 4: Long-term (Months 4-6)
| Priority | Task | Direction |
|----------|------|-----------|
| 1 | Conference submission | A or B |
| 2 | Neuromorphic prototype | B |
| 3 | Stochastic thermodynamics | C.3 |
| 4 | Information thermodynamics | C.5 |

---

## Decision Framework

### Choose Direction A (Sparse Coding) if:
- Want fastest path to publication
- Prefer empirical over theoretical work
- Risk-averse (known to work)

### Choose Direction B (Debug Thermodynamics) if:
- Believe in original hypothesis
- Want to understand WHY it didn't work
- Have patience for debugging

### Choose Direction C (Alternative Frameworks) if:
- Want novel theoretical contribution
- Interested in physics-ML intersection
- Willing to explore uncharted territory

### Choose Direction D (Hybrid) if:
- Want best of both worlds
- Results from A and B are both positive
- Have resources for larger research program

---

## Resource Requirements

| Direction | Compute | Time | Risk | Novelty |
|-----------|---------|------|------|---------|
| A: Sparse | Low | 6-8 weeks | Low | Medium |
| B: Debug | Low | 4-6 weeks | Medium | Medium |
| C: Alternative | Medium | 8-12 weeks | High | High |
| D: Hybrid | Medium | 12+ weeks | Medium | High |

---

## Success Metrics

| Metric | Target | Direction |
|--------|--------|-----------|
| Forgetting reduction | >70% vs standard | A, D |
| SOTA comparison | Within 5% of best | A |
| Thermodynamic correlation | p < 0.01 | B, C |
| Paper acceptance | Top venue | All |
| Theoretical insight | Novel theorem | A, C |

---

## Next Steps

1. **Decide primary direction** (A, B, C, or D)
2. **Run Phase 1 experiments** (2 weeks)
3. **Evaluate results** and adjust plan
4. **Begin paper writing** (week 4)
5. **Submit to conference** (week 8-12)
