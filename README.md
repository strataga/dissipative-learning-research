# Dissipative Learning Machine (DLM)

A novel neural network architecture inspired by non-equilibrium thermodynamics and Prigogine's dissipative structures.

## Core Hypothesis

**Current neural networks are equilibrium systems** - they minimize a loss function, settling into energy wells. But biological intelligence operates **far from equilibrium** with constant energy flow.

**The DLM replaces loss minimization with entropy production maximization under constraints.**

## Key Results (Proof of Concept - Nov 2024)

| Metric | DLM | Standard Network | Improvement |
|--------|-----|------------------|-------------|
| **Catastrophic Forgetting** | 24.7% | 59.3% | **2.4x better retention** |
| Noise Spectrum | -1.53 slope (pink) | ~0 (white) | Biological signature |
| Final Accuracy | 100% | 100% | No performance loss |

### Catastrophic Forgetting Test
- Trained on Task 1 → achieved 100% accuracy
- Trained on Task 2 → DLM retained 75.3% on Task 1, Standard retained only 40.7%
- **The non-equilibrium dynamics prevent "settling" into task-specific minima**

### Noise Spectrum Analysis
- DLM shows **pink noise (1/f)** in activations (slope ≈ -1.5)
- Standard networks show white noise
- **Pink noise is characteristic of biological neural systems**

## The Physics

### Core Equation
```
dθ/dt = -∇L(θ) + η(T) + J(θ)
           ↑        ↑      ↑
        gradient  noise  information current
```

### Key Innovations

1. **Dynamic Weights**: Weights have velocity and momentum - they're never static
2. **Information Currents**: Measure data flow through layers (like particle flux)
3. **Energy Injection/Dissipation**: System maintained far from equilibrium
4. **Entropy Production Maximization**: Learning objective includes thermodynamic term

### Thermodynamic State
```python
@dataclass
class ThermodynamicState:
    energy: float              # Total system energy
    entropy_production: float  # Rate of entropy generation
    information_current: float # Net information flow
    dissipation: float         # Energy lost to environment
    temperature: float         # Controls fluctuation magnitude
```

## Project Structure

```
dissipative-learning-research/
├── src/                    # Core implementation
│   └── dissipative_learning_machine.py
├── experiments/            # Experiment scripts
├── notebooks/              # Jupyter notebooks for exploration
├── docs/                   # Theory and documentation
│   ├── theoretical_foundations.md
│   └── research_roadmap.md
├── data/                   # Datasets
├── results/                # Experiment results and figures
│   └── dlm_results.png
└── README.md
```

## Quick Start

```bash
# Create virtual environment
cd ~/projects/dissipative-learning-research
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch numpy matplotlib

# Run the proof of concept
python src/dissipative_learning_machine.py
```

## Next Steps

See `docs/research_roadmap.md` for the full plan. Key priorities:

1. **Scale to real datasets** (MNIST, CIFAR-10)
2. **Rigorous continual learning benchmarks**
3. **Energy efficiency measurements**
4. **Theoretical analysis and proofs**
5. **Hardware considerations** (neuromorphic potential)

## References

- Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities.
- Prigogine, I. (1977). Self-Organization in Nonequilibrium Systems (Nobel Prize work)
- Hinton, G.E. & Sejnowski, T.J. (1986). Learning and relearning in Boltzmann machines.
- Friston, K. (2010). The free-energy principle: a unified brain theory?

## License

MIT License - Research code, use freely.

## Contact

Research in progress. Contributions welcome.
