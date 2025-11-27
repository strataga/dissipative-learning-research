# Reproducibility Checklist

For NeurIPS/ICML/ICLR submission compliance.

## Code and Data

| Item | Status | Location |
|------|--------|----------|
| Source code | ✓ | `src/` |
| Experiment scripts | ✓ | `experiments/` |
| Data download scripts | ✓ | Auto-download via torchvision |
| Pre-trained models | N/A | Training is fast (<5 min per experiment) |
| Random seeds | ✓ | Set in each experiment |

## Experimental Details

### Datasets

| Dataset | Source | Preprocessing |
|---------|--------|---------------|
| MNIST | torchvision | Normalize (μ=0.1307, σ=0.3081), flatten |
| CIFAR-10 | torchvision | Normalize (μ=[0.49,0.48,0.45], σ=[0.25,0.24,0.26]), flatten |

### Hyperparameters

| Parameter | Value | Tuning Method |
|-----------|-------|---------------|
| Learning rate | 0.001 | Grid search [0.0001, 0.001, 0.01] |
| Batch size | 64 | Fixed |
| Epochs per task | 3 | Grid search [1, 3, 5, 10] |
| Sparsity | 5% | Sweep [1%, 5%, 10%, 15%, 25%, 50%, 100%] |
| EWC λ | 2000 | Sweep [100, 500, 1000, 2000, 5000, 10000] |
| Temperature | 1.0 | Sweep [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] |
| Fisher samples | 200 | Fixed (following Kirkpatrick et al.) |

### Architecture

```
Split MNIST:  [784, 256, 10] with ReLU/k-WTA
Permuted MNIST: [784, 128, 10] with ReLU/k-WTA
CIFAR-10: [3072, 256, 10] with ReLU/k-WTA
```

### Compute Resources

| Resource | Details |
|----------|---------|
| Hardware | Apple M-series / Intel CPU (no GPU required) |
| Time per experiment | 2-10 minutes |
| Total compute | ~4 hours for all 16 experiments |
| Memory | <4GB RAM |

## Statistical Reporting

### Main Results (Split MNIST)

| Method | Forgetting (mean±std) | Accuracy (mean±std) | n_runs |
|--------|----------------------|---------------------|--------|
| Standard | 0.997 ± 0.002 | 0.197 ± 0.01 | 3 |
| EWC | 0.948 ± 0.02 | 0.238 ± 0.02 | 3 |
| Sparse 5% | 0.678 ± 0.05 | 0.431 ± 0.03 | 3 |
| Sparse + EWC | 0.323 ± 0.04 | 0.526 ± 0.02 | 3 |

### Statistical Tests

| Comparison | Test | p-value | Significant? |
|------------|------|---------|--------------|
| Sparse vs Standard | t-test | <0.001 | Yes |
| Sparse+EWC vs EWC | t-test | <0.001 | Yes |
| Sparsity-Overlap correlation | Pearson | 0.017 | Yes (r=0.89) |

## Experiment Scripts

| Experiment | Script | Command |
|------------|--------|---------|
| EXP-001-010 | `validate_theory.py` | `python experiments/validate_theory.py` |
| EXP-011 | `debug_entropy.py` | `python experiments/debug_entropy.py` |
| EXP-012 | `thermodynamic_loss.py` | `python experiments/thermodynamic_loss.py` |
| EXP-013 | `sparse_thermodynamic.py` | `python experiments/sparse_thermodynamic.py` |
| EXP-014 | `triple_combination.py` | `python experiments/triple_combination.py` |
| EXP-015 | `permuted_mnist.py` | `python experiments/permuted_mnist.py` |
| EXP-016 | `cifar10_validation.py` | `python experiments/cifar10_validation.py` |

## Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision numpy matplotlib scipy

# Run all experiments
for exp in experiments/*.py; do
    python "$exp"
done
```

## Results Verification

All results can be reproduced by running:

```bash
cd /path/to/dissipative-learning-research
source venv/bin/activate
python experiments/final_summary.py
```

This generates:
- `results/paper_figure_main.png`
- `results/paper_figure_thermo.png`
- LaTeX tables (printed to stdout)

## Negative Results Reported

| Claim Tested | Result | Experiment |
|--------------|--------|------------|
| Thermodynamic loss alone helps | NO effect | EXP-012 |
| Dissipation rate affects forgetting | NO correlation | EXP-004 |
| Sleep cycles help | <1% improvement | EXP-006 |
| Temperature is tunable | Phase transition, not tunable | EXP-003 |

## Limitations Acknowledged

1. **Architecture**: Only tested on MLPs; CNN results pending
2. **Scale**: CIFAR-10 effect smaller (3% vs 68%)
3. **Benchmarks**: Limited to MNIST variants and CIFAR-10
4. **Variance**: Some run-to-run variance (~5%)

## Checklist Items (NeurIPS 2024)

- [x] Includes code submission
- [x] Documents all hyperparameters
- [x] Reports error bars / confidence intervals
- [x] Describes compute resources
- [x] Includes negative results
- [x] Discusses limitations
- [x] Provides reproducibility instructions
