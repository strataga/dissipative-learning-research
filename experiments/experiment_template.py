"""
EXPERIMENT TEMPLATE
===================

Copy this template for new experiments. Fill in all sections.

Experiment ID: EXP-XXX
Date: YYYY-MM-DD
Author: [Name]

## Quick Reference

When complete, add entry to docs/experiment_log.md with:
- ID, Date, Title, Result summary, Key finding
- Full detailed record in the log

"""

import sys
import os
import json
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# EXPERIMENT METADATA - FILL THIS IN
# =============================================================================

EXPERIMENT = {
    'id': 'EXP-XXX',
    'title': 'Experiment Title',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'hypothesis': """
    State your hypothesis clearly.
    What do you expect to happen and why?
    """,
    'method': """
    Describe the experimental setup:
    - Dataset
    - Architecture
    - Training procedure
    - Hyperparameters
    - Metrics
    """,
    'status': 'planned',  # planned, running, complete, failed
}

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Dataset
    'dataset': 'MNIST',
    'batch_size': 64,
    
    # Architecture
    'layer_sizes': [784, 256, 10],
    
    # Training
    'epochs': 10,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    
    # Model-specific
    'temperature': 1.0,
    'sparsity': 0.05,
    'dissipation_rate': 0.02,
    
    # Experiment-specific
    # Add your parameters here
}

# =============================================================================
# RESULTS STORAGE
# =============================================================================

RESULTS = {
    'metrics': {},
    'raw_data': {},
    'figures': [],
    'analysis': '',
    'implications': '',
    'limitations': '',
}


def save_results(experiment_id: str, results: dict, config: dict):
    """Save experiment results to JSON"""
    output = {
        'experiment': EXPERIMENT,
        'config': config,
        'results': {k: v for k, v in results.items() if k != 'raw_data'},
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save JSON
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'experiments')
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, f'{experiment_id}.json')
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"Results saved to {filepath}")
    return filepath


def log_metric(name: str, value, step: int = None):
    """Log a metric value"""
    if name not in RESULTS['metrics']:
        RESULTS['metrics'][name] = []
    
    entry = {'value': value, 'step': step, 'time': datetime.now().isoformat()}
    RESULTS['metrics'][name].append(entry)


def log_figure(fig, name: str):
    """Save a figure and log it"""
    filepath = os.path.join(PROJECT_ROOT, 'results', f'{EXPERIMENT["id"]}_{name}.png')
    fig.savefig(filepath, dpi=150)
    RESULTS['figures'].append(filepath)
    print(f"Figure saved to {filepath}")


# =============================================================================
# EXPERIMENT CODE - IMPLEMENT YOUR EXPERIMENT HERE
# =============================================================================

def setup():
    """Setup experiment (load data, create models, etc.)"""
    print(f"Setting up {EXPERIMENT['id']}: {EXPERIMENT['title']}")
    print(f"Hypothesis: {EXPERIMENT['hypothesis']}")
    
    # TODO: Implement setup
    # - Load dataset
    # - Create model(s)
    # - Initialize optimizer(s)
    
    pass


def run():
    """Run the experiment"""
    EXPERIMENT['status'] = 'running'
    print(f"\nRunning experiment...")
    
    # TODO: Implement experiment
    # - Training loop
    # - Evaluation
    # - Metric logging
    
    # Example:
    # for epoch in range(CONFIG['epochs']):
    #     train_loss = train_epoch(model, train_loader, optimizer)
    #     test_acc = evaluate(model, test_loader)
    #     log_metric('train_loss', train_loss, epoch)
    #     log_metric('test_acc', test_acc, epoch)
    
    pass


def analyze():
    """Analyze results"""
    print(f"\nAnalyzing results...")
    
    # TODO: Implement analysis
    # - Statistical tests
    # - Compute summary statistics
    # - Generate figures
    
    # Example:
    # RESULTS['analysis'] = """
    # Key findings:
    # 1. Finding one
    # 2. Finding two
    # """
    
    pass


def report():
    """Generate final report"""
    EXPERIMENT['status'] = 'complete'
    
    print("\n" + "=" * 70)
    print(f"EXPERIMENT REPORT: {EXPERIMENT['id']}")
    print("=" * 70)
    
    print(f"\nTitle: {EXPERIMENT['title']}")
    print(f"Date: {EXPERIMENT['date']}")
    print(f"Status: {EXPERIMENT['status']}")
    
    print(f"\nHypothesis:\n{EXPERIMENT['hypothesis']}")
    
    print(f"\nMethod:\n{EXPERIMENT['method']}")
    
    print(f"\nResults:")
    for metric, values in RESULTS['metrics'].items():
        if values:
            final = values[-1]['value']
            print(f"  {metric}: {final}")
    
    print(f"\nAnalysis:\n{RESULTS['analysis']}")
    
    print(f"\nImplications:\n{RESULTS['implications']}")
    
    print(f"\nLimitations:\n{RESULTS['limitations']}")
    
    print(f"\nFigures: {RESULTS['figures']}")
    
    # Save results
    save_results(EXPERIMENT['id'], RESULTS, CONFIG)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete experiment pipeline"""
    setup()
    run()
    analyze()
    report()


if __name__ == "__main__":
    main()
