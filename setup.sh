#!/bin/bash
# Setup script for Dissipative Learning Research

set -e

echo "=========================================="
echo "Dissipative Learning Research - Setup"
echo "=========================================="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To get started:"
echo "  cd ~/projects/dissipative-learning-research"
echo "  source venv/bin/activate"
echo "  python src/dissipative_learning_machine.py"
echo ""
echo "Key files:"
echo "  - src/dissipative_learning_machine.py  (main implementation)"
echo "  - docs/research_roadmap.md             (research plan)"
echo "  - docs/theoretical_foundations.md      (theory)"
echo "  - results/dlm_results.png              (initial results)"
echo ""
