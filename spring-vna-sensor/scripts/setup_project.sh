#!/bin/bash
# Project setup script for spring-vna-sensor

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  SPRING COIL VNA SENSOR - PROJECT SETUP                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo

# 1. Python virtual environment
echo "[1/5] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ venv created"
else
    echo "  ✓ venv already exists"
fi

source venv/bin/activate

# 2. Install dependencies
echo "[2/5] Installing Python dependencies..."
pip install --quiet numpy scipy matplotlib pyyaml
pip install --quiet pytest pytest-cov
echo "  ✓ Dependencies installed"

# 3. Create required directories
echo "[3/5] Creating output directories..."
mkdir -p data/raw data/processed data/synthetic
mkdir -p reports/figures
mkdir -p tests/__pycache__
echo "  ✓ Directories created"

# 4. Test imports
echo "[4/5] Testing Python imports..."
python3 -c "import numpy; import scipy; import matplotlib; import yaml; print('  ✓ All imports successful')"

# 5. Run M0 validation (optional)
echo "[5/5] Project setup complete!"
echo

echo "Next steps:"
echo "  1. Run M0 framework validation:"
echo "     python3 scripts/m0_synthetic_validation.py"
echo
echo "  2. Run M1 geometry generation:"
echo "     python3 scripts/m1_geometry_generation.py"
echo
echo "  3. Run complete analysis pipeline:"
echo "     python3 scripts/run_complete_analysis.py"
echo

echo "✓ Project setup complete!"
