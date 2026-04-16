#!/usr/bin/env bash
# reproduce.sh — Reproduce the BN class-conditional statistics experiment
# Expected runtime: ~80-90 minutes on Apple Silicon (MPS), longer on CPU
# Expected output: results.json in the same directory as experiment.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "BN Statistics: Calibration vs Feature Encoding"
echo "============================================"
echo ""
echo "This script reproduces the experiment from:"
echo "  'Batch Normalization Statistics as Output Calibrators,"
echo "   Not Feature Encoders'"
echo ""
echo "Requirements:"
echo "  - Python 3.8+"
echo "  - PyTorch >= 2.0 (with MPS or CPU support)"
echo "  - torchvision"
echo "  - numpy"
echo "  - scipy"
echo ""

# Check Python dependencies
python3 -c "import torch, torchvision, numpy, scipy" 2>/dev/null || {
    echo "ERROR: Missing Python dependencies."
    echo "Install with: pip install torch torchvision numpy scipy"
    exit 1
}

echo "Device detection:"
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA available: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('  MPS (Apple Silicon) available')
else:
    print('  CPU only (this will be slow, ~3-4 hours)')
"
echo ""

echo "Starting experiment..."
echo "CIFAR-10 will be downloaded to ${SCRIPT_DIR}/data/ if not present."
echo ""

cd "$SCRIPT_DIR"
python3 experiment.py

echo ""
echo "============================================"
echo "Experiment complete!"
echo "Results saved to: ${SCRIPT_DIR}/results.json"
echo "============================================"
