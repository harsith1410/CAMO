#!/usr/bin/env bash
# Environment check. Run once per machine before any training.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== installing requirements =="
# The upstream requirements.txt is a conda spec list that pip cannot read (issue #8).
python -m pip install -r requirements-pip.txt

echo; echo "== dataset =="
python tools/verify_data.py --root data/USEDATA --strict

echo; echo "== decomposition unit tests (T16) =="
python tests/test_wavelet.py

echo; echo "== cost of each K (T09) =="
python tools/measure_cost.py --sweep 2 3 4 6 --learnable_wavelet --testsize 384

echo; echo "Setup OK."
