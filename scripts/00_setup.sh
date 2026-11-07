#!/usr/bin/env bash
# Environment check. Run once per machine before any training.
set -euo pipefail
cd "$(dirname "$0")/.."

# Resolve the interpreter. Git Bash on Windows does not inherit the venv that
# a PowerShell "(.venv)" prompt advertises, and the `python` it does find is
# usually a system install without torch -- which fails later and less
# obviously. So prefer the venv explicitly, and refuse to guess.
if [ -z "${PY:-}" ]; then
  for cand in ./.venv/Scripts/python.exe ./.venv/bin/python python3 python py; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
fi
if [ -z "${PY:-}" ]; then
  echo "No python found. Activate the venv, or pass PY=/path/to/python." >&2
  exit 1
fi
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: '$PY' cannot import torch." >&2
  echo "       On Windows use: PY=./.venv/Scripts/python.exe bash $0" >&2
  exit 1
fi
echo "interpreter: $PY"

echo "== installing requirements =="
# The upstream requirements.txt is a conda spec list that pip cannot read (issue #8).
"$PY" -m pip install -r requirements-pip.txt

echo; echo "== dataset =="
"$PY" tools/verify_data.py --root data/USEDATA --strict

echo; echo "== decomposition unit tests (T16) =="
"$PY" tests/test_wavelet.py

echo; echo "== cost of each K (T09) =="
"$PY" tools/measure_cost.py --sweep 2 3 4 6 --learnable_wavelet --testsize 384

echo; echo "Setup OK."
