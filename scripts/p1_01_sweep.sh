#!/usr/bin/env bash
# PART 1, stage 1 -- the K sweep.
#
# Proposed addition 1: replace the two-band DWD with a learnable decomposition
# into K bands, K in {2,3,4,6}, with per-band adaptive weighting and a
# dedicated mid-band path routed to the OER module. This script runs the
# headline configuration at each K; everything else is held fixed.
#
# K=2 is the control. With two bands there is no mid band, so --edge_feed
# kband collapses to the released routing and the row differs from FEDER only
# by learnable filters and the band weighting -- which is what makes it the
# right comparison point for the rest of the column.
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

EPOCHS=${EPOCHS:-25}
BS=${BS:-4}                # batchsize * ACC must be 36: the 1e-4 LR was tuned there.
ACC=${ACC:-9}              # 6 GB laptop card: 4 * 9. On 24 GB use BS=12 ACC=3.
AMP=${AMP:-}               # off: every timing and budget here assumes fp32.
                           # AMP=--amp turns it on; if you do, turn it on for
                           # every row of the table, not just some.
SIZE=${SIZE:-384}          # 256 does not run at the released GF radius; see README
SEED=${SEED:-42}
KS=${KS:-"2 3 4 6"}
NOTE="${SIZE}px ${EPOCHS}ep eff-batch $((BS*ACC)) seed=${SEED}${AMP:+ AMP} reduced-protocol"

# A second seed must not overwrite the first, and must not collide in the
# results table either -- make_tables.py keeps one row per tag, so a reused
# tag would silently replace the earlier seed instead of sitting beside it.
SUF=""
TAGSEED=""
if [ "$SEED" != "42" ]; then
  SUF="_s${SEED}"
  TAGSEED=" seed$SEED"
fi

for K in $KS; do
  RUN="snapshot/p1_k${K}${SUF}"
  echo "=== K=$K ==============================================="
  RESUME=""
  if [ -f "$RUN/last.pth" ]; then RESUME="--resume $RUN/last.pth"; echo "resuming"; fi

  "$PY" Train.py \
    --epoch "$EPOCHS" --lr 1e-4 --batchsize "$BS" --accum_steps "$ACC" \
    --trainsize "$SIZE" --seed "$SEED" $AMP --val_freq 5 --save_freq 5 \
    --bands "$K" --learnable_wavelet --band_weight adaptive --edge_feed kband \
    --train_root ./data/USEDATA/TrainDataset/ \
    --val_root   ./data/USEDATA/TestDataset/COD10K/ \
    --save_path "$RUN/" $RESUME

  "$PY" tools/benchmark.py --pth "$RUN/Net_epoch_best.pth" \
    --datasets COD10K --testsize "$SIZE" \
    --tag "K=$K adaptive kband$TAGSEED" --notes "$NOTE"
done

echo
echo "=== how the rows actually compare ==="
"$PY" tools/val_curve.py snapshot/p1_k* --last 3

cat <<'EOF'

25 epochs is a reduced protocol, not convergence: the reference 160-epoch run
is still improving at that point, and peaks at epoch 110. It is a fair comparison because every row
uses the same schedule and the same --seed 42, which fixes weight init, data
order and augmentation -- so the rows are paired and differ only by K. But
what it measures is how fast each K learns, not how good each K ends up, and
the report should say that in those words.

If two K values land within ~0.005 S-alpha, val_curve.py will say the gap does
not clear the noise. Report them as a tie, or spend the spare budget breaking
it on just the top two:

    EPOCHS=40 KS="3 4" bash scripts/p1_01_sweep.sh

Otherwise pick BEST_K from this table and run scripts/p1_02_ablate.sh.
EOF
