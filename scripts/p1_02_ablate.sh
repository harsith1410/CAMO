#!/usr/bin/env bash
# PART 1, stage 2 -- which piece of the proposal is doing the work?
#
# Stage 1 shows whether the K-band model beats K=2. It cannot say why, because
# the headline row changes four things at once: more bands, learnable filters,
# per-band adaptive weighting, and mid bands routed to the OER module. Each
# run below switches exactly one of those back off, at the K stage 1 chose.
#
# Two of the five are run by default. That is the pair that answers the
# question a marker will actually ask -- "is it the extra bands or the fact
# that they are learned?" and "did the weighting do anything?" -- and it is
# what fits beside the sweep in a 40-50 h laptop budget. Add the others with
# ABLATIONS=... if time allows; each is one more run.
#
#   fixed    filters frozen at the Haar / B-spline init
#   bwnone   no per-band weighting at all
#   bwstat   weighting learned but not content-dependent
#   edge2    OER fed only the two extreme bands, as released FEDER does
#   nomid    mid bands dropped while K and its parameter count stay
set -euo pipefail
cd "$(dirname "$0")/.."

BEST_K=${BEST_K:?set BEST_K from the stage 1 table, e.g. BEST_K=3 bash $0}
EPOCHS=${EPOCHS:-25}
BS=${BS:-4}                # batchsize * ACC must be 36: the 1e-4 LR was tuned there.
ACC=${ACC:-9}              # 6 GB laptop card: 4 * 9. On 24 GB use BS=12 ACC=3.
AMP=${AMP:-}               # off: every timing and budget here assumes fp32.
                           # AMP=--amp turns it on; if you do, turn it on for
                           # every row of the table, not just some.
SIZE=${SIZE:-384}
SEED=${SEED:-42}
ABLATIONS=${ABLATIONS:-"fixed bwnone"}
NOTE="${SIZE}px ${EPOCHS}ep eff-batch $((BS*ACC)) seed=${SEED}${AMP:+ AMP} K=${BEST_K} ablation"

# $1 run suffix, $2 results tag, $3.. flags replacing the headline ones
run_one () {
  local RUN="snapshot/p1_ab_$1"; local TAG="$2"; shift 2
  echo "=== $TAG ==============================================="
  local RESUME=""
  if [ -f "$RUN/last.pth" ]; then RESUME="--resume $RUN/last.pth"; fi
  python Train.py \
    --epoch "$EPOCHS" --lr 1e-4 --batchsize "$BS" --accum_steps "$ACC" \
    --trainsize "$SIZE" --seed "$SEED" $AMP --val_freq 5 --save_freq 10 \
    --bands "$BEST_K" "$@" \
    --train_root ./data/USEDATA/TrainDataset/ \
    --val_root   ./data/USEDATA/TestDataset/COD10K/ \
    --save_path "$RUN/" $RESUME
  python tools/benchmark.py --pth "$RUN/Net_epoch_best.pth" \
    --datasets COD10K --testsize "$SIZE" --tag "$TAG" --notes "$NOTE"
}

for AB in $ABLATIONS; do
  case "$AB" in
    fixed)
      run_one fixed  "K=$BEST_K ablation: fixed filters" \
              --band_weight adaptive --edge_feed kband ;;
    bwnone)
      run_one bwnone "K=$BEST_K ablation: no band weighting" \
              --learnable_wavelet --band_weight none --edge_feed kband ;;
    bwstat)
      run_one bwstat "K=$BEST_K ablation: static band weighting" \
              --learnable_wavelet --band_weight static --edge_feed kband ;;
    edge2)
      # Report boundary-F here, not just S-alpha: edge quality moves
      # boundary-F well before it moves S-alpha.
      run_one edge2  "K=$BEST_K ablation: edge fed 2-band" \
              --learnable_wavelet --band_weight adaptive --edge_feed 2band ;;
    nomid)
      run_one nomid  "K=$BEST_K ablation: mid bands ablated" \
              --learnable_wavelet --band_weight adaptive --edge_feed kband --ablate_mid ;;
    *)
      echo "unknown ablation '$AB' -- expected one of: fixed bwnone bwstat edge2 nomid" >&2
      exit 2 ;;
  esac
done

echo
python tools/val_curve.py "snapshot/p1_k${BEST_K}" snapshot/p1_ab_* --last 3
echo
echo "Learned per-band weights (which frequencies the model chose to rely on):"
python tools/band_report.py --pth "snapshot/p1_k${BEST_K}/Net_epoch_best.pth" || true
