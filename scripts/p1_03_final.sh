#!/usr/bin/env bash
# PART 1, stage 3 -- the headline run, all four test sets.
#
# This is the stage that belongs on the fast card. It is the only row that
# sits beside the published FEDER numbers, it is a single run rather than a
# table, and at 100 epochs with AMP it costs about 6 h on a 24 GB Ada
# against about 20 h on a 6 GB laptop. Stages 1 and 2 stay on the laptop:
# they are many runs whose value is in comparing with each other, and a
# table is only comparable if every row came from the same card.
#
# Same architecture stage 1 picked, trained on the long schedule and scored
# everywhere FEDER reports. This is the row that goes next to the published
# numbers; everything before it is a 25-epoch reduced protocol and must be
# labelled as such in the report.
set -euo pipefail
cd "$(dirname "$0")/.."

BEST_K=${BEST_K:?set BEST_K from the stage 1 table, e.g. BEST_K=3 bash $0}
# The reference baseline peaks at epoch 110 and gains nothing after it, so 100
# is a cheap place to stop. Overnight cost on a 24 GB Ada in fp32, including
# the benchmark pass over all four test sets:
#     K=2 -> 7.4 h    K=3 -> 8.4 h    K=4 -> 9.9 h    K=6 -> 11.5 h
EPOCHS=${EPOCHS:-100}
BS=${BS:-12}               # 24 GB card: 12 * 3 = 36. On 6-8 GB use BS=4 ACC=9.
ACC=${ACC:-3}
SEED=${SEED:-42}
RUN=${RUN:-snapshot/p1_final}
AMP=${AMP:-}               # off: the hours below assume fp32

RESUME=""
if [ -f "$RUN/last.pth" ]; then RESUME="--resume $RUN/last.pth"; echo "resuming"; fi

python Train.py \
  --epoch "$EPOCHS" --lr 1e-4 --batchsize "$BS" --accum_steps "$ACC" \
  --trainsize 384 --seed "$SEED" $AMP --val_freq 5 --save_freq 10 \
  --bands "$BEST_K" --learnable_wavelet --band_weight adaptive --edge_feed kband \
  --train_root ./data/USEDATA/TrainDataset/ \
  --val_root   ./data/USEDATA/TestDataset/COD10K/ \
  --save_path "$RUN/" $RESUME

# Keep the maps: the qualitative figure needs them and regenerating costs
# another full inference pass over 6473 images.
python tools/benchmark.py --pth "$RUN/Net_epoch_best.pth" \
  --datasets COD10K NC4K CAMO CHAMELEON --save_maps \
  --tag "Part 1 final K=$BEST_K" \
  --notes "384px ${EPOCHS}ep eff-batch $((BS*ACC)) seed=${SEED} AMP K=${BEST_K} adaptive kband"

echo
echo "Learned per-band weights:"
python tools/band_report.py --pth "$RUN/Net_epoch_best.pth" || true

echo
python tools/make_tables.py --table part1 --out results/tables_part1.md || true

cat <<'EOF'

Two things to do before this goes in the report:

1. The 25-epoch rows and this 100-epoch row are different protocols. Do not
   put them in one column without saying so -- the Protocol column in
   tools/make_tables.py output exists for exactly this.
2. Build the qualitative figure from the saved maps, and include failure
   cases, not just the flattering ones:

   python tools/make_figures.py \
     --imgs data/USEDATA/TestDataset/COD10K/Imgs \
     --gt   data/USEDATA/TestDataset/COD10K/GT \
     --pred "FEDER:res/baseline/COD10K" \
     --pred "Ours:res/p1_final/COD10K" \
     --mode worst --n 4 --out results/figures/failures.png
EOF
