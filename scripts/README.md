# Experiment runbook

Every command below assumes the repo root as the working directory and a
dataset tree at `data/USEDATA/` (`TrainDataset/` with `Imgs/ GT/ Edge/`, and
`TestDataset/{COD10K,NC4K,CAMO,CHAMELEON}/`). Verify it first:

```bash
python tools/verify_data.py --root data/USEDATA --strict
```

## Part 1 only -- current scope

The report covers **proposed addition 1** alone: replace the two-band DWD with
a learnable decomposition into K bands (K in {2,3,4,6}), with per-band adaptive
weighting and a dedicated mid-band path routed to the OER module. Part 2
(PatchNCE) has been removed from this tree along with the superseded
per-RQ scripts; only the Part 1 path remains.

All three pieces of the proposal are flags, so each can be switched off on its
own:

| piece of the proposal | flag |
|---|---|
| decomposition into K bands | `--bands K` |
| learnable filters | `--learnable_wavelet` |
| per-band adaptive weighting | `--band_weight adaptive` |
| mid-band path to the OER module | `--edge_feed kband` |

`--band_weight none` is the unweighted control and `static` is the
non-content-dependent middle case. All three start from an identical identity
initialisation, so turning the weighting on does not change what the network
computes at step 0 -- any difference is something the optimiser found.

### The three stages, and which card each belongs on

| stage | script | card | hours |
|---|---|---|---|
| 1 | `p1_01_sweep.sh` -- K in {2,3,4,6}, 25 epochs | 3050 | 30.7 |
| 2 | `p1_02_ablate.sh` -- 2 ablations at best K, 25 epochs | 3050 | 13.8 |
| 3 | `p1_03_final.sh` -- 100 epochs, all four test sets | Ada | 8.4 |

**3050: 44.5 h of a 40-50 h budget. Ada: 8.4 h, one overnight run.** Everything
is fp32; no timing here assumes AMP.

```bash
bash scripts/p1_01_sweep.sh                          # laptop, ~31 h
BEST_K=<from stage 1> bash scripts/p1_02_ablate.sh   # laptop, ~14 h
BEST_K=<from stage 1> bash scripts/p1_03_final.sh    # Ada, overnight
```

Stage 3 needs `BEST_K` from stage 1, so the laptop work has to finish first.
Defaults are already set per card -- `BS=4 ACC=9` in stages 1 and 2, `BS=12
ACC=3` in stage 3 -- and the product must stay 36, because the 1e-4 learning
rate was tuned there.

### Why the split falls this way

Per-unit costs are backed out of two real runs, so dataloader stalls and
checkpoint writes are included: **RTX 3050 6 GB** 12.02 min/epoch and 12.76
min/validation; **RTX 4500 Ada 24 GB** 3.57 and 3.79. The Ada is 3.4x the
laptop. Scaled by the per-K latency factors from `tools/measure_cost.py`
(K=2 1.00, K=3 1.14, K=4 1.34, K=6 1.57):

| | 3050 | Ada |
|---|---|---|
| one 25-epoch run, K=2 / K=3 / K=4 / K=6 | 6.1 / 6.9 / 8.1 / 9.5 h | 1.8 / 2.1 / 2.4 / 2.8 h |
| final 100 epochs + 4 test sets, K=3 | 27.9 h | 8.4 h |

Stages 1 and 2 are nine-ish runs whose entire value is in comparing them with
each other, and a table is only comparable if every row came from the same
card -- so they have to stay together, and only the laptop has 45 h to give.
Stage 3 is a single run, it is the one row that sits beside the published
FEDER numbers, and it is 3.4x cheaper on the Ada. That is the whole argument.

At 100 epochs in fp32 the Ada run costs 7.4 h at K=2, 8.4 h at K=3, 9.9 h at
K=4 and 11.5 h at K=6, benchmark included. All of those are overnight; only
K=6 is long enough to want a shorter schedule.

### No second seed at this budget

44.5 h of the laptop's 50 leaves less than one more run, so every row is n=1.
Say so in the report rather than leaving it implied. `tools/val_curve.py`
prints the within-run spread of the last three validations next to the gap
between the best two rows, and that spread is the only uncertainty estimate
available here -- when a gap does not clear it, report a tie.

If the ranking matters more than knowing which component did the work, the
trade is to drop one ablation and re-run the winning K under a second seed:

```bash
BEST_K=3 ABLATIONS="fixed" bash scripts/p1_02_ablate.sh    # one ablation, ~7 h
SEED=7 KS="3" bash scripts/p1_01_sweep.sh                  # second seed, ~7 h
```

A non-default `SEED` writes to `snapshot/p1_k<K>_s<seed>/` and tags its results
row `... seed<N>`, so the two seeds sit side by side instead of one silently
replacing the other.

### Reduced protocol, stated honestly

25 epochs is nowhere near convergence -- the reference 160-epoch run is still
improving there and peaks at epoch 110. Two things make the comparison usable
anyway:

- **Every row shares `--seed 42`**, which fixes weight init, data order and
  augmentation. Without it each row would differ by a random seed as well as
  by K, and at short schedules that nuisance variation is about 0.004 MAE --
  larger than the effects being measured. The rows are paired, so the
  differences are far more reliable than that figure alone suggests.
- **`tools/val_curve.py` compares the mean of the last three validations**
  rather than a single best checkpoint, which at 25 epochs is partly luck.

What a 25-epoch sweep measures is *how fast each K learns*, not how good each K
ends up. That is a fair comparison, because the schedule is identical across
rows -- but it is a different claim, and the report should make it in those
words. The 100-epoch final run is the only row that belongs beside the
published FEDER numbers, and `tools/make_tables.py` keeps a Protocol column so
the two cannot be confused.

## Protocol decisions, and why

**Resolution is 384, not 256.** The plan assumed 256 as the cheap protocol.
It does not work: `AGF` builds a guided filter of radius 2 whose
`h_lrx > 2r+1` assertion needs the coarsest band to exceed 5 px, and at 256
input that band is 4 px. This fails identically in the *released* model, so it
is not something the extension introduced. Only 384, 448 and 512 run at the
published radius — `tests/test_wavelet.py` pins this down.

Two ways to buy a cheaper protocol:

| option | command | trade-off |
|---|---|---|
| fewer epochs at 384 | `--epoch 40 --trainsize 384` | comparable to the baseline and to the paper; **recommended** |
| 256 at a smaller radius | `--trainsize 256 --gf_radius 1` | ~2.2x cheaper per epoch, but changes the architecture, so it is only valid if **every row of the table uses it** |

Whichever you pick, say so in the Notes column. A reduced protocol honestly
labelled is a result; an undocumented one is not.

**Effective batch stays 36.** The 1e-4 learning rate was tuned for batch 36 on
two 24 GB cards. Keep `batchsize * accum_steps == 36` so the LR stays valid:
`--batchsize 4 --accum_steps 9` on a 6-12 GB card, `--batchsize 12
--accum_steps 3` on a 24 GB card.

**AMP is off for the baseline.** Turn it on only after your baseline number
matches, so a reproduction gap can never be blamed on fp16. If the wavelet or
ODE blocks produce NaN under AMP, force the offending term to fp32 with
`torch.amp.autocast('cuda', enabled=False)` rather than abandoning AMP.

**Keep the backbone fixed.** Every row of the RQ1 table must use the same
encoder initialisation. That confound is the entire point of the experiment.

## Resuming after a session is killed

Free tiers cut sessions at 12 h and a full run is longer than that. Every run
writes `last.pth` each epoch with model, optimizer, scaler, epoch and best
metric. Resume with:

```bash
python Train.py --resume snapshot/<run>/last.pth ... # same flags as the original run
```

`--load` restores weights only and restarts the optimizer and schedule; use
`--resume` for continuation, `--load` only for fine-tuning from another run.

## Reading the results

`tools/benchmark.py` appends to `results/results.csv` and prints the gap
against the published FEDER row. A gain of about 0.01 S-alpha is a real
improvement; 0.002-0.003 is run-to-run noise and needs multiple seeds before
you claim it.

For the `edge fed 2-band` ablation, report boundary-F (printed by
`tools/benchmark.py`) alongside S-alpha: an edge-quality change moves
boundary-F well before it moves S-alpha.

`tools/band_report.py` prints the per-band weights a run actually learned,
which is the part of RQ1 a metrics table cannot show.

## Assembling the report

Once runs exist:

```bash
python tools/make_tables.py --table all --out results/tables.md
python tools/make_tables.py --table rq1 --format latex

python tools/make_figures.py \
  --imgs data/USEDATA/TestDataset/COD10K/Imgs \
  --gt   data/USEDATA/TestDataset/COD10K/GT \
  --pred "FEDER:res/baseline/COD10K" \
  --pred "Ours:res/p1_final/COD10K" \
  --mode worst --n 4 --out results/figures/failures.png
```

`--mode worst` picks the genuine failure cases rather than the prettiest ones.
Show at least two, and say in the caption what they have in common.
