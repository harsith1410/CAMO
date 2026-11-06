#!/usr/bin/env python3
"""Report the per-band weights a trained K-band model actually learned.

RQ1 asks whether a finer frequency decomposition helps. A metrics table can
only answer that as a yes or no. This answers the more interesting half: once
the model is free to re-balance the bands, which ones does it lean on?

The weights come from ``BandWeight``: one learnable logit per band, softmax
over bands times K, so 1.0 is "no change from the unweighted model", above 1.0
is "the model amplified this band", below is "it suppressed it". Bands print
high frequency first, approximation last -- the same order the decomposition
returns them in.

With ``band_weight=adaptive`` there is also a content-dependent gate on top of
these, which averages to 1.0 at init but drifts; this prints the static part,
which is the part that is comparable across images and worth a table row.

Usage:
    python tools/band_report.py --pth snapshot/p1_k4/Net_epoch_best.pth
    python tools/band_report.py --pth snapshot/p1_k*/Net_epoch_best.pth
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lib.config import load_config_for_checkpoint

BAND_KEY_HINTS = ('band_weight.logits', 'GCM3.band_weight.logits')


def find_logits(state):
    for k, v in state.items():
        if k.endswith('band_weight.logits'):
            return k, v
    return None, None


def load_state(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        for key in ('model', 'state_dict'):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise ValueError('{} does not hold a state dict'.format(path))
    return {k[7:] if k.startswith('module.') else k: v for k, v in obj.items()}


def label(k, K):
    if k == 0:
        return 'band 0 (finest detail)'
    if k == K - 1:
        return 'band {} (approximation)'.format(k)
    return 'band {} (mid)'.format(k)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pth', nargs='+', required=True,
                    help='checkpoint path(s); globs are expanded')
    a = ap.parse_args()

    paths = []
    for p in a.pth:
        paths.extend(sorted(glob.glob(p)) or [p])

    any_found = False
    for p in paths:
        if not os.path.exists(p):
            print('{}: not found'.format(p))
            continue
        try:
            state = load_state(p)
        except Exception as e:                       # noqa: BLE001 - report, continue
            print('{}: could not read ({})'.format(p, e))
            continue

        name, logits = find_logits(state)
        run = os.path.basename(os.path.dirname(os.path.normpath(p))) or p
        if logits is None:
            cfg = load_config_for_checkpoint(p) or {}
            bw = cfg.get('band_weight', 'none')
            print('\n{}: no band weights stored (band_weight={!r}) -- '
                  'nothing to report for this run.'.format(run, bw))
            continue

        any_found = True
        K = logits.numel()
        w = (torch.softmax(logits.float(), 0) * K).tolist()
        print('\n{}  (K={}, from {})'.format(run, K, name))
        hi = max(range(K), key=lambda i: w[i])
        for k in range(K):
            bar = '#' * int(round(w[k] * 20))
            print('  {:<26} {:6.3f}  {}{}'.format(
                label(k, K), w[k], bar, '   <- most amplified' if k == hi else ''))
        spread = max(w) - min(w)
        if spread < 0.05:
            print('  Spread is {:.3f}: the model left the bands essentially '
                  'balanced, so the weighting is not where any gain came from.'
                  .format(spread))
        else:
            print('  Spread is {:.3f}, most weight on {}.'.format(spread, label(hi, K)))

    if not any_found:
        print('\nNo run had band weights. Train with --band_weight static or '
              '--band_weight adaptive for this to have anything to say.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
