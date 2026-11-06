#!/usr/bin/env python3
"""Read validation MAE out of run logs and compare runs robustly.

Why this exists. At the full 160-epoch schedule the val curve is flat at the
end, so ``Net_epoch_best.pth`` is a stable choice: in the reference baseline
the last 16 validations have sd 0.0005 MAE. At the 25-epoch schedule the
deadline forces, the curve is still falling *and* still noisy -- the same
baseline swings 0.0462 -> 0.0386 -> 0.0405 across epochs 15/20/25. Picking
the single best of five such draws rewards luck, and the luck differs per row.

So for short runs, compare the mean of the last few validations rather than
the best one, and look at the spread before believing any ordering.

Usage:
    python tools/val_curve.py snapshot/rq1_k*            # compare runs
    python tools/val_curve.py snapshot/rq1_k3 --last 3   # window size
"""
import argparse
import glob
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Two shapes appear in practice: the file logger's "[Val Info]" line, and the
# bare stdout print that shows up when output is piped to a file.
PATTERNS = [
    re.compile(r'\[Val Info\]:Epoch:(\d+)\s+MAE:([\d.]+)'),
    re.compile(r'^Epoch:\s*(\d+),\s*MAE:\s*([\d.]+)', re.M),
]


def read_log(path):
    """Return [(epoch, mae)] from a log file, newest duplicate epoch winning."""
    raw = open(path, 'rb').read()
    txt = None
    for enc in ('utf-8', 'utf-16', 'latin-1'):
        try:
            cand = raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
        if 'MAE' in cand:
            txt = cand
            break
    if txt is None:
        return []
    out = {}
    for pat in PATTERNS:
        for m in pat.finditer(txt):
            out[int(m.group(1))] = float(m.group(2))
        if out:
            break
    return sorted(out.items())


def find_logs(target):
    if os.path.isfile(target):
        return [target]
    hits = sorted(glob.glob(os.path.join(target, '*.log')))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run directories or log files')
    ap.add_argument('--last', type=int, default=3,
                    help='average over the last N validations (default 3)')
    a = ap.parse_args()

    targets = []
    for r in a.runs:
        targets.extend(sorted(glob.glob(r)) or [r])

    rows = []
    for t in targets:
        logs = find_logs(t)
        if not logs:
            print('no log found in {}'.format(t))
            continue
        curve = []
        for lg in logs:
            curve = read_log(lg) or curve
        if not curve:
            print('no validation lines in {}'.format(t))
            continue
        tail = [v for _, v in curve[-a.last:]]
        rows.append(dict(
            name=os.path.basename(os.path.normpath(t)),
            n=len(curve),
            last_ep=curve[-1][0],
            best=min(v for _, v in curve),
            best_ep=min(curve, key=lambda kv: kv[1])[0],
            mean=st.mean(tail),
            sd=st.stdev(tail) if len(tail) > 1 else float('nan'),
            curve=curve))

    if not rows:
        return 1

    print('\n{:<24} {:>5} {:>8} {:>7} {:>10} {:>8}'.format(
        'run', 'vals', 'best', '@ep', 'mean/last%d' % a.last, 'sd'))
    print('-' * 66)
    for r in sorted(rows, key=lambda r: r['mean']):
        print('{:<24} {:>5} {:>8.5f} {:>7} {:>10.5f} {:>8.5f}'.format(
            r['name'], r['n'], r['best'], r['best_ep'], r['mean'], r['sd']))

    for r in rows:
        print('\n{}: {}'.format(r['name'], '  '.join(
            'ep{}={:.5f}'.format(e, v) for e, v in r['curve'])))

    if len(rows) > 1:
        ranked = sorted(rows, key=lambda r: r['mean'])
        a0, b0 = ranked[0], ranked[1]
        gap = b0['mean'] - a0['mean']
        noise = max(x for x in (a0['sd'], b0['sd']) if x == x)
        print('\nbest two: {} beats {} by {:.5f} MAE; '
              'within-run spread is {:.5f}.'.format(a0['name'], b0['name'], gap, noise))
        if gap < noise:
            print('That gap is SMALLER than the noise. Report it as a tie -- '
                  'ranking these two needs more epochs or more seeds.')
        else:
            print('That gap clears the within-run spread, but a single seed '
                  'still cannot separate close rows. Say "n=1" in the caption.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
