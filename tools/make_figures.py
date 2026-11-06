#!/usr/bin/env python3
"""T25: side-by-side qualitative figures, including failure cases.

Builds a grid -- image, ground truth, then one column per model -- from the
prediction maps ``tools/benchmark.py --save_maps`` writes.

``--mode worst`` ranks by the *last* model's MAE and shows where it fails.
The plan asks for at least two failure cases and it is worth saying why:
a figure showing only successes tells a reader nothing they can check, and
reviewers trust papers that show the failures. Pick the worst cases honestly
rather than the prettiest ones.

Uses OpenCV only, so it adds no plotting dependency.

Usage:
    python tools/make_figures.py \
        --imgs data/USEDATA/TestDataset/COD10K/Imgs \
        --gt   data/USEDATA/TestDataset/COD10K/GT \
        --pred "FEDER:res/baseline/COD10K" \
        --pred "Ours:res/final/COD10K" \
        --mode worst --n 4 --out results/figures/failures.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
PAD = 6
LABEL_H = 26


def index(d):
    if not os.path.isdir(d):
        return {}
    return {os.path.splitext(f)[0]: os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith(EXTS) and not f.startswith('.')}


def load_gray(path, size):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[:2] != size:
        m = cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return m


def label_strip(width, text):
    strip = np.full((LABEL_H, width, 3), 255, np.uint8)
    cv2.putText(strip, text[:40], (4, LABEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return strip


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--imgs', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--pred', action='append', required=True,
                    help='"Label:path/to/maps", repeatable; ranking uses the last one')
    ap.add_argument('--mode', default='worst', choices=['worst', 'best', 'random', 'named'])
    ap.add_argument('--names', nargs='*', default=[], help='stems, with --mode named')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--cell', type=int, default=256, help='cell size in pixels')
    ap.add_argument('--out', default='./results/figures/qualitative.png')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    preds = []
    for spec in a.pred:
        if ':' not in spec:
            print('ERROR: --pred needs "Label:path", got {!r}'.format(spec))
            return 2
        # Split once from the left, so a Windows drive letter in the path
        # ("Ours:C:/runs/maps") stays attached to the path.
        label, path = spec.split(':', 1)
        idx = index(path)
        if not idx:
            print('ERROR: no maps found in {}'.format(path))
            return 2
        preds.append((label, idx))

    gi, gg = index(a.imgs), index(a.gt)
    common = set(gi) & set(gg)
    for _, idx in preds:
        common &= set(idx)
    common = sorted(common)
    if not common:
        print('ERROR: no filenames common to images, GT and every --pred directory')
        return 2
    print('{} images common to all sources'.format(len(common)))

    if a.mode == 'named':
        chosen = [n for n in a.names if n in common]
        if not chosen:
            print('ERROR: none of --names are present')
            return 2
    elif a.mode == 'random':
        rng = np.random.default_rng(a.seed)
        chosen = [common[i] for i in rng.choice(len(common),
                                                min(a.n, len(common)), replace=False)]
    else:
        # Rank by the last model's MAE: that is the model the figure is about.
        label, idx = preds[-1]
        print('ranking {} images by {} MAE ...'.format(len(common), label))
        scored = []
        for k in common:
            g = cv2.imread(gg[k], cv2.IMREAD_GRAYSCALE)
            p = load_gray(idx[k], g.shape[:2]) if g is not None else None
            if g is None or p is None:
                continue
            scored.append((np.abs(p.astype(np.float32) / 255 -
                                  (g > 127).astype(np.float32)).mean(), k))
        scored.sort(reverse=(a.mode == 'worst'))
        chosen = [k for _, k in scored[:a.n]]
        print('{} cases: {}'.format(a.mode, ', '.join(chosen)))

    cols = ['Image', 'GT'] + [l for l, _ in preds]
    cell = a.cell
    rows = []
    for k in chosen:
        img = cv2.imread(gi[k], cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (cell, cell))
        gt = load_gray(gg[k], (cell, cell))
        tiles = [img, cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)]
        for _, idx in preds:
            p = load_gray(idx[k], (cell, cell))
            tiles.append(cv2.cvtColor(p, cv2.COLOR_GRAY2BGR))
        row = []
        for t in tiles:
            row.append(cv2.copyMakeBorder(t, PAD, PAD, PAD, PAD,
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        rows.append(np.hstack(row))

    if not rows:
        print('ERROR: nothing could be rendered')
        return 2

    grid = np.vstack(rows)
    header = np.hstack([label_strip(cell + 2 * PAD, c) for c in cols])
    out = np.vstack([header, grid])

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    cv2.imwrite(a.out, out)
    print('wrote {}  ({} rows x {} cols, {}x{} px)'.format(
        a.out, len(rows), len(cols), out.shape[1], out.shape[0]))
    if a.mode == 'worst':
        print('These are failure cases. Show them -- and say in the caption what')
        print('they have in common, if anything (thin structures, low contrast, '
              'multiple objects).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
