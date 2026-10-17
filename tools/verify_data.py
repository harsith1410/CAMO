#!/usr/bin/env python3
"""Verify a COD dataset tree before spending GPU hours on it.

Checks layout, counts, Imgs/GT pairing, mask validity and edge-map presence,
and compares counts against the published benchmark sizes.

Usage:
    python tools/verify_data.py --root /content/data
    python tools/verify_data.py --root /content/data --strict
"""
import argparse, os, sys
from collections import Counter

import numpy as np
from PIL import Image

# Published sizes. Sources: COD10K (Fan et al., CVPR 2020), NC4K (Lv et al.,
# CVPR 2021), CAMO (Le et al., CVIU 2019). The standard COD training set is
# 3,040 COD10K-train + 1,000 CAMO-train = 4,040.
EXPECTED = {
    "TrainDataset":        (4040, "standard combined set (3,040 COD10K + 1,000 CAMO)"),
    "TestDataset/COD10K":  (2026, "COD10K-test"),
    "TestDataset/CAMO":    (250,  "CAMO-test"),
    "TestDataset/CHAMELEON": (76,  "CHAMELEON (test only)"),
    "TestDataset/NC4K":    (4121, "NC4K (test only)"),
}
ALT_TRAIN = {3040: "COD10K-train only -- you still need the 1,000 CAMO-train images"}
IMG_DIRS = ("Imgs", "Image", "Images", "image", "img")
GT_DIRS = ("GT", "gt", "Mask", "masks", "GT_Object")
EDGE_DIRS = ("Edge", "edge", "Edges")
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def pick(parent, names):
    for n in names:
        p = os.path.join(parent, n)
        if os.path.isdir(p):
            return p
    return None


def stems(d):
    if d is None:
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(d)
            if f.lower().endswith(EXTS) and not f.startswith(".")}


def sample_masks(gt_dir, names, n=25):
    """Look at a sample of masks: are they single-channel and effectively binary?"""
    bad_mode, non_binary, sizes = [], [], []
    for name in sorted(names)[:n]:
        hit = None
        for e in EXTS:
            c = os.path.join(gt_dir, name + e)
            if os.path.exists(c):
                hit = c
                break
        if hit is None:
            continue
        im = Image.open(hit)
        if im.mode not in ("L", "1", "P"):
            bad_mode.append(f"{os.path.basename(hit)} is mode {im.mode}")
        a = np.array(im.convert("L"))
        sizes.append(a.shape)
        mid = ((a > 10) & (a < 245)).mean()
        if mid > 0.05:
            non_binary.append(f"{os.path.basename(hit)} has {mid:.1%} mid-grey pixels")
    return bad_mode, non_binary, sizes


def check_split(root, rel, need_edges, strict):
    d = os.path.join(root, rel)
    issues, notes = [], []
    if not os.path.isdir(d):
        return None, [f"MISSING: {rel} not found under {root}"], []

    img_d, gt_d, edge_d = pick(d, IMG_DIRS), pick(d, GT_DIRS), pick(d, EDGE_DIRS)
    if img_d is None:
        issues.append(f"{rel}: no image directory (looked for {', '.join(IMG_DIRS)})")
    if gt_d is None:
        issues.append(f"{rel}: no ground-truth directory (looked for {', '.join(GT_DIRS)})")
    if img_d is None or gt_d is None:
        return None, issues, notes

    si, sg = stems(img_d), stems(gt_d)
    n = len(si)
    only_img, only_gt = si - sg, sg - si
    if only_img:
        issues.append(f"{rel}: {len(only_img)} image(s) with no mask, e.g. {sorted(only_img)[:3]}")
    if only_gt:
        issues.append(f"{rel}: {len(only_gt)} mask(s) with no image, e.g. {sorted(only_gt)[:3]}")

    exp = EXPECTED.get(rel)
    if exp:
        want, label = exp
        if n == want:
            notes.append(f"{rel}: {n} images -- matches {label}")
        elif rel == "TrainDataset" and n in ALT_TRAIN:
            issues.append(f"{rel}: {n} images -- {ALT_TRAIN[n]}")
        else:
            issues.append(f"{rel}: {n} images, expected {want} ({label}). "
                          f"Your numbers will not be comparable to published ones.")
    else:
        notes.append(f"{rel}: {n} images")

    bad_mode, non_binary, sizes = sample_masks(gt_d, si & sg)
    for m in bad_mode[:3]:
        notes.append(f"{rel}: {m} (converted to L on load -- usually fine)")
    for m in non_binary[:3]:
        issues.append(f"{rel}: {m} -- masks should be binary, check you did not grab soft maps")
    if sizes:
        odd = [s for s in sizes if s[0] % 2 or s[1] % 2]
        if odd:
            notes.append(f"{rel}: {len(odd)}/{len(sizes)} sampled masks have an odd dimension. "
                         "Harmless at load time (images are resized), but keep the network "
                         "input size a power of two -- the wavelet decomposition breaks on odd sizes.")

    if need_edges:
        if edge_d is None:
            issues.append(f"{rel}: no Edge/ directory. FEDER's dataloader requires it. "
                          "Generate it with tools/make_edges.py -- no public COD release ships edges.")
        else:
            se = stems(edge_d)
            missing = (si & sg) - se
            if missing:
                issues.append(f"{rel}: Edge/ exists but is missing {len(missing)} map(s). Re-run make_edges.py.")
            else:
                notes.append(f"{rel}: Edge/ complete ({len(se)} maps)")
    return n, issues, notes


def main():
    ap = argparse.ArgumentParser(description="Verify a COD dataset tree.")
    ap.add_argument("--root", required=True, help="directory holding TrainDataset/ and TestDataset/")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any issue is found")
    ap.add_argument("--no-edges", action="store_true", help="skip the Edge/ requirement")
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        print(f"ERROR: --root {a.root} is not a directory")
        return 2

    all_issues, all_notes = [], []
    for rel in EXPECTED:
        need_edges = (rel == "TrainDataset") and not a.no_edges
        _, iss, nts = check_split(a.root, rel, need_edges, a.strict)
        all_issues += iss
        all_notes += nts

    print("=" * 72)
    print("COD DATASET CHECK")
    print("=" * 72)
    if all_notes:
        print("\nOK")
        for n in all_notes:
            print("  +", n)
    if all_issues:
        print("\nNEEDS ATTENTION")
        for i in all_issues:
            print("  !", i)
    else:
        print("\nNo problems found. Safe to start training.")
    print()
    return 1 if (all_issues and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
