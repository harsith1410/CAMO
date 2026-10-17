#!/usr/bin/env python3
"""Generate the Edge/ maps FEDER's dataloader expects, from GT masks.

No public COD release ships edge maps; the FEDER authors' repackaged dataset
did, and its download links are dead. This reconstructs them.

Default method is the morphological gradient (dilate - erode), which gives a
boundary centred on the mask edge. Width is controlled by --kernel.

Usage:
    python tools/make_edges.py --gt /content/data/TrainDataset/GT \
                               --out /content/data/TrainDataset/Edge
"""
import argparse, os, sys

import cv2
import numpy as np

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def edge_from_mask(mask, method="gradient", kernel=3):
    """mask: uint8 HxW. Returns uint8 HxW edge map in {0,255}."""
    binary = (mask > 127).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    if method == "gradient":
        # symmetric about the true boundary
        return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, k)
    if method == "inner":
        # boundary lies strictly inside the object
        return cv2.subtract(binary, cv2.erode(binary, k))
    if method == "outer":
        return cv2.subtract(cv2.dilate(binary, k), binary)
    if method == "canny":
        return cv2.Canny(binary, 100, 200)
    raise ValueError(f"unknown method {method}")


def main():
    ap = argparse.ArgumentParser(description="Generate edge maps from GT masks.")
    ap.add_argument("--gt", required=True, help="directory of ground-truth masks")
    ap.add_argument("--out", required=True, help="directory to write edge maps into")
    ap.add_argument("--method", default="gradient",
                    choices=["gradient", "inner", "outer", "canny"])
    ap.add_argument("--kernel", type=int, default=3,
                    help="structuring element size; larger means thicker edges (default 3)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if not os.path.isdir(a.gt):
        print(f"ERROR: --gt {a.gt} is not a directory")
        return 2
    if a.kernel % 2 == 0:
        print(f"ERROR: --kernel must be odd, got {a.kernel}")
        return 2
    os.makedirs(a.out, exist_ok=True)

    files = sorted(f for f in os.listdir(a.gt)
                   if f.lower().endswith(EXTS) and not f.startswith("."))
    if not files:
        print(f"ERROR: no mask images found in {a.gt}")
        return 2

    written = skipped = failed = 0
    empty = []
    for i, f in enumerate(files, 1):
        stem = os.path.splitext(f)[0]
        dst = os.path.join(a.out, stem + ".png")
        if os.path.exists(dst) and not a.overwrite:
            skipped += 1
            continue
        m = cv2.imread(os.path.join(a.gt, f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"  could not read {f}")
            failed += 1
            continue
        e = edge_from_mask(m, a.method, a.kernel)
        if e.max() == 0:
            empty.append(f)
        cv2.imwrite(dst, e)
        written += 1
        if i % 500 == 0:
            print(f"  {i}/{len(files)}")

    print(f"\nwrote {written}, skipped {skipped} existing, failed {failed}")
    print(f"method={a.method} kernel={a.kernel} -> {a.out}")
    if empty:
        print(f"\nWARNING: {len(empty)} mask(s) produced a blank edge map "
              f"(the mask is probably all-foreground or all-background): {empty[:5]}")
    if skipped and not a.overwrite:
        print("Pass --overwrite to regenerate the skipped files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
