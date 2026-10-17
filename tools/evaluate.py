#!/usr/bin/env python3
"""Evaluate COD prediction maps: S-alpha, weighted-F, E-phi, MAE.

Two independent implementations are included. `pysodmetrics` is the reference
the community uses; `--crosscheck` additionally runs a from-scratch
implementation written directly from the original papers and reports any
disagreement. If the two agree your numbers are trustworthy.

Metric sources:
  S-measure        Fan et al., ICCV 2017      (alpha = 0.5)
  weighted F       Margolin et al., CVPR 2014 (beta^2 = 1, NOT 0.3)
  E-measure        Fan et al., IJCAI 2018     (adaptive / mean / max)
  MAE              mean |pred - gt|

Usage:
    python tools/evaluate.py --pred preds/COD10K --gt data/TestDataset/COD10K/GT
    python tools/evaluate.py --pred ... --gt ... --crosscheck --expect feder_cod10k
"""
import argparse, os, sys

import cv2
import numpy as np

EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# Published reference rows, for validating your pipeline before you train.
# FEDER (Res2Net-50), He et al., CVPR 2023.
PUBLISHED = {
    "feder_cod10k": dict(sm=0.844, wfm=0.748, em=0.911, mae=0.029,
                         label="FEDER Res2Net-50 on COD10K (CVPR 2023)"),
    "feder_camo":   dict(sm=0.836, wfm=0.807, em=0.897, mae=0.066,
                         label="FEDER Res2Net-50 on CAMO (CVPR 2023)"),
    "feder_nc4k":   dict(sm=0.862, wfm=0.824, em=0.913, mae=0.042,
                         label="FEDER Res2Net-50 on NC4K (CVPR 2023)"),
}

# ----------------------------------------------------------------------------
# From-scratch reference implementation (used only with --crosscheck)
# ----------------------------------------------------------------------------
def _ssim(pred, gt):
    h, w = pred.shape
    N = h * w
    x, y = pred.mean(), gt.mean()
    sx = ((pred - x) ** 2).sum() / (N - 1 + 1e-20)
    sy = ((gt - y) ** 2).sum() / (N - 1 + 1e-20)
    sxy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)
    a = 4 * x * y * sxy
    b = (x * x + y * y) * (sx + sy)
    if a != 0:
        return a / (b + 1e-20)
    return 1.0 if b == 0 else 0.0


def _region(pred, gt):
    """Split at the GT centroid into 4 blocks, area-weighted SSIM."""
    h, w = gt.shape
    total = gt.sum()
    if total == 0:
        cy, cx = h // 2, w // 2
    else:
        ys, xs = np.mgrid[0:h, 0:w]
        cx = int(round((xs * gt).sum() / total))
        cy = int(round((ys * gt).sum() / total))
    cx, cy = max(1, min(w - 1, cx)), max(1, min(h - 1, cy))
    quads = [(slice(0, cy), slice(0, cx)), (slice(0, cy), slice(cx, w)),
             (slice(cy, h), slice(0, cx)), (slice(cy, h), slice(cx, w))]
    area = h * w
    score = 0.0
    for qy, qx in quads:
        g, p = gt[qy, qx], pred[qy, qx]
        if g.size == 0:
            continue
        wgt = g.size / area
        score += wgt * _ssim(p, g)
    return score


def ref_smeasure(pred, gt, alpha=0.5):
    y = gt.mean()
    if y == 0:
        return 1.0 - pred.mean()
    if y == 1:
        return pred.mean()
    fg, bg = pred[gt > 0.5], 1.0 - pred[gt <= 0.5]
    o_fg = 2 * fg.mean() / (fg.mean() ** 2 + 1 + fg.std() + 1e-20) if fg.size else 0.0
    o_bg = 2 * bg.mean() / (bg.mean() ** 2 + 1 + bg.std() + 1e-20) if bg.size else 0.0
    s_obj = y * o_fg + (1 - y) * o_bg
    return max(0.0, alpha * s_obj + (1 - alpha) * _region(pred, gt))


def _enhanced(fm_bin, gt):
    dfm, dgt = fm_bin - fm_bin.mean(), gt - gt.mean()
    align = 2 * dgt * dfm / (dgt * dgt + dfm * dfm + 1e-20)
    return ((align + 1) ** 2 / 4).mean()


def ref_emeasure(pred, gt):
    """Returns (adaptive, mean-over-255-thresholds, max)."""
    adp_t = min(2 * pred.mean(), 1.0)
    adp = _enhanced((pred >= adp_t).astype(np.float64), gt)
    vals = [_enhanced((pred >= t / 255.0).astype(np.float64), gt)
            for t in range(256)]
    return adp, float(np.mean(vals)), float(np.max(vals))


def ref_wfmeasure(pred, gt, beta2=1.0):
    """Margolin et al., CVPR 2014 -- transcribed from the authors' WFb.m."""
    from scipy.ndimage import distance_transform_edt, convolve
    gt_b = gt > 0.5
    if gt_b.sum() == 0:
        return 0.0
    E = np.abs(pred - gt_b.astype(np.float64))
    dst, idx = distance_transform_edt(~gt_b, return_indices=True)
    # dependency: borrow the error of the nearest foreground pixel
    Et = E.copy()
    Et[~gt_b] = E[idx[0][~gt_b], idx[1][~gt_b]]
    # 7x7 gaussian, sigma = sqrt(5)
    k = cv2.getGaussianKernel(7, 5 ** 0.5)
    K = (k @ k.T)
    K /= K.sum()
    EA = convolve(Et, K, mode="constant", cval=0.0)
    MIN = E.copy()
    sel = gt_b & (EA < E)
    MIN[sel] = EA[sel]
    # equal-importance: errors far from the object cost more
    B = np.ones_like(gt, dtype=np.float64)
    B[~gt_b] = 2 - np.exp(np.log(0.5) / 5 * dst[~gt_b])
    Ew = MIN * B
    TPw = gt_b.sum() - Ew[gt_b].sum()
    FPw = Ew[~gt_b].sum()
    R = 1 - Ew[gt_b].mean()
    P = TPw / (TPw + FPw + 1e-20)
    if P + R == 0:
        return 0.0
    return float((1 + beta2) * R * P / (beta2 * P + R + 1e-20))


# ----------------------------------------------------------------------------
def prepare(pred, gt):
    """Match the canonical preprocessing every COD toolbox uses.

    Predictions are min-max normalised (MATLAB's mapminmax on im2double) and
    ground truth is binarised at >128. This matters: a sigmoid output that
    never reaches 0 or 1 gets rescaled, and scoring without this step is a
    common reason reproduction numbers come out below the published ones.
    """
    gt_b = gt > 128
    p = pred.astype(np.float64) / 255.0
    if p.max() != p.min():
        p = (p - p.min()) / (p.max() - p.min())
    return p, gt_b


def load_pair(pred_path, gt_path):
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if pred is None or gt is None:
        return None, None
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    return pred, gt


def main():
    ap = argparse.ArgumentParser(description="Evaluate COD prediction maps.")
    ap.add_argument("--pred", required=True, help="directory of predicted maps")
    ap.add_argument("--gt", required=True, help="directory of ground-truth masks")
    ap.add_argument("--crosscheck", action="store_true",
                    help="also run the from-scratch implementation and compare (slow)")
    ap.add_argument("--expect", choices=sorted(PUBLISHED),
                    help="compare against a published row to validate the pipeline")
    ap.add_argument("--tol", type=float, default=0.003,
                    help="tolerance when using --expect (default 0.003)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N (debugging)")
    a = ap.parse_args()

    for d, n in ((a.pred, "--pred"), (a.gt, "--gt")):
        if not os.path.isdir(d):
            print(f"ERROR: {n} {d} is not a directory")
            return 2

    gt_files = {os.path.splitext(f)[0]: f for f in os.listdir(a.gt)
                if f.lower().endswith(EXTS) and not f.startswith(".")}
    pred_files = {os.path.splitext(f)[0]: f for f in os.listdir(a.pred)
                  if f.lower().endswith(EXTS) and not f.startswith(".")}
    common = sorted(set(gt_files) & set(pred_files))
    if not common:
        print("ERROR: no matching filenames between --pred and --gt")
        return 2
    missing = set(gt_files) - set(pred_files)
    if missing:
        print(f"WARNING: {len(missing)} ground-truth mask(s) have no prediction. "
              f"Evaluating {len(common)} of {len(gt_files)}. "
              f"Published numbers cover the full set, so this is NOT comparable.")
    if a.limit:
        common = common[:a.limit]
        print(f"(--limit {a.limit}: partial run, not comparable)")

    try:
        from py_sod_metrics import Smeasure, Emeasure, WeightedFmeasure, MAE
    except ImportError:
        print("ERROR: pysodmetrics is not installed.  pip install pysodmetrics")
        return 2

    SM, EM, WFM, M = Smeasure(alpha=0.5), Emeasure(), WeightedFmeasure(beta=1), MAE()
    ref = {"sm": [], "wfm": [], "adp": [], "em": [], "mae": []}

    for i, k in enumerate(common, 1):
        p, g = load_pair(os.path.join(a.pred, pred_files[k]),
                         os.path.join(a.gt, gt_files[k]))
        if p is None:
            print(f"  unreadable pair for {k}, skipping")
            continue
        SM.step(pred=p, gt=g); EM.step(pred=p, gt=g)
        WFM.step(pred=p, gt=g); M.step(pred=p, gt=g)
        if a.crosscheck:
            pf, gb = prepare(p, g)
            gf = gb.astype(np.float64)
            ref["mae"].append(np.abs(pf - gf).mean())
            ref["sm"].append(ref_smeasure(pf, gf))
            ref["wfm"].append(ref_wfmeasure(pf, gf))
            adp, mean_e, _ = ref_emeasure(pf, gf)
            ref["adp"].append(adp); ref["em"].append(mean_e)
        if i % 250 == 0:
            print(f"  {i}/{len(common)}")

    e = EM.get_results()["em"]
    res = dict(sm=SM.get_results()["sm"],
               wfm=WFM.get_results()["wfm"],
               em_mean=float(e["curve"].mean()), em_max=float(e["curve"].max()),
               em_adp=float(e["adp"]), mae=M.get_results()["mae"])

    print("\n" + "=" * 62)
    print(f"RESULTS over {len(common)} images   ({os.path.basename(a.pred.rstrip('/'))})")
    print("=" * 62)
    print(f"  S-alpha        {res['sm']:.4f}   (higher better)")
    print(f"  F-w-beta       {res['wfm']:.4f}   (higher better, beta^2=1)")
    print(f"  E-phi mean     {res['em_mean']:.4f}   (higher better)")
    print(f"  E-phi max      {res['em_max']:.4f}")
    print(f"  E-phi adaptive {res['em_adp']:.4f}")
    print(f"  MAE            {res['mae']:.4f}   (lower better)")
    print("\nReport mean E-phi unless you say otherwise -- adaptive, mean and max")
    print("are different numbers and are not comparable across papers.")

    rc = 0
    if a.crosscheck:
        print("\n" + "-" * 62)
        print("CROSS-CHECK: library vs from-scratch implementation")
        print("-" * 62)
        pairs = [("S-alpha", res["sm"], np.mean(ref["sm"])),
                 ("F-w-beta", res["wfm"], np.mean(ref["wfm"])),
                 ("E-phi mean", res["em_mean"], np.mean(ref["em"])),
                 ("E-phi adp", res["em_adp"], np.mean(ref["adp"])),
                 ("MAE", res["mae"], np.mean(ref["mae"]))]
        worst = 0.0
        for name, lib, mine in pairs:
            d = abs(lib - mine)
            worst = max(worst, d)
            print(f"  {name:<12} {lib:.4f}  vs  {mine:.4f}   diff {d:.5f}"
                  f"{'   <-- CHECK' if d > 0.005 else ''}")
        print(f"\n  worst disagreement {worst:.5f}",
              "-- implementations agree" if worst <= 0.005 else "-- INVESTIGATE before trusting these numbers")
        if worst > 0.005:
            rc = 1

    if a.expect:
        exp = PUBLISHED[a.expect]
        print("\n" + "-" * 62)
        print(f"VALIDATION against {exp['label']}")
        print("-" * 62)
        checks = [("S-alpha", res["sm"], exp["sm"]), ("F-w-beta", res["wfm"], exp["wfm"]),
                  ("E-phi mean", res["em_mean"], exp["em"]), ("MAE", res["mae"], exp["mae"])]
        ok = True
        for name, got, want in checks:
            d = abs(got - want)
            good = d <= a.tol
            ok &= good
            print(f"  {name:<12} got {got:.4f}  published {want:.4f}  "
                  f"diff {d:.4f}  {'OK' if good else 'OUT OF TOLERANCE'}")
        print("\n  " + ("Pipeline validated -- your evaluation matches the paper."
                        if ok else
                        "Does NOT match. Fix this before training: every later number "
                        "inherits the error.\n  Common causes: wrong test split, "
                        "resized predictions, or the wrong E-measure variant."))
        if not ok:
            rc = 1
    print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
