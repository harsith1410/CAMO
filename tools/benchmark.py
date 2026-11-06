#!/usr/bin/env python3
"""Predict and score a checkpoint on the COD test sets, appending a results row.

Combines what would otherwise be three manual steps -- Test.py, then
tools/evaluate.py per dataset, then copying numbers into a table -- into one
command, so the Results tables are generated rather than transcribed.

The architecture is read from the config.json the training run wrote beside
its checkpoints, so a K-band run is scored with the model that
produced it rather than with whatever flags were typed last.

Usage:
    python tools/benchmark.py --pth snapshot/k3/Net_epoch_best.pth \
        --datasets COD10K --tag "K=3 learnable"
    python tools/benchmark.py --pth snapshot/final/Net_epoch_best.pth \
        --datasets COD10K NC4K CAMO CHAMELEON --tag final
"""
import argparse
import csv
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.Network import Network
from lib.config import (add_model_args, model_kwargs, load_weights,
                        load_config_for_checkpoint, describe)
from utils.data_val import test_dataset

ALL_SETS = ['COD10K', 'NC4K', 'CAMO', 'CHAMELEON']

# Published FEDER rows (He et al., CVPR 2023), for the gap column.
PUBLISHED = {
    'COD10K': dict(sm=0.844, wfm=0.748, em=0.911, mae=0.029),
    'CAMO':   dict(sm=0.836, wfm=0.807, em=0.897, mae=0.066),
    'NC4K':   dict(sm=0.862, wfm=0.824, em=0.913, mae=0.042),
}

FIELDS = ['timestamp', 'tag', 'checkpoint', 'dataset', 'protocol', 'n_images',
          'sm', 'wfm', 'em_mean', 'em_adp', 'mae', 'boundary_f',
          'pub_sm', 'd_sm', 'notes']


def boundary_f(pred, gt, tol=2, thr=0.5):
    """Boundary F-measure: F1 over mask contours within ``tol`` pixels.

    RQ3 is about edge quality, and S-alpha moves too little to show it: a
    model can fix a boundary everywhere and shift S-alpha by 0.002. This
    scores the contours directly.
    """
    pb = (pred > thr).astype(np.uint8)
    gb = (gt > 0.5).astype(np.uint8)
    if pb.sum() == 0 or gb.sum() == 0:
        return np.nan
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pe = cv2.morphologyEx(pb, cv2.MORPH_GRADIENT, k)
    ge = cv2.morphologyEx(gb, cv2.MORPH_GRADIENT, k)
    if pe.sum() == 0 or ge.sum() == 0:
        return np.nan
    d = 2 * tol + 1
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    # A predicted edge counts as correct if a GT edge lies within tol pixels.
    prec = (pe & cv2.dilate(ge, dk)).sum() / max(1, pe.sum())
    rec = (ge & cv2.dilate(pe, dk)).sum() / max(1, ge.sum())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def run_dataset(model, root, name, testsize, device, save_dir=None,
                limit=0, want_boundary=True):
    from py_sod_metrics import Smeasure, Emeasure, WeightedFmeasure, MAE

    loader = test_dataset(os.path.join(root, name, 'Imgs') + '/',
                          os.path.join(root, name, 'GT') + '/', testsize)
    SM, EM, WFM, M = Smeasure(alpha=0.5), Emeasure(), WeightedFmeasure(beta=1), MAE()
    bf = []
    n = loader.size if not limit else min(limit, loader.size)
    for i in range(n):
        image, gt, img_name, _ = loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt = gt / (gt.max() + 1e-8)
        with torch.no_grad():
            out = model(image.to(device))
        res = F.interpolate(out[4], size=gt.shape, mode='bilinear',
                            align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # py_sod_metrics expects uint8 prediction and uint8 GT, matching the
        # MATLAB toolbox every published COD number was produced with.
        p8 = (res * 255).astype(np.uint8)
        g8 = (gt * 255).astype(np.uint8)
        SM.step(pred=p8, gt=g8); EM.step(pred=p8, gt=g8)
        WFM.step(pred=p8, gt=g8); M.step(pred=p8, gt=g8)
        if want_boundary:
            bf.append(boundary_f(res, gt))
        if save_dir:
            cv2.imwrite(os.path.join(save_dir,
                                     os.path.splitext(img_name)[0] + '.png'), p8)
        if (i + 1) % 250 == 0:
            print('    {}/{}'.format(i + 1, n))

    e = EM.get_results()['em']
    return dict(n_images=n,
                sm=SM.get_results()['sm'],
                wfm=WFM.get_results()['wfm'],
                em_mean=float(e['curve'].mean()),
                em_adp=float(e['adp']),
                mae=M.get_results()['mae'],
                boundary_f=float(np.nanmean(bf)) if want_boundary and bf else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pth', required=True)
    ap.add_argument('--data_root', default='./data/USEDATA/TestDataset')
    ap.add_argument('--datasets', nargs='+', default=['COD10K'], choices=ALL_SETS)
    ap.add_argument('--testsize', type=int, default=384)
    ap.add_argument('--tag', default='', help='label for the results row')
    ap.add_argument('--notes', default='', help='free text, e.g. protocol deviations')
    ap.add_argument('--csv', default='./results/results.csv')
    ap.add_argument('--save_maps', action='store_true',
                    help='also write the prediction PNGs (needed for figures)')
    ap.add_argument('--save_root', default='./res')
    ap.add_argument('--limit', type=int, default=0, help='debugging only')
    ap.add_argument('--no_boundary', action='store_true',
                    help='skip boundary-F (it is the slow metric)')
    ap.add_argument('--gpu_id', type=str, default='0')
    add_model_args(ap)
    opt = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = load_config_for_checkpoint(opt.pth)
    if cfg:
        for k, v in cfg.items():
            setattr(opt, k, v)
        print('architecture from config.json beside the checkpoint')
    else:
        print('no config.json beside the checkpoint; using command-line flags')
    protocol = '{}px {}'.format(opt.testsize, describe(opt))
    print('CONFIG: ' + protocol + '\n')

    kw = model_kwargs(opt)
    channels = kw.pop('channels', 96)
    model = Network(channels, pretrained=False, **kw)
    load_weights(model, opt.pth)
    model.to(device).eval()

    run_name = os.path.basename(os.path.dirname(opt.pth.rstrip('/'))) or 'run'
    rows = []
    for name in opt.datasets:
        if not os.path.isdir(os.path.join(opt.data_root, name)):
            print('skipping {}: not found under {}'.format(name, opt.data_root))
            continue
        save_dir = None
        if opt.save_maps:
            save_dir = os.path.join(opt.save_root, run_name, name)
            os.makedirs(save_dir, exist_ok=True)
        print('  {} ...'.format(name))
        r = run_dataset(model, opt.data_root, name, opt.testsize, device,
                        save_dir, opt.limit, not opt.no_boundary)
        pub = PUBLISHED.get(name)
        r.update(timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
                 tag=opt.tag or run_name, checkpoint=opt.pth, dataset=name,
                 protocol=protocol, notes=opt.notes,
                 pub_sm=pub['sm'] if pub else '',
                 d_sm=round(r['sm'] - pub['sm'], 4) if pub else '')
        rows.append(r)

    if not rows:
        print('nothing evaluated')
        return 2

    hdr = '{:<11} {:>7} {:>8} {:>8} {:>8} {:>7} {:>8} {:>8}'.format(
        'dataset', 'n', 'S-alpha', 'F-w-beta', 'E-phi', 'MAE', 'bound-F', 'd(S) vs pub')
    print('\n' + '=' * len(hdr)); print(hdr); print('=' * len(hdr))
    for r in rows:
        print('{:<11} {:>7} {:>8.4f} {:>8.4f} {:>8.4f} {:>7.4f} {:>8} {:>8}'.format(
            r['dataset'], r['n_images'], r['sm'], r['wfm'], r['em_mean'], r['mae'],
            '{:.4f}'.format(r['boundary_f']) if r['boundary_f'] == r['boundary_f'] else 'n/a',
            r['d_sm'] if r['d_sm'] != '' else 'n/a'))

    os.makedirs(os.path.dirname(opt.csv) or '.', exist_ok=True)
    new = not os.path.exists(opt.csv)
    with open(opt.csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print('\nappended {} row(s) to {}'.format(len(rows), opt.csv))
    print('Report mean E-phi. A gain of ~0.01 S-alpha is real; 0.002-0.003 is')
    print('run-to-run noise and needs multiple seeds before you claim it.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
