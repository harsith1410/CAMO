#!/usr/bin/env python3
"""Measure parameters, FLOPs and inference latency for a model configuration.

The FEDER paper claims "cheaper computational and memory costs" but publishes
no table -- only a scatter plot -- so there is no number to cite. This
measures them directly, and because it takes the same architecture flags as
training it also answers what K costs, which the RQ1 table needs alongside the
accuracy column.

Usage:
    python tools/measure_cost.py --testsize 384
    python tools/measure_cost.py --sweep 2 3 4 6 --learnable_wavelet
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lib.Network import Network
from lib.config import add_model_args, model_kwargs


def build(opt, bands=None):
    kw = model_kwargs(opt)
    channels = kw.pop('channels', 96)
    if bands is not None:
        kw['bands'] = bands
        kw['learnable_wavelet'] = kw.get('learnable_wavelet', False) or bands != 2
    # pretrained=False: ImageNet weights change no shape and so no cost, and
    # downloading them to count FLOPs would be wasteful.
    return Network(channels, pretrained=False, **kw)


def count(model, size, device):
    model = model.to(device).eval()
    x = torch.rand(1, 3, size, size, device=device)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops = None
    try:
        from thop import profile
        import copy
        # thop mutates the model by attaching buffers; profile a copy.
        flops, _ = profile(copy.deepcopy(model), inputs=(x,), verbose=False)
    except ImportError:
        print('  (thop not installed -- pip install thop for the FLOPs column)')
    except Exception as exc:
        print('  (thop failed: {}: {})'.format(type(exc).__name__, exc))

    latency = peak = None
    if device.type == 'cuda':
        with torch.no_grad():
            for _ in range(5):
                model(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(20):
                model(x)
            torch.cuda.synchronize()
            latency = (time.perf_counter() - t0) / 20 * 1000
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
    return params, trainable, flops, latency, peak


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--testsize', type=int, default=384)
    ap.add_argument('--sweep', type=int, nargs='*', default=None,
                    help='measure these K values instead of a single config')
    ap.add_argument('--cpu', action='store_true', help='force CPU (no latency column)')
    add_model_args(ap)
    opt = ap.parse_args()

    device = torch.device('cpu' if opt.cpu or not torch.cuda.is_available() else 'cuda')
    if device.type == 'cuda':
        print('device: {}'.format(torch.cuda.get_device_name(0)))
    print('input: {}x{}   gf_radius={}\n'.format(opt.testsize, opt.testsize, opt.gf_radius))

    header = '{:<22} {:>10} {:>12} {:>11} {:>10}'.format(
        'config', 'params(M)', 'GFLOPs', 'latency(ms)', 'peak(MiB)')
    print(header)
    print('-' * len(header))

    configs = [(None, 'K={} (flags)'.format(opt.bands))] if not opt.sweep else \
              [(k, 'K={}'.format(k)) for k in opt.sweep]

    for bands, label in configs:
        model = build(opt, bands)
        try:
            p, tr, fl, lat, peak = count(model, opt.testsize, device)
        except Exception as exc:
            print('{:<22} FAILED {}: {}'.format(label, type(exc).__name__, str(exc)[:60]))
            del model
            torch.cuda.empty_cache()
            continue
        print('{:<22} {:>10.2f} {:>12} {:>11} {:>10}'.format(
            label, p / 1e6,
            '{:.1f}'.format(fl / 1e9) if fl else 'n/a',
            '{:.1f}'.format(lat) if lat else 'n/a',
            '{:.0f}'.format(peak) if peak else 'n/a'))
        if tr != p:
            print('{:<22} {:>10.2f} trainable'.format('', tr / 1e6))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print('\nFLOPs are thop multiply-accumulates for one 1x{s}x{s} image; papers '
          'that report\n"FLOPs" often mean MACs, so state which when you cite '
          'this.'.format(s=opt.testsize))
    return 0


if __name__ == '__main__':
    sys.exit(main())
