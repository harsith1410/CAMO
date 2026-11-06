"""Shared model configuration for Train.py, Test.py and the tools.

Every entry point that builds a Network needs the same architecture flags, and
an architecture mismatch between training and evaluation fails either loudly
(shape error) or, worse, quietly. Defining the flags once here means a
checkpoint can also record exactly what built it.
"""

import argparse
import json
import os

import torch

MODEL_KEYS = ('channels', 'bands', 'taps', 'hf_mode', 'learnable_wavelet',
              'edge_feed', 'ablate_mid', 'gf_radius', 'band_weight')


def add_model_args(parser):
    g = parser.add_argument_group('model / RQ configuration')
    g.add_argument('--channels', type=int, default=96, help='decoder width')
    g.add_argument('--bands', type=int, default=2,
                   help='RQ1: number of frequency bands K. 2 reproduces FEDER.')
    g.add_argument('--taps', type=int, default=3,
                   help='lifting filter length (odd)')
    g.add_argument('--hf_mode', choices=['hh', 'learned'], default='hh',
                   help="'hh' uses HH as the finest band exactly as FEDER does; "
                        "'learned' also uses LH/HL, which the released forward "
                        "pass computes and discards")
    g.add_argument('--learnable_wavelet', action='store_true',
                   help='train the decomposition filters. Off keeps them frozen '
                        'at the Haar / B-spline initialisation, which is the '
                        'fixed-filter control for RQ1.')
    g.add_argument('--edge_feed', choices=['2band', 'kband'], default='2band',
                   help='RQ3: what the edge module sees. 2band is FEDER.')
    g.add_argument('--ablate_mid', action='store_true',
                   help='drop the mid bands while keeping K. Isolates whether '
                        'the mid band is doing the work.')
    g.add_argument('--band_weight', choices=['none', 'static', 'adaptive'],
                   default='none',
                   help='RQ1: per-band weighting of the K bands. none is the '
                        'unweighted control; static learns one weight per '
                        'band; adaptive makes it content-dependent. All three '
                        'start from the identical identity initialisation.')
    g.add_argument('--gf_radius', type=int, default=2,
                   help='guided-filter radius. 2 is the released value and needs '
                        'inputs >= 384; 1 admits 256. Hold it fixed across any '
                        'one comparison.')
    return parser


def model_kwargs(opt):
    """Pull just the architecture flags out of a parsed namespace."""
    return {k: getattr(opt, k) for k in MODEL_KEYS if hasattr(opt, k)}


def build_network(opt, pretrained=True):
    from lib.Network import Network
    kw = model_kwargs(opt)
    ch = kw.pop('channels', 96)
    return Network(ch, pretrained=pretrained, **kw)


def describe(opt):
    """One-line protocol string for logs and results tables."""
    kw = model_kwargs(opt)
    bits = ['K={}'.format(kw.get('bands', 2))]
    if kw.get('learnable_wavelet'):
        bits.append('learnable')
    else:
        bits.append('fixed-filters')
    bits.append('hf={}'.format(kw.get('hf_mode', 'hh')))
    bits.append('edge={}'.format(kw.get('edge_feed', '2band')))
    if kw.get('ablate_mid'):
        bits.append('mid-ablated')
    if kw.get('band_weight', 'none') != 'none':
        bits.append('bw={}'.format(kw['band_weight']))
    bits.append('r={}'.format(kw.get('gf_radius', 2)))
    return ' '.join(bits)


def save_config(path, opt):
    """Write the full run configuration next to the checkpoints."""
    with open(path, 'w') as f:
        json.dump({k: v for k, v in vars(opt).items()
                   if isinstance(v, (int, float, str, bool, type(None)))},
                  f, indent=2, sort_keys=True)


def load_config_for_checkpoint(pth_path):
    """Find the config.json a training run wrote beside its checkpoints.

    Returns a dict of architecture flags, or None. Lets Test.py rebuild the
    right architecture instead of guessing it from the command line.
    """
    cand = os.path.join(os.path.dirname(pth_path), 'config.json')
    if not os.path.exists(cand):
        return None
    with open(cand) as f:
        cfg = json.load(f)
    return {k: cfg[k] for k in MODEL_KEYS if k in cfg}


def load_weights(model, pth_path):
    """Load a checkpoint from any of the three formats this repo has produced.

    The released code saved a bare state_dict; training under DataParallel
    prefixes every key with 'module.'; the current Train.py saves a dict with
    optimizer and epoch alongside the weights.
    """
    ck = torch.load(pth_path, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print('WARNING: checkpoint does not match the architecture exactly.')
        if missing:
            print('  missing {} key(s), e.g. {}'.format(len(missing), missing[:4]))
        if unexpected:
            print('  unexpected {} key(s), e.g. {}'.format(len(unexpected), unexpected[:4]))
        print('  Check the --bands / --learnable_wavelet / --edge_feed flags '
              'match the run that produced this checkpoint.')
    return model
