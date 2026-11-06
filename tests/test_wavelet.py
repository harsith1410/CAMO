#!/usr/bin/env python3
"""T16: unit tests for the K-band decomposition.

Runs under pytest, or standalone with ``python tests/test_wavelet.py``.

What these actually guarantee, and why each matters to the experiment:

  * reconstruction round-trip ~ 0  -- the decomposition loses nothing, so a
    band carries information rather than an artefact of a lossy transform;
  * Haar equivalence at init      -- the K=2 row of the RQ1 table is the FEDER
    baseline itself, so a difference across K cannot come from a changed
    starting point;
  * identical band shapes         -- band-wise aggregation is well posed at
    every K, including K=6 where an octave cascade would have collapsed the
    coarsest encoder stage;
  * the documented resolution floor -- 256 fails on the released architecture
    as surely as on this one, and the test pins that down so nobody spends GPU
    hours discovering it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lib.Modules import DWT
from lib.ModulesK import min_input_size
from lib.Network import Network
from lib.WaveletKBand import LearnableKBandWavelet, LiftingDWT2D

TOL = 1e-10          # float64 round-trip; the measured error is ~1e-15


def test_haar_equivalence_at_init():
    """The learnable lifting reproduces the fixed Haar DWT exactly at init."""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 48, 48, dtype=torch.float64)
    (ll, lh, hl, hh), _ = LiftingDWT2D(8).double()(x)
    rll, rlh, rhl, rhh = DWT()(x)
    for name, got, want in (('LL', ll, rll), ('LH', lh, rlh),
                            ('HL', hl, rhl), ('HH', hh, rhh)):
        err = (got - want).abs().max().item()
        assert err < TOL, '{} differs from Haar by {:.2e}'.format(name, err)


def test_roundtrip_all_k():
    """Analysis then synthesis returns the input, for every K and both parities."""
    torch.manual_seed(0)
    for size in (32, 48, 64):
        x = torch.randn(2, 6, size, size, dtype=torch.float64)
        for k in (2, 3, 4, 6, 8):
            w = LearnableKBandWavelet(6, bands=k).double()
            bands, cache = w(x)
            err = (w.reconstruct(cache) - x).abs().max().item()
            assert err < TOL, 'K={} size={} round-trip error {:.2e}'.format(k, size, err)
            assert len(bands) == k


def test_roundtrip_survives_training():
    """Invertibility is structural, not a property of the initial filters."""
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32, 32, dtype=torch.float64)
    w = LearnableKBandWavelet(6, bands=4).double()
    with torch.no_grad():        # perturb every filter away from Haar
        for p in w.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    _, cache = w(x)
    err = (w.reconstruct(cache) - x).abs().max().item()
    assert err < TOL, 'round-trip broke after perturbing filters: {:.2e}'.format(err)


def test_odd_sizes_are_handled():
    """Odd inputs are padded rather than silently producing mismatched shapes.

    This is upstream issue #9 / #4: the sliced Haar returns ceil/floor sized
    subbands on odd inputs and the failure only surfaces later.
    """
    for h, w_ in ((33, 33), (31, 45), (17, 16)):
        x = torch.randn(1, 4, h, w_, dtype=torch.float64)
        wav = LearnableKBandWavelet(4, bands=3).double()
        bands, cache = wav(x)
        shapes = {tuple(b.shape) for b in bands}
        assert len(shapes) == 1, 'bands disagree on shape at {}x{}: {}'.format(h, w_, shapes)
        assert (wav.reconstruct(cache) - x).abs().max().item() < TOL


def test_all_bands_share_one_resolution():
    """Every band is H/2 x W/2 regardless of K -- what GCM3K's aggregation needs."""
    x = torch.randn(1, 4, 24, 24)
    for k in (2, 3, 4, 6):
        bands, _ = LearnableKBandWavelet(4, bands=k)(x)
        for b in bands:
            assert b.shape[-2:] == (12, 12), 'K={} band shape {}'.format(k, b.shape)


def test_frozen_filters_have_no_gradient():
    """learnable=False gives a genuine fixed-filter control for RQ1."""
    w = LearnableKBandWavelet(4, bands=4, learnable=False)
    assert all(not p.requires_grad for p in w.parameters())
    w2 = LearnableKBandWavelet(4, bands=4, learnable=True)
    assert any(p.requires_grad for p in w2.parameters())


def test_gradients_reach_every_band():
    """No band is silently dropped from the graph.

    The reduction is deliberately weighted rather than a plain ``sum()``: the
    a trous detail bands are DC-free (see the test below), so the gradient of
    their *sum* is zero by construction and a sum-based check would report a
    false failure.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 4, 24, 24, dtype=torch.float64, requires_grad=True)
    w = LearnableKBandWavelet(4, bands=4).double()
    bands, _ = w(x)
    for i, b in enumerate(bands):
        x.grad = None
        (b * torch.randn_like(b)).sum().backward(retain_graph=True)
        assert x.grad is not None and x.grad.abs().sum() > 0, \
            'band {} has no gradient'.format(i)


def test_detail_bands_are_dc_free():
    """Mid bands carry no DC, which is what makes them band-*pass*.

    The B-spline smoother preserves DC, so detail = x - S(x) removes it. If
    this ever fails the mid bands have started duplicating the approximation
    band and the RQ1 comparison stops being meaningful.
    """
    # A constant field is pure DC: a band-pass filter must annihilate it.
    x = torch.full((2, 4, 32, 32), 3.7, dtype=torch.float64)
    bands, _ = LearnableKBandWavelet(4, bands=4).double()(x)
    for i, b in enumerate(bands[1:-1], start=1):
        dc = b.abs().max().item()
        assert dc < 1e-9, 'mid band {} responds to a constant input: {:.2e}'.format(i, dc)
    # ...while the approximation band must preserve it.
    assert bands[-1].mean().item() > 1.0, 'approximation band lost the DC component'


def test_k2_network_matches_released_path():
    """bands=2 with frozen filters selects the untouched GCM3, so old
    checkpoints still load and the baseline cannot drift."""
    m = Network(96, bands=2, learnable_wavelet=False, pretrained=False)
    assert not m.use_kband
    assert type(m.GCM3).__name__ == 'GCM3'
    k = Network(96, bands=3, learnable_wavelet=True, pretrained=False)
    assert k.use_kband and type(k.GCM3).__name__ == 'GCM3K'


def test_resolution_floor_is_documented_and_enforced():
    """256 is rejected with an explanation rather than a bare AssertionError."""
    assert min_input_size(2) == 384 and min_input_size(1) == 256
    m = Network(96, bands=2, pretrained=False)
    try:
        m(torch.rand(1, 3, 256, 256))
    except ValueError as exc:
        assert '384' in str(exc), 'error should name the usable sizes'
    else:
        raise AssertionError('256 should have been rejected at gf_radius=2')


def test_ablate_mid_removes_mid_bands():
    """The mid-band ablation keeps K's capacity but stops using the mid bands."""
    m = Network(96, bands=4, learnable_wavelet=True, edge_feed='kband',
                ablate_mid=True, pretrained=False)
    assert m.mid_f2 is None, 'mid fusion should not be built when ablating'
    full = Network(96, bands=4, learnable_wavelet=True, edge_feed='kband',
                   pretrained=False)
    assert full.mid_f2 is not None


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print('  PASS  {}'.format(t.__name__))
        except Exception as exc:
            failed += 1
            print('  FAIL  {}: {}: {}'.format(t.__name__, type(exc).__name__, exc))
    print('\n{}/{} passed'.format(len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
