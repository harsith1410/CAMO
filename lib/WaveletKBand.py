"""Learnable K-band frequency decomposition (RQ1).

FEDER decomposes features with a fixed Haar DWT and then uses exactly two of
the four subbands: LL feeds the semantic branch and HH feeds the fine-detail
branch (LH and HL are computed and discarded in ``Network.forward``). In the
language of the project plan that is a two-band split, W_LF / W_HF.

This module generalises that to K learnable bands while keeping three
properties the experiment depends on:

1.  **Exact invertibility.**  Every step is a lifting step or an additive
    residual, so analysis followed by synthesis returns the input to machine
    precision for *any* filter values, trained or not.  T16's round-trip test
    is therefore a real check on the implementation rather than on the
    optimiser.

2.  **Haar at initialisation.**  With K=2 and default settings the forward
    pass is numerically identical to ``lib.Modules.DWT``.  The K=2 row of the
    RQ1 table is then the FEDER baseline itself, not an approximation of it,
    so a difference across K cannot be blamed on a changed starting point.

3.  **One spatial resolution for all bands.**  A classic octave cascade
    halves the feature map at every level, and the coarsest encoder stage is
    only 12x12 at 384 input -- three levels would reduce it below 2x2 and K=6
    would be impossible.  Levels beyond the first are therefore *undecimated*
    (a trous): dilated filters peel successively lower-frequency detail off
    the approximation branch without touching resolution.  Every band comes
    out at H/2 x W/2, which is what the band-wise AGFG aggregation in
    ``GCM3K`` expects.

Band ordering is always highest frequency first:

    bands[0]      finest detail   -> fine-detail branch (HH's role in FEDER)
    bands[1:-1]   mid bands       -> edge module (RQ3), ablatable
    bands[-1]     approximation   -> semantic branch (LL's role in FEDER)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Scale constants that make the separable lifting below agree exactly with the
# orthonormal 2D Haar in lib.Modules.DWT. Separable lifting with
# a=(e+o)/2, d=o-e yields LL=(a+b+c+d)/4 and HH=a-b-c+d, whereas the
# orthonormal convention is LL=(a+b+c+d)/2 and HH=(a-b-c+d)/2.
_HAAR_SCALE = {"ll": 2.0, "lh": 1.0, "hl": 1.0, "hh": 0.5}


class _Lift1D(nn.Module):
    """One learnable lifting filter applied along a single spatial axis.

    Depthwise so each feature channel keeps its own filter, and replicate
    padding so the transform does not invent zeros at the border.
    """

    def __init__(self, channels, axis, taps=3, init_center=1.0):
        super().__init__()
        if taps % 2 == 0:
            raise ValueError("taps must be odd, got {}".format(taps))
        self.axis = axis  # 'h' or 'w'
        self.taps = taps
        kernel = (taps, 1) if axis == "h" else (1, taps)
        self.conv = nn.Conv2d(channels, channels, kernel, groups=channels, bias=False)
        # Identity-at-centre initialisation reproduces the Haar lifting step.
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :, taps // 2 if axis == "h" else 0,
                             0 if axis == "h" else taps // 2] = init_center

    def forward(self, x):
        p = self.taps // 2
        pad = (0, 0, p, p) if self.axis == "h" else (p, p, 0, 0)
        return self.conv(F.pad(x, pad, mode="replicate"))


class _LiftingAxis(nn.Module):
    """Predict/update lifting pair along one axis. Invertible by construction."""

    def __init__(self, channels, axis, taps=3):
        super().__init__()
        self.axis = axis
        self.predict = _Lift1D(channels, axis, taps, init_center=1.0)
        self.update = _Lift1D(channels, axis, taps, init_center=0.5)

    def _split(self, x):
        if self.axis == "h":
            return x[:, :, 0::2, :], x[:, :, 1::2, :]
        return x[:, :, :, 0::2], x[:, :, :, 1::2]

    def _merge(self, even, odd):
        if self.axis == "h":
            b, c, h, w = even.shape
            out = even.new_empty(b, c, h * 2, w)
            out[:, :, 0::2, :] = even
            out[:, :, 1::2, :] = odd
            return out
        b, c, h, w = even.shape
        out = even.new_empty(b, c, h, w * 2)
        out[:, :, :, 0::2] = even
        out[:, :, :, 1::2] = odd
        return out

    def forward(self, x):
        even, odd = self._split(x)
        detail = odd - self.predict(even)
        approx = even + self.update(detail)
        return approx, detail

    def inverse(self, approx, detail):
        even = approx - self.update(detail)
        odd = detail + self.predict(even)
        return self._merge(even, odd)


class LiftingDWT2D(nn.Module):
    """Separable learnable 2D wavelet transform, Haar at initialisation.

    Returns (ll, lh, hl, hh) matching ``lib.Modules.DWT`` exactly before any
    training step. ``inverse`` undoes it to machine precision at all times.
    """

    def __init__(self, channels, taps=3, haar_scale=True):
        super().__init__()
        self.row = _LiftingAxis(channels, "h", taps)
        self.col_a = _LiftingAxis(channels, "w", taps)
        self.col_d = _LiftingAxis(channels, "w", taps)
        self.scale = _HAAR_SCALE if haar_scale else {k: 1.0 for k in _HAAR_SCALE}

    @staticmethod
    def _pad_even(x):
        """Lifting decimates by two, so odd sizes must be padded.

        Issue #9 in the upstream repo is exactly this: the sliced Haar silently
        produces mismatched shapes on odd inputs and the error only surfaces
        later, in the aggregation. Padding here makes the transform total.
        """
        h, w = x.shape[-2:]
        ph, pw = h % 2, w % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return x, (h, w)

    def forward(self, x):
        x, orig = self._pad_even(x)
        a_r, d_r = self.row(x)
        ll, hl = self.col_a(a_r)
        lh, hh = self.col_d(d_r)
        s = self.scale
        return (ll * s["ll"], lh * s["lh"], hl * s["hl"], hh * s["hh"]), orig

    def inverse(self, ll, lh, hl, hh, orig=None):
        s = self.scale
        a_r = self.col_a.inverse(ll / s["ll"], hl / s["hl"])
        d_r = self.col_d.inverse(lh / s["lh"], hh / s["hh"])
        x = self.row.inverse(a_r, d_r)
        if orig is not None:
            x = x[:, :, : orig[0], : orig[1]]
        return x


class _AtrousPeel(nn.Module):
    """Undecimated detail extraction: approx = S(x), detail = x - S(x).

    ``S`` is a learnable depthwise dilated filter initialised to the separable
    B-spline kernel [1,2,1]/4 used by the a trous algorithm, so it preserves DC
    and the detail starts with zero mean. Resolution is unchanged and
    ``x = approx + detail`` holds exactly for any filter.
    """

    def __init__(self, channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.smooth = nn.Conv2d(channels, channels, 3, padding=0,
                                dilation=dilation, groups=channels, bias=False)
        with torch.no_grad():
            k = torch.tensor([1.0, 2.0, 1.0]) / 4.0
            self.smooth.weight.copy_(torch.outer(k, k).view(1, 1, 3, 3)
                                     .repeat(channels, 1, 1, 1))

    def forward(self, x):
        p = self.dilation
        approx = self.smooth(F.pad(x, (p, p, p, p), mode="replicate"))
        return approx, x - approx


class LearnableKBandWavelet(nn.Module):
    """K-band learnable decomposition of a feature map.

    Args:
        channels: feature channels (shared across the four encoder scales).
        bands:    K >= 2. K=2 reproduces the FEDER split exactly at init.
        taps:     lifting filter length (odd).
        hf_mode:  how bands[0] is formed from the level-1 detail subbands.
                  ``'hh'``   -- use HH alone, exactly as FEDER does. Default,
                               because it makes the K=2 row the true baseline.
                  ``'learned'`` -- 1x1 conv over (LH, HL, HH), which uses the
                               two subbands the original computes and throws
                               away.
        learnable: False freezes every filter at its Haar / B-spline
                  initialisation, giving a fixed-filter K-band control that
                  isolates "more bands" from "learnable bands".

    Forward returns ``(bands, cache)`` where ``bands`` is a list of K tensors
    at H/2 x W/2 ordered high frequency first, and ``cache`` carries what
    ``reconstruct`` needs for the exact round trip.
    """

    def __init__(self, channels, bands=2, taps=3, hf_mode="hh", learnable=True):
        super().__init__()
        if bands < 2:
            raise ValueError("bands (K) must be at least 2, got {}".format(bands))
        if hf_mode not in ("hh", "learned"):
            raise ValueError("hf_mode must be 'hh' or 'learned', got {!r}".format(hf_mode))
        self.K = bands
        self.hf_mode = hf_mode
        self.dwt = LiftingDWT2D(channels, taps=taps)
        # K-2 undecimated peels off the approximation branch, each one octave
        # lower than the last.
        self.peels = nn.ModuleList(
            [_AtrousPeel(channels, dilation=2 ** i) for i in range(bands - 2)]
        )
        self.hf_fuse = (nn.Conv2d(channels * 3, channels, 1)
                        if hf_mode == "learned" else None)
        if not learnable:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        (ll, lh, hl, hh), orig = self.dwt(x)
        high = hh if self.hf_mode == "hh" else self.hf_fuse(torch.cat([lh, hl, hh], dim=1))

        mids, approx = [], ll
        for peel in self.peels:
            approx, detail = peel(approx)
            mids.append(detail)

        bands = [high] + mids + [approx]
        cache = {"lh": lh, "hl": hl, "hh": hh, "orig": orig,
                 "mids": mids, "approx": approx}
        return bands, cache

    def reconstruct(self, cache):
        """Exact inverse. Used by the T16 round-trip test."""
        ll = cache["approx"]
        for detail in reversed(cache["mids"]):
            ll = ll + detail
        return self.dwt.inverse(ll, cache["lh"], cache["hl"], cache["hh"],
                                cache["orig"])
