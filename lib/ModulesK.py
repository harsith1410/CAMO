"""K-band variants of the FEDER aggregation blocks (RQ1 / RQ3).

Kept in a separate file so ``lib/Modules.py`` stays byte-identical to the
released code and the baseline cannot drift while the extension is developed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.Modules import ETM, AGF, AGFG, GF, BasicConv2d
from lib.WaveletKBand import LearnableKBandWavelet


class AGFr(AGF):
    """AGF with a configurable guided-filter radius.

    ``AGF`` hardcodes ``GF(r=2)``, whose ``h_lrx > 2r + 1`` assertion sets a
    hard floor on input resolution: the coarsest band must exceed 5 px, which
    needs a 384 input. 256 and 320 both fail there, in the *released* model as
    much as in this one, so the plan's "use 256" reduced protocol is not
    available at r=2. Lowering the radius lifts that floor at the cost of a
    narrower guided-filter window.
    """

    def __init__(self, channels, lf, r=2, eps=1e-2):
        super().__init__(channels, lf)
        self.GF = GF(r=r, eps=eps)


class AGFGr(nn.Module):
    """AGFG with a configurable guided-filter radius. r=2 is the released value."""

    def __init__(self, channels, lf, r=2):
        super().__init__()
        self.GF1 = AGFr(channels, lf, r)
        self.GF2 = AGFr(channels, lf, r)
        self.GF3 = AGFr(channels, lf, r)

    def forward(self, f1, f2, f3, f4):
        y = self.GF1(f2, f1)
        y = self.GF2(f3, y)
        y = self.GF3(f4, y)
        return y


def min_input_size(gf_radius=2):
    """Smallest network input the guided filter admits, in pixels.

    The chain is: input/16 -> wavelet halves it -> AGF halves it again to form
    ``high_level_small``, which must exceed ``2r + 1``. Sizes must also leave
    every intermediate even, which in practice means a multiple of 64.
    """
    need = 2 * gf_radius + 2          # strictly greater than 2r+1
    size = 64
    while size // 64 < need:
        size += 64
    return size


class BandWeight(nn.Module):
    """Per-band adaptive weighting of the K aggregated bands.

    The proposal asks for the K-band decomposition to carry "per-band adaptive
    weighting". Two strengths are provided so the claim can be ablated rather
    than assumed:

    ``static``    one learnable logit per band, softmax-normalised across bands
                  and multiplied by K. The balance between bands is learned but
                  fixed once trained.
    ``adaptive``  the static weight times a content-dependent gate computed
                  from each band's own channel statistics, so the balance can
                  shift from image to image. This is the mode the proposal
                  describes.

    Both modes compute *exactly* the identity at initialisation: the logits
    start at zero so ``softmax(0) * K == 1``, and the gate's last layer is
    zeroed so ``2 * sigmoid(0) == 1``. So the module changes nothing about the
    function the network computes at step 0, and whatever the weighted run
    gains is something the optimiser found rather than a luckier starting
    point. (It is not the same *model* as an unweighted one: constructing the
    gates draws from the RNG, which shifts the init of every layer built
    afterwards. Only the forward pass is unchanged.)
    """

    def __init__(self, channels, bands, mode="none", reduction=4):
        super().__init__()
        if mode not in ("none", "static", "adaptive"):
            raise ValueError("band_weight must be none/static/adaptive, got {!r}"
                             .format(mode))
        self.mode = mode
        self.K = bands
        if mode == "none":
            return
        self.logits = nn.Parameter(torch.zeros(bands))
        if mode == "adaptive":
            hidden = max(4, channels // reduction)
            self.gate = nn.ModuleList([
                nn.Sequential(nn.Linear(channels, hidden),
                              nn.ReLU(inplace=True),
                              nn.Linear(hidden, channels))
                for _ in range(bands)])
            for g in self.gate:
                nn.init.zeros_(g[-1].weight)
                nn.init.zeros_(g[-1].bias)

    def band_weights(self):
        """The learned static weight per band, high frequency first.

        Worth reporting: it says which parts of the spectrum the model decided
        to rely on, which is the actual question behind RQ1.
        """
        if self.mode == "none":
            return torch.ones(self.K)
        return torch.softmax(self.logits.detach(), 0) * self.K

    def forward(self, bands):
        if self.mode == "none":
            return bands
        w = torch.softmax(self.logits, 0) * self.K
        out = []
        for k, b in enumerate(bands):
            s = w[k] * b
            if self.mode == "adaptive":
                g = 2.0 * torch.sigmoid(self.gate[k](b.mean(dim=(2, 3))))
                s = s * g[:, :, None, None]
            out.append(s)
        return out


class GCM3K(nn.Module):
    """Deep wavelet-like decomposition with K learnable bands.

    Mirrors ``lib.Modules.GCM3``: the four encoder stages are projected to a
    common width by ETM, decomposed, and then aggregated *band-wise* across
    scales by one AGFG per band. The only change is that the fixed Haar's four
    subbands become K learnable bands.

    The lowest band keeps ``lf=True`` (instance norm, average-pooled channel
    attention) exactly as LL did; every higher band uses ``lf=False``, as the
    detail subbands did.

    Returns ``(bands, f1, f2, f3, f4)`` with ``bands`` ordered high frequency
    first and all K entries at the resolution GCM3 produced for LL/HH.
    """

    def __init__(self, in_channels, out_channels, bands=2, taps=3,
                 hf_mode="hh", learnable=True, gf_radius=2,
                 band_weight="none"):
        super().__init__()
        self.K = bands
        self.T1 = ETM(in_channels, out_channels)
        self.T2 = ETM(in_channels * 2, out_channels)
        self.T3 = ETM(in_channels * 4, out_channels)
        self.T4 = ETM(in_channels * 8, out_channels)

        self.wavelet = LearnableKBandWavelet(
            out_channels, bands=bands, taps=taps, hf_mode=hf_mode,
            learnable=learnable)
        # One aggregation path per band. lf=True only for the approximation.
        self.AGFG = nn.ModuleList(
            [AGFGr(out_channels, lf=(k == bands - 1), r=gf_radius)
             for k in range(bands)])
        # Per-band adaptive weighting. Identity at init, so band_weight='none'
        # and a freshly built weighted model are the same starting point.
        self.band_weight = BandWeight(out_channels, bands, mode=band_weight)

    def forward(self, f1, f2, f3, f4):
        f1, f2, f3, f4 = self.T1(f1), self.T2(f2), self.T3(f3), self.T4(f4)
        # Same wavelet instance for every scale, as the shared DWT was.
        wf = [self.wavelet(f)[0] for f in (f1, f2, f3, f4)]
        bands = [self.AGFG[k](wf[3][k], wf[2][k], wf[1][k], wf[0][k])
                 for k in range(self.K)]
        bands = self.band_weight(bands)
        return bands, f1, f2, f3, f4


class MidBandFuse(nn.Module):
    """Fuse the mid bands and inject them into a decoder stage (RQ3).

    RQ3 asks whether the edge-reconstruction module benefits from seeing the
    intermediate frequency bands rather than only the two extremes FEDER
    routes to it. This block fuses ``bands[1:-1]`` to a single tensor and
    merges it into a decoder feature map at that map's own resolution.

    With one mid band the fuse is a 1x1 projection; with none the module is
    never constructed.
    """

    def __init__(self, channels, n_mid):
        super().__init__()
        if n_mid < 1:
            raise ValueError("MidBandFuse needs at least one mid band")
        self.proj = nn.Conv2d(channels * n_mid, channels, 1)
        self.merge = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, feat, mids):
        m = self.proj(torch.cat(mids, dim=1)) if len(mids) > 1 else self.proj(mids[0])
        if m.shape[-2:] != feat.shape[-2:]:
            m = F.interpolate(m, size=feat.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.merge(torch.cat([m, feat], dim=1))
