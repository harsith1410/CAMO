from lib.Modules import GCM3, GPM, REM11, BasicConv2d
from lib.ModulesK import GCM3K, MidBandFuse, min_input_size
import timm
import torch.nn as nn
import torch
import torch.nn.functional as F

'''
backbone: resnet50
'''


class Network(nn.Module):
    """FEDER, with the RQ1/RQ3 extension points behind config flags.

    Defaults reproduce the released model exactly: ``bands=2`` with
    ``learnable_wavelet=False`` selects the original fixed-Haar ``GCM3``, so
    checkpoints trained before the extension still load.

    Args:
        channels:          decoder width.
        bands:             K for the learnable decomposition (RQ1). K=2 with
                           ``learnable_wavelet=True`` is the FEDER split with
                           trainable filters; K>2 adds mid bands.
        taps:              lifting filter length (odd).
        hf_mode:           'hh' uses HH as the finest band, exactly as FEDER
                           does. 'learned' fuses LH/HL/HH, which the released
                           forward pass computes and then discards.
        learnable_wavelet: False keeps the filters frozen at their Haar /
                           B-spline initialisation, giving a fixed-filter
                           K-band control.
        edge_feed:         RQ3. '2band' feeds the edge module only the two
                           extreme bands, as FEDER does. 'kband' additionally
                           injects the fused mid bands into the f2 and f3
                           decoder stages.
        ablate_mid:        drop the mid bands entirely while keeping K's
                           parameter count and band-wise aggregation. Isolates
                           whether the mid band is doing the work.
        gf_radius:         guided-filter radius. 2 is the released value and
                           forces inputs of at least 384; lowering it admits
                           smaller inputs but changes the architecture, so keep
                           it fixed across any one comparison. Only honoured on
                           the K-band path.
        pretrained:        ImageNet initialisation for the encoder. The
                           released code shipped this as False, which cannot
                           reproduce the paper.
    """

    # resnet based encoder decoder
    def __init__(self, channels=96, bands=2, taps=3, hf_mode='hh',
                 learnable_wavelet=False, edge_feed='2band', ablate_mid=False,
                 band_weight='none',
                 gf_radius=2, pretrained=True):
        super(Network, self).__init__()
        if edge_feed not in ('2band', 'kband'):
            raise ValueError("edge_feed must be '2band' or 'kband', got {!r}".format(edge_feed))
        self.bands = bands
        self.gf_radius = gf_radius
        self.min_size = min_input_size(gf_radius)
        self.edge_feed = edge_feed
        self.ablate_mid = ablate_mid
        self.use_kband = bool(learnable_wavelet or bands != 2)

        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=pretrained, in_chans=3, features_only=True)
        if self.use_kband:
            self.GCM3 = GCM3K(256, channels, bands=bands, taps=taps,
                              band_weight=band_weight,
                              hf_mode=hf_mode, learnable=learnable_wavelet,
                              gf_radius=gf_radius)
        else:
            self.GCM3 = GCM3(256, channels)
            self.min_size = min_input_size(2)
        self.GPM = GPM()
        self.REM11 = REM11(channels, channels)

        self.LL_down = nn.Sequential(
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1),
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1)
        )
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        self.one_conv_f4_ll = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1)
        self.one_conv_f1_hh = nn.Conv2d(in_channels=channels + channels // 4, out_channels=channels, kernel_size=1)

        # RQ3: mid-band injection into the edge module's f2/f3 inputs.
        n_mid = max(0, bands - 2)
        if self.use_kband and edge_feed == 'kband' and n_mid > 0 and not ablate_mid:
            self.mid_f2 = MidBandFuse(channels, n_mid)
            self.mid_f3 = MidBandFuse(channels, n_mid)
        else:
            self.mid_f2 = self.mid_f3 = None

    def forward(self, x, return_feats=False):
        image = x
        h, w = x.shape[-2:]
        if min(h, w) < self.min_size or h % 64 or w % 64:
            raise ValueError(
                "input {}x{} is not usable: this architecture needs a multiple "
                "of 64 of at least {} px (guided-filter radius r={}). 384, 448 "
                "and 512 work; 256 and 320 do not, in the released model as "
                "well as this one. Reduce epochs rather than resolution, or "
                "lower gf_radius and keep it fixed across the comparison."
                .format(h, w, self.min_size, self.gf_radius))
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats

        if self.use_kband:
            bands, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)
            high, low = bands[0], bands[-1]
            mids = [] if self.ablate_mid else bands[1:-1]
        else:
            LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)
            high, low, mids = HH, LL, []

        HH_up = self.dePixelShuffle(high)
        f1_HH = torch.cat([HH_up, f1], dim=1)
        f1_HH = self.one_conv_f1_hh(f1_HH)

        LL_down = self.LL_down(low)
        f4_LL = torch.cat([LL_down, f4], dim=1)
        f4_LL = self.one_conv_f4_ll(f4_LL)

        # RQ3: give the edge module the mid bands as well as the extremes.
        if self.mid_f2 is not None and mids:
            f2 = self.mid_f2(f2, mids)
            f3 = self.mid_f3(f3, mids)

        prior_cam = self.GPM(x4)
        pred_0 = F.interpolate(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        out = self.REM11([f1_HH, f2, f3, f4_LL], prior_cam, image)
        f4, f3_o, f2_o, f1_o, bound_f4, bound_f3, bound_f2, bound_f1 = out
        preds = (pred_0, f4, f3_o, f2_o, f1_o, bound_f4, bound_f3, bound_f2, bound_f1)
        if return_feats:
            # f1_HH is the finest decoder map (stride 4) and is what the
            # auxiliary heads sample patches from.
            return preds, {'f1': f1_HH, 'f2': f2, 'f3': f3, 'f4': f4_LL,
                           'bands': bands if self.use_kband else [high, low]}
        return preds


if __name__ == '__main__':
    image = torch.rand(2, 3, 384, 384).cuda()
    model = Network(96).cuda()
    pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = model(image)
    print(pred_0.shape)
    print(f4.shape)
    print(f3.shape)
    print(f2.shape)
    print(f1.shape)
    print(bound_f4.shape)
    print(bound_f3.shape)
    print(bound_f2.shape)
    print(bound_f1.shape)
