"""Write prediction maps for one checkpoint across the COD test sets.

Changes from the released script, all of which blocked use here:
  * the GPU was hardcoded to device '1';
  * ``torch.load(...).items()`` assumed a bare state_dict, so it raised on the
    checkpoints Train.py now writes (model + optimizer + epoch);
  * the architecture was fixed, so a K-band run could not be
    tested at all. The flags are read from the config.json the training run
    left beside its checkpoints, and can still be overridden.
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.Network import Network
from lib.config import (add_model_args, model_kwargs, load_weights,
                        load_config_for_checkpoint, describe)
from utils.data_val import test_dataset

ALL_SETS = ['CAMO', 'COD10K', 'CHAMELEON', 'NC4K']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--testsize', type=int, default=384,
                        help='must match the training resolution')
    parser.add_argument('--pth_path', type=str, required=True)
    parser.add_argument('--test_dataset_path', type=str,
                        default='./data/USEDATA/TestDataset')
    parser.add_argument('--save_root', type=str, default='./res',
                        help='predictions go to <save_root>/<run name>/<dataset>/')
    parser.add_argument('--datasets', nargs='+', default=ALL_SETS, choices=ALL_SETS)
    parser.add_argument('--gpu_id', type=str, default='0')
    add_model_args(parser)
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA device visible. Check --gpu_id (you passed {!r}).'
                         .format(opt.gpu_id))

    # Prefer the architecture the training run recorded; only fall back to the
    # command line when there is no config.json to read.
    cfg = load_config_for_checkpoint(opt.pth_path)
    if cfg:
        print('using architecture from {}'.format(
            os.path.join(os.path.dirname(opt.pth_path), 'config.json')))
        for k, v in cfg.items():
            setattr(opt, k, v)
    else:
        print('no config.json beside the checkpoint; using the command-line flags')
    print('CONFIG: ' + describe(opt))

    kw = model_kwargs(opt)
    channels = kw.pop('channels', 96)
    model = Network(channels, pretrained=False, **kw)
    load_weights(model, opt.pth_path)
    model.cuda().eval()

    run_name = os.path.basename(os.path.dirname(opt.pth_path.rstrip('/'))) or 'run'
    ckpt_tag = os.path.splitext(os.path.basename(opt.pth_path))[0]

    for name in opt.datasets:
        data_path = os.path.join(opt.test_dataset_path, name)
        if not os.path.isdir(data_path):
            print('skipping {}: {} not found'.format(name, data_path))
            continue
        save_path = os.path.join(opt.save_root, '{}_{}'.format(run_name, ckpt_tag), name)
        os.makedirs(save_path, exist_ok=True)

        loader = test_dataset(os.path.join(data_path, 'Imgs') + '/',
                              os.path.join(data_path, 'GT') + '/', opt.testsize)
        for i in range(loader.size):
            image, gt, img_name, _ = loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            with torch.no_grad():
                result = model(image.cuda())
            res = F.interpolate(result[4], size=gt.shape, mode='bilinear',
                                align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # Always .png: a .jpg prediction is re-quantised by the encoder and
            # scores measurably worse than the map the model produced.
            stem = os.path.splitext(img_name)[0]
            cv2.imwrite(os.path.join(save_path, stem + '.png'), res * 255)
            if (i + 1) % 250 == 0:
                print('  {} {}/{}'.format(name, i + 1, loader.size))
        print('{}: wrote {} maps to {}'.format(name, loader.size, save_path))


if __name__ == '__main__':
    main()
