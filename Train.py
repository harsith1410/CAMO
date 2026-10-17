import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.Network import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, get_coef,cal_ual
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from torch import optim


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def save_checkpoint(path, model, optimizer, epoch, scaler=None):
    """Full checkpoint: weights, optimizer, epoch and best-so-far.

    The released code saved model weights only, so --load restarted the
    optimizer and the LR schedule from scratch.
    """
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'best_mae': globals().get('best_mae', 1.0),
        'best_epoch': globals().get('best_epoch', 0),
    }, path)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """Returns the epoch to resume from. Accepts old weights-only files too."""
    global best_mae, best_epoch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'model' in ck:
        model.load_state_dict(ck['model'])
        if optimizer is not None and ck.get('optimizer') is not None:
            optimizer.load_state_dict(ck['optimizer'])
        if scaler is not None and ck.get('scaler') is not None:
            scaler.load_state_dict(ck['scaler'])
        best_mae = ck.get('best_mae', 1.0)
        best_epoch = ck.get('best_epoch', 0)
        print('resumed from {} at epoch {}'.format(path, ck.get('epoch', 0)))
        return int(ck.get('epoch', 0)) + 1
    model.load_state_dict(ck)
    print('loaded weights-only checkpoint from', path)
    return 1


def train(train_loader, model, optimizer, epoch, save_path, writer, scaler=None):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    accum = max(1, opt.accum_steps)
    optimizer.zero_grad(set_to_none=True)
    try:
        for i, (images, gts, edges) in enumerate(train_loader, start=1):
            images = images.cuda(device=device_ids[0], non_blocking=True)
            gts = gts.cuda(device=device_ids[0], non_blocking=True)
            edges = edges.cuda(device=device_ids[0], non_blocking=True)

            with torch.amp.autocast('cuda', enabled=opt.amp):
                preds = model(images)

                ual_coef = get_coef(iter_percentage=i/total_step, method='cos')
                ual_loss = cal_ual(seg_logits=preds[4], seg_gts=gts)
                ual_loss *= ual_coef

                loss_init = structure_loss(preds[0], gts)*0.0625 + structure_loss(preds[1], gts)*0.125 + structure_loss(preds[2], gts)*0.25 + \
                            structure_loss(preds[3], gts)*0.5
                loss_final = structure_loss(preds[4], gts)

                loss_edge = dice_loss(preds[6], edges)*0.125 + dice_loss(preds[7], edges)*0.25 + \
                            dice_loss(preds[8], edges)*0.5

                loss = loss_init + loss_final + loss_edge + 2*ual_loss

            # Gradient accumulation: the paper's batch 36 was spread over two
            # 24GB cards. Use a smaller micro-batch with accum_steps so the
            # effective batch stays 36 and the tuned 1e-4 LR remains valid.
            scaler.scale(loss / accum).backward()
            if i % accum == 0 or i == total_step:
                scaler.unscale_(optimizer)
                clip_gradient(optimizer, opt.clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f} Loss3:{:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data, loss_edge.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f} Loss3:{:0.4f}'.
                        format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data, loss_edge.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_init': loss_init.data, 'Loss_final': loss_final.data, 'Loss_edge': loss_edge.data, 'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)
                grid_image = make_grid(edges[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Edge', grid_image, step)

                # TensorboardX-Outputs
                res = preds[0][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_init', torch.tensor(res), step, dataformats='HW')

                res = preds[4][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')

                res = preds[8][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_edge', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        # Checkpoint every save_freq epochs, and always keep 'last'. The
        # original saved only every 80 epochs and stored weights alone, so an
        # interrupted run could not resume its optimizer or schedule.
        save_checkpoint(save_path + 'last.pth', model, optimizer, epoch, scaler)
        if epoch % opt.save_freq == 0:
            save_checkpoint(save_path + 'Net_epoch_{}.pth'.format(epoch), model, optimizer, epoch, scaler)
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_checkpoint(save_path + 'interrupt.pth', model, optimizer, epoch, scaler)
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer, optimizer=None):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        mae_sum_edge = 0
        for i in range(test_loader.size):
            image, gt,  name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda(device=device_ids[0])

            result = model(image)

            res = F.interpolate(result[4], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
            best_epoch = 1
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                save_checkpoint(save_path + 'Net_epoch_best.pth', model, optimizer, epoch)
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=36, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--amp', action='store_true',
                        help='mixed precision. Leave OFF for the baseline reproduction so the '
                             'protocol matches the paper; turn it on once your numbers match.')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='gradient accumulation. batchsize * accum_steps should equal 36 '
                             'so the tuned 1e-4 learning rate stays valid.')
    parser.add_argument('--val_freq', type=int, default=5,
                        help='validate every N epochs. The original validated every epoch at '
                             'batch size 1 over 2,026 images, a large share of wall clock.')
    parser.add_argument('--save_freq', type=int, default=10, help='checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume from a full checkpoint (model + optimizer + epoch)')
    parser.add_argument('--train_root', type=str, default='./data/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./data/TestDataset/COD10K/',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,default='./snapshot/model_new/',help='the path to save model and log')
    opt = parser.parse_args()



    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA device visible. Check --gpu_id (you passed {!r}).'.format(opt.gpu_id))

    # Derive device_ids from what is actually visible. The original hardcoded
    # [0, 1] while --gpu_id defaulted to '4,5', so it crashed on one GPU.
    device_ids = list(range(torch.cuda.device_count()))
    print('USE GPU {} -> {} visible device(s)'.format(opt.gpu_id, len(device_ids)))

    model = Network(channels=96)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=opt.amp)

    start_epoch = 1
    if opt.resume is not None:
        start_epoch = load_checkpoint(opt.resume, model, optimizer, scaler)
    elif opt.load is not None:
        load_checkpoint(opt.load, model)
        print('load model from ', opt.load)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              edge_root=opt.train_root + 'Edge/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)


    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0
    
    # learning rate schedule
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    print("Start train...")
    # NOTE: adjust_lr (step decay) and cosine_schedule both write the LR each
    # epoch, and adjust_lr overwrites the cosine value. This is the released
    # behaviour and is left untouched so the baseline reproduces; revisit it
    # only after you have a baseline number to compare against.
    for epoch in range(start_epoch, opt.epoch + 1):

        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_last_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_last_lr()[0]))

        train(train_loader, model, optimizer, epoch, save_path, writer, scaler)
        if epoch % opt.val_freq == 0 or epoch == opt.epoch:
            val(val_loader, model, epoch, save_path, writer, optimizer)

