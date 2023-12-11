import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from utils.util import AverageMeter, accuracy, TrackMeter
from utils.util import set_seed

from utils.config import Config, ConfigDict, DictAction
from losses import build_loss
from builder import build_optimizer
from models.build import build_model
from utils.util import format_time
from builder import build_logger
from datasets import build_dataset
from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--load', type=str, help='Load init weights for fine-tune (default: None)')
    parser.add_argument('--cfgname', help='specify log_file; for debug use')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override the config; e.g., --cfg-options port=10001 k1=a,b k2="[a,b]"'
                             'Note that the quotation marks are necessary and that no white space is allowed.')
    parser.add_argument('--local_rank', type=int, help='  ')
    args = parser.parse_args()
    return args


def get_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        dirname = os.path.dirname(args.config).replace('configs', 'checkpoints', 1)
        filename = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = os.path.join(dirname, filename)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # cfgname
    if args.cfgname is not None:
        cfg.cfgname = args.cfgname
    else:
        cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
    assert cfg.cfgname is not None

    # seed
    if args.seed != 0:
        cfg.seed = args.seed
    elif not hasattr(cfg, 'seed'):
        cfg.seed = 42
    set_seed(cfg.seed)

    # resume or load init weights
    if args.resume:
        cfg.resume = args.resume
    if args.load:
        cfg.load = args.load
    assert not (cfg.resume and cfg.load)

    return cfg


def val(val_loader, model, criterion, it,  writer, output_json=None):
    """validation"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    true_top1 = AverageMeter()
    true_top5 = AverageMeter()

    result_dump = []

    time1 = time.time()
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Val:\t', unit='batch', ncols=100) as pbar:
            for idx, (images, ins_mask, labels, true_label, segmentation, name) in enumerate(val_loader):
                # if idx == 10: break
                images = images.cuda()
                labels = labels.cuda()
                ins_mask = ins_mask.cuda()

                true_label_id = true_label['id'].cuda()

                bsz = labels.shape[0]

                # forward
                output = model(images, ins_mask)
                loss_all = criterion(output, labels)

                loss = loss_all.mean()

                # update metric
                losses.update(loss.item(), bsz)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                true_acc1, true_acc5 = accuracy(output, true_label_id, topk=(1, 5))
                top1.update(acc1[0], bsz)
                top5.update(acc5[0], bsz)
                true_top1.update(true_acc1[0], bsz)
                true_top5.update(true_acc5[0], bsz)
                pbar.set_postfix(**{'Acc@1': top1.avg.item(),
                                    # 'Acc@5': top5.avg.item(),
                                    'TAcc@1': true_top1.avg.item(),
                                    # 'TAcc@5': true_top5.avg.item(),
                                    'Loss': losses.avg,
                                    })
                pbar.update()
                scores, pred_ids = torch.softmax(output, 1).max(1)

                # confusion matrix
                positive = labels == true_label_id
                negative = labels != true_label_id
                true_pred = pred_ids == true_label_id
                false_pred = pred_ids != true_label_id
                ps_pred = pred_ids == labels

                TP_ids = true_pred * positive
                TN_ids = true_pred * negative
                FP_ids = false_pred * positive
                FN_ids = false_pred * negative
                segmentation_size = [list(map(lambda x: x.item(), size)) for size in zip(*segmentation['size'])]
                segmentation['size']=segmentation_size
                for idx in range(bsz):
                    record = {'name': name[idx],
                              'score': scores[idx].item(),
                              'pred_id': pred_ids[idx].item(),
                              'psl_id': labels[idx].item(),
                              'true_id': true_label_id[idx].item(),
                              'purity': true_label['purity'][idx].item(),
                              'positive': true_label['positive'][idx].item(),
                              'TP': int(TP_ids[idx].item()),
                              'TN': int(TN_ids[idx].item()),
                              'FP': int(FP_ids[idx].item()),
                              'FN': int(FN_ids[idx].item()),
                              'T': int(true_pred[idx].item()),
                              'PST': int(ps_pred[idx].item()),
                              'loss': loss_all[idx].item(),
                              'segmentation': {k: v[idx] for k, v in segmentation.items()},
                              }
                    result_dump.append(record)

    writer.add_scalar(f'Loss/src_val', losses.avg, it)
    writer.add_scalar(f'Acc/src_val', top1.avg, it)

    if output_json:
        return true_top1.avg, result_dump

    return true_top1.avg


def main():
    # args & cfg
    args = parse_args()
    cfg = get_cfg(args)  # may modify cfg according to args
    cudnn.benchmark = True

    # write cfg
    print('work dir: {}'.format(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_name = '_'.join(cfg.load.split('/')[-2:])[:-4]
    log_file = os.path.join(cfg.work_dir, f'val_{log_name}_{timestamp}.cfg')
    with open(log_file, 'w') as f:
        f.write(cfg.pretty_text)

    # logger
    writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, f'tensorboard'))

    '''
    # -----------------------------------------
    # build dataset/dataloader
    # -----------------------------------------
    '''
    loader_dict = {}
    val_set = build_dataset(cfg.data.val)
    loader_dict['src_val'] = DataLoader(val_set, batch_size=cfg.batch_size,
                                        shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    print(f'==> DataLoader built.')

    '''
    # -----------------------------------------
    # build model & optimizer
    # -----------------------------------------
    '''
    model = build_model(cfg.model).cuda()
    val_criterion = build_loss(cfg.loss.val).cuda()
    model_state = torch.load(cfg.load)['model_state']
    model_state_new = {}
    for name, data in model_state.items():
        model_state_new[name[7:]] = data
    model.load_state_dict(model_state_new)
    print('==> Model built.')

    '''
    # -----------------------------------------
    # Start source training
    # -----------------------------------------
    '''
    print("==> Start training...")

    it = 0
    val_acc, result_dump = val(loader_dict['src_val'], model, val_criterion, it, writer, output_json=True)
    json_file = log_file.replace('.cfg', '.json')
    with open(json_file, 'w') as f:
        json.dump(result_dump, f)


if __name__ == '__main__':
    main()
