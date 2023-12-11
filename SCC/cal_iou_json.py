import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image
import json
import pycocotools.mask as mutils

cityspallete = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]


def Cal_iou(mask_a, mask_b):
    i = np.sum(mask_a * mask_b)
    u = np.sum(mask_a) + np.sum(mask_b) - i
    return i / (u + 0.1)


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_iou(root_path, id_path, test_data_dir=None,test_ins_dir=None, test_ins_json_dir=None, ignore_test_255=False, filter_mode=None):
    assert test_data_dir is not None or test_ins_json_dir is not None or test_ins_dir is not None
    assert filter_mode in (None, 'loss', 'score', 'both')
    Class_num = 19
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    with open(id_path, 'r') as f:
        list_ids = f.read().splitlines()
    ins_cnt = 0
    ori_ins_cnt = 0
    for id in tqdm(list_ids):
        # psd_label_ori = Image.open(os.path.join(root_path, id.split(' ')[1].replace('pix_top_25/train','{}/train'.format(test_data))   ))
        gt = Image.open(os.path.join(root_path, id.split(' ')[0].replace('leftImg8bit/train', 'gtFine/train').replace('leftImg8bit', 'gtFine_labelTrainIds')))
        size = gt.size
        if test_data_dir is not None:
            psd_label_ori = Image.open(os.path.join('{}/train'.format(test_data_dir), *id.split(' ')[1].split('/')[-2:]))
            psd_label_ori = psd_label_ori.resize(gt.size)
            psd_label = np.array(psd_label_ori)
        else:
            psd_label = np.ones(gt.size[::-1]) * 255

        if test_ins_dir is not None:
            psd_label_ins = Image.open(os.path.join('{}/train'.format(test_ins_dir), *id.split(' ')[1].split('/')[-2:]))
            psd_label_ins = psd_label_ins.resize(gt.size)
            psd_label_ins = np.array(psd_label_ins)
            psd_label[psd_label_ins != 255] = psd_label_ins[psd_label_ins != 255]

        if test_ins_json_dir is not None:
            with open(os.path.join('{}/train'.format(test_ins_json_dir), *id.split(' ')[1].split('/')[-2:]).replace('.png', '.json'), 'r') as f:
                ins_mask_list = json.load(f)
            for ins_mask in ins_mask_list:
                ori_ins_cnt += 1
                """过滤规则"""
                if filter_mode == 'loss':
                    if ins_mask['loss'] > ins_mask['loss_threshold']:
                        continue
                elif filter_mode == 'score':
                    if ins_mask['score'] < ins_mask['score_threshold']:
                        continue
                elif filter_mode == 'both':
                    if ins_mask['loss'] > ins_mask['loss_threshold'] or ins_mask['score'] < ins_mask['score_threshold']:
                        continue

                ins_cnt += 1
                psd_label[cv2.resize(mutils.decode(ins_mask['segmentation']), size, interpolation=cv2.INTER_NEAREST) == 1] = ins_mask['category_id']

        gt = np.array(gt)
        if ignore_test_255:
            gt[psd_label == 255] = 255

        intersection, union, target = intersectionAndUnion(psd_label, gt, K=Class_num, ignore_index=255)
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    axis_name = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    print('ori_ins_cnt:{}\tins_cnt:{}'.format(ori_ins_cnt, ins_cnt))
    print('SAM result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(Class_num):
        print('{} {} iou/accuracy: {:.4f}/{:.4f}.'.format(i, axis_name[i], iou_class[i], accuracy_class[i]))


if __name__ == '__main__':

    # test_data = r'/data/yrz/repos/sam/data/top_20_pre43.3'
    test_data = None

    test_ins_dir = '/data/yrz/repos/sam/data/seco_vit_h_seco'
    # test_ins_dir = None

    # test_ins_json_dir = r'/data/yrz/repos/sam/input_json/sam_v6_vit_h_pix_top_25'
    test_ins_json_dir = None

    ignore_test_255 = True
    id_path = './splits/cityscapes/pix_top25_top_50_image/all_data.txt'
    root_path = 'data'
    filter_mode = None
    # filter_mode = 'both'

    name = ''
    if test_data is not None:
        name = ' + '.join([name, '/'.join(test_data.split('/')[-2:])])
    if test_ins_dir is not None:
        name = ' + '.join([name, '/'.join(test_ins_dir.split('/')[-2:])])
    if test_ins_json_dir is not None:
        name = ' + '.join([name, '/'.join(test_ins_json_dir.split('/')[-2:])])
    if ignore_test_255:
        name = ' + '.join([name, 'ignore_test_255'])
    if filter_mode is not None:
        name = ' + '.join([name, filter_mode])
    print('\nname:{}\n'
          'tset_data:{}\n'
          'test_ins_json:{}\n'
          'ignore_test_255:{}\n'.format(name, test_data, test_ins_json_dir, ignore_test_255))
    """
    Priority： test_ins_json_dir > test_ins_dir > test_data
    """
    cal_iou(root_path, id_path,
            test_data_dir=test_data,
            test_ins_dir=test_ins_dir,
            test_ins_json_dir=test_ins_json_dir,
            ignore_test_255=ignore_test_255, filter_mode=filter_mode)
