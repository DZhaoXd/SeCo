import os
from PIL import Image
import numpy as np
import cv2
import time
import pycocotools.mask as mutils
import json
from tqdm import tqdm

import matplotlib.pyplot as plt


def determine_threshold(loss_list, target_ratio):
    """
    # 示例使用
    loss_list = [0.1, 0.5, 0.2, 0.3, 0.7, 0.4, 0.6, 0.8, 0.9, 0.5]
    target_ratio = 0.8

    threshold = determine_threshold(loss_list, target_ratio)
    print(threshold)
    """
    sorted_losses = sorted(loss_list)  # 将损失函数列表按照从小到大排序
    total_samples = len(sorted_losses)  # 总样本数量
    threshold_index = min(int(total_samples * target_ratio), total_samples - 1)  # 根据目标比例确定阈值位置
    threshold = sorted_losses[threshold_index]  # 获取阈值

    return threshold


def get_color_pallete(npimg, dataset='none'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    if dataset == 'city':
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
            255, 255, 255
        ]
        zero_pad = 256 * 3 - len(cityspallete)
        for i in range(zero_pad):
            cityspallete.append(255)
        out_img.putpalette(cityspallete)
    elif dataset == 'city_16':
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
            # 152, 251, 152,
            0, 130, 180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            # 0, 0, 70,
            0, 60, 100,
            # 0, 80, 100,
            0, 0, 230,
            119, 11, 32,
        ]
        zero_pad = 256 * 3 - len(cityspallete)
        for i in range(zero_pad):
            cityspallete.append(255)
        out_img.putpalette(cityspallete)
    elif dataset == 'none':
        None
    else:
        raise ValueError
    return out_img


def vis_json(root, save_dir=None, vis_trues=False, filter_mode=None, dataset_name='city'):
    assert filter_mode in (None, 'loss', 'score', 'both', 'local_loss', 'local_score', 'local_both', 'both_loss', 'both_score', 'both_both')
    if save_dir is None:
        save_dir = root + '_vis'
    if filter_mode is not None:
        save_dir = save_dir + f'_{filter_mode}'
    for root_path, dirs, files in os.walk(root):
        for file in tqdm(sorted(files)):
            file_name = os.path.splitext(file)[0]
            with open(os.path.join(root_path, file), 'r') as f:
                ins_mask_list = json.load(f)
            seg_mask = np.ones(ins_mask_list[0]["segmentation"]['size']) * 255

            for ins_mask in ins_mask_list:
                """过滤规则"""
                if filter_mode == 'loss':
                    if ins_mask['loss'] > ins_mask['loss_threshold']: continue
                elif filter_mode == 'score':
                    if ins_mask['score'] < ins_mask['score_threshold']: continue
                elif filter_mode == 'both':
                    if ins_mask['loss'] > ins_mask['loss_threshold'] or ins_mask['score'] < ins_mask['score_threshold']: continue
                elif filter_mode == 'local_loss':
                    if ins_mask['loss'] > ins_mask['local_loss_threshold']: continue
                elif filter_mode == 'local_score':
                    if ins_mask['score'] < ins_mask['local_score_threshold']: continue
                elif filter_mode == 'local_both':
                    if ins_mask['loss'] > ins_mask['local_loss_threshold'] or ins_mask['score'] < ins_mask['local_score_threshold']: continue
                elif filter_mode == 'both_loss':
                    if (ins_mask['loss'] > ins_mask['local_loss_threshold']) * (ins_mask['loss'] > ins_mask['global_loss_threshold']): continue
                elif filter_mode == 'both_score':
                    if (ins_mask['score'] < ins_mask['local_score_threshold']) * (ins_mask['score'] < ins_mask['global_score_threshold']): continue
                elif filter_mode == 'both_both':
                    if ((ins_mask['loss'] > ins_mask['local_loss_threshold']) *
                        (ins_mask['loss'] > ins_mask['global_loss_threshold'])) or (
                            (ins_mask['score'] < ins_mask['local_score_threshold']) *
                            (ins_mask['score'] < ins_mask['global_score_threshold'])): continue
                seg_mask[mutils.decode(ins_mask['segmentation']) == 1] = ins_mask['category_id']
                # print('label classes:',np.unique(seg_mask))
            mask = get_color_pallete(seg_mask, dataset_name)
            save_path = os.path.join(root_path.replace(root, save_dir), file_name + ".png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            mask.save(save_path)
            if not vis_trues:
                continue

            seg_mask = np.ones(ins_mask_list[0]["segmentation"]['size']) * 255

            for ins_mask in ins_mask_list:
                """过滤规则"""
                # if filter_mode == 'loss':
                #     if ins_mask['loss'] > ins_mask['loss_threshold']: continue
                # elif filter_mode == 'score':
                #     if ins_mask['score'] < ins_mask['score_threshold']: continue
                # elif filter_mode == 'both':
                #     if ins_mask['loss'] > ins_mask['loss_threshold'] or ins_mask['score'] < ins_mask['score_threshold']: continue
                # elif filter_mode == 'local loss':
                #     if ins_mask['loss'] > ins_mask['local_loss_threshold']: continue
                # elif filter_mode == 'local score':
                #     if ins_mask['score'] < ins_mask['local_score_threshold']: continue
                # elif filter_mode == 'local both':
                #     if ins_mask['loss'] > ins_mask['local_loss_threshold'] or ins_mask['score'] < ins_mask['local_score_threshold']: continue
                # elif filter_mode == 'both loss':
                #     if (ins_mask['loss'] > ins_mask['local_loss_threshold']) * (ins_mask['loss'] > ins_mask['global_loss_threshold']): continue
                # elif filter_mode == 'both score':
                #     if (ins_mask['score'] < ins_mask['local_score_threshold']) * (ins_mask['score'] < ins_mask['global_score_threshold']): continue
                # elif filter_mode == 'both both':
                #     if ((ins_mask['loss'] > ins_mask['local_loss_threshold']) *
                #         (ins_mask['loss'] > ins_mask['global_loss_threshold'])) or (
                #             (ins_mask['score'] < ins_mask['local_score_threshold']) *
                #             (ins_mask['score'] < ins_mask['global_score_threshold'])): continue
                seg_mask[mutils.decode(ins_mask['segmentation']) == 1] = ins_mask['true_label']['id']
                # print('label classes:',np.unique(seg_mask))
            mask = get_color_pallete(seg_mask, dataset_name)
            save_path = os.path.join(root_path.replace(root, save_dir + '_GT'), file_name + ".png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            mask.save(save_path)


if __name__ == '__main__':
    # case
    """output show"""
    # """DTST_synthia_sam_vit_h_16"""
    # data_path = r'/data/yrz/repos/sam/output_json/val_DTST_synthia_sam_vit_h_16_iter_10000'
    # save_dir = r'/data/yrz/repos/BETA/data/output_json_show'
    # filter_mode = 'both_loss'
    # dataset_name = 'city_16'

    # data_path = r'/data/yrz/repos/sam/output_json/val_DTST_synthia_sam_vit_h_16_iter_10000_0.3'
    # save_dir = r'/data/yrz/repos/BETA/data/output_json_show'
    # filter_mode = 'both_loss'
    # dataset_name = 'city_16'

    """val_DTST_bdd_sam_vit_h"""
    # data_path = r'/data/yrz/repos/sam/output_json/val_DTST_bdd_sam_vit_h_iter_50000_0.5'
    # save_dir = r'/data/yrz/repos/BETA/data/output_json_show'
    # filter_mode = 'both_loss'
    # dataset_name = 'city'

    """input show"""
    # """DTST_synthia_sam_vit_h_16"""
    # data_path = r'/data/yrz/repos/sam/input_json/DTST_synthia_sam_vit_h_16'
    # save_dir = r'/data/yrz/repos/BETA/data/input_json_show'
    # filter_mode = None
    # dataset_name = 'city_16'

    # """bdd/DTST_bdd_sam_vit_h"""
    # data_path = r'/data/yrz/repos/sam/input_json/bdd/DTST_bdd_sam_vit_h'
    # save_dir = r'/data/yrz/repos/BETA/data/input_json_show/bdd'
    # filter_mode = None
    # dataset_name = 'city'

    import sys

    # data_path, save_dir, filter_mode, dataset_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    data_path, save_dir, filter_mode = sys.argv[1], sys.argv[2], sys.argv[3]
    save_dir = os.path.join(save_dir, os.path.basename(data_path))
    vis_json(data_path, save_dir, vis_trues=True, filter_mode=filter_mode, dataset_name='city')
