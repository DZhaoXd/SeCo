import json
import os
import numpy as np
import pycocotools.mask as mutils
from tqdm import tqdm


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


def convert_2input(json_path, save_dir):
    with open(json_path, 'r') as f:
        ins_mask_list = json.load(f)

        img_dir_name_dict = {}
        loss_list = []
        score_list = []
        pred_idx_list = []
        for idx, ins_mask in tqdm(enumerate(ins_mask_list)):
            img_name = ins_mask['name']
            img_dir = img_name.split('_')[0] if '_' in img_name else ''
            loss_list.append(ins_mask['loss'])
            score_list.append(ins_mask['score'])
            pred_idx_list.append(ins_mask['pred_id'])
            img_dir_name = (img_dir, img_name)
            if img_dir_name not in img_dir_name_dict:
                img_dir_name_dict[img_dir_name] = [idx]
            else:
                img_dir_name_dict[img_dir_name].append(idx)
        ratio = 0.5
        loss_threshold = determine_threshold(loss_list, ratio)
        score_threshold = determine_threshold(score_list, 1 - ratio)
        loss_list = np.array(loss_list)
        score_list = np.array(score_list)
        pred_idx_list = np.array(pred_idx_list)

        local_loss_threshold = np.zeros_like(loss_list)
        class_id_set = np.unique(pred_idx_list)
        for class_id in tqdm(class_id_set):
            class_id_index = pred_idx_list == class_id
            class_loss_threshold = determine_threshold(loss_list[class_id_index], ratio)
            local_loss_threshold[class_id_index] = class_loss_threshold
        global_loss_threshold = np.mean(local_loss_threshold)

        local_score_threshold = np.zeros_like(score_list)
        class_id_set = np.unique(pred_idx_list)
        for class_id in tqdm(class_id_set):
            class_id_index = pred_idx_list == class_id
            class_score_threshold = determine_threshold(score_list[class_id_index], 1 - ratio)
            local_score_threshold[class_id_index] = class_score_threshold
        global_score_threshold = np.mean(local_score_threshold)

        for img_dir_name, idx_list in tqdm(img_dir_name_dict.items()):
            if len(idx_list) == 0: continue
            record_list = []
            for i, idx in enumerate(idx_list):
                record = {}
                segmeantation = ins_mask_list[idx]['segmentation']

                bbox = [int(_) for _ in mutils.toBbox(segmeantation)]
                area = int(mutils.area(segmeantation))
                category_id = ins_mask_list[idx]['pred_id']
                true_label = {'id': ins_mask_list[idx]['true_id'],
                              'purity': ins_mask_list[idx]['purity'],
                              'positive': ins_mask_list[idx]['positive']}

                record['loss'] = ins_mask_list[idx]['loss']
                record['score'] = ins_mask_list[idx]['score']

                record['segmentation'] = segmeantation
                record['bbox'] = bbox
                record['area'] = area
                record['id'] = i
                record['category_id'] = category_id
                record['true_label'] = true_label
                record['score_threshold'] = score_threshold
                record['loss_threshold'] = loss_threshold
                record['local_score_threshold'] = local_score_threshold[idx]
                record['local_loss_threshold'] = local_loss_threshold[idx]
                record['global_score_threshold'] = global_score_threshold
                record['global_loss_threshold'] = global_loss_threshold
                record_list.append(record)

            save_path = os.path.join(save_dir, 'train', *img_dir_name).replace('.png', '.json').replace('.jpg', '.json')
            # pass
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(record_list, f, indent=2)
        print('Convert Over')


if __name__ == '__main__':
    import sys

    json_path, save_dir = sys.argv[1], sys.argv[2]
    # use:
    # python convert_json_2input [/output/json/file/path] [/save/inputJsonFormatLike/path]
    # case:
    # json_path = r'/data/yrz/repos/BETA/checkpoints/bdd/val_src_base/val_DTST_bdd_sam_vit_h_iter_50000_20231209_183444.json'
    # save_dir = r'data/output_json/val_DTST_bdd_sam_vit_h_iter_50000_0.5_1'
    convert_2input(json_path, save_dir)
