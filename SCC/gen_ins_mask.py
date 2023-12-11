import os
from PIL import Image
import numpy as np
import cv2
import pycocotools.mask as mutils
import json
import matplotlib.pyplot as plt

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

cityspallete_16 = [
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

cityscapes_category_16 = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "light",
    "sign",
    "vegetation",
    "sky",
    "person",
    "rider",
    "car",
    "bus",
    "motocycle",
    "bicycle", ]


def transform_label(pred):
    synthia_to_city = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        10: 9,
        11: 10,
        12: 11,
        13: 12,
        15: 13,
        17: 14,
        18: 15,
    }
    label_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in synthia_to_city.items():
        label_copy[pred == k] = v
    return label_copy.copy()


def show_ndarray(data, putpalette=True):
    if data.max() != 255:
        data = 255 * data
    image = Image.fromarray(data.astype('uint8')).convert('P')
    if putpalette:
        image.putpalette(cityspallete)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def gen_ins_mask(result_dir, save_dir, info_path, root_path, transform_19to16='none'):
    assert transform_19to16 in ['none', 'only_image', 'only_label', 'all']
    img_cnt = 0.
    ins_cnt = 0.
    max_cnt = -np.inf
    min_cnt = np.inf
    positive_cnt = 0.
    purity_list = []

    ids = [i.strip().split(' ')[0] for i in open(info_path, 'r').readlines()]

    for id in ids:
        psd_label_path = os.path.join(result_dir, *id.split('/')[1:])
        psd_label = np.array(Image.open(os.path.join(result_dir, *id.split('/')[1:])))
        gt = Image.open(os.path.join(root_path, id.replace('leftImg8bit/train', 'gtFine/train').replace('leftImg8bit', 'gtFine_labelTrainIds')))
        gt = np.array(gt)
        gt = cv2.resize(gt, dsize=psd_label.shape[::-1], interpolation=cv2.INTER_NEAREST)

        if transform_19to16 == 'only_label':
            # print(np.unique(gt),np.unique(psd_label))
            gt = transform_label(gt).astype('uint8')
        elif transform_19to16 == 'only_image':
            psd_label = transform_label(psd_label).astype('uint8')
        elif transform_19to16 == 'all':
            gt = transform_label(gt).astype('uint8')
            psd_label = transform_label(psd_label).astype('uint8')
        else:
            None
        unique_label = np.unique(psd_label)
        # show_ndarray(psd_label)
        dts = []
        for label_id in unique_label:
            if label_id == 255: continue
            mask = psd_label == label_id
            mask = mask.astype(np.uint8)
            nc, label = cv2.connectedComponents(mask, connectivity=8)
            for c in range(nc):
                if c == 0: continue
                ann = np.asfortranarray((label == c).astype(np.uint8))
                true_label_count = np.bincount(gt[ann == 1])
                true_label_id = true_label_count.argmax()
                purity = round(true_label_count.max() / true_label_count.sum() * 100, 2)
                rle = mutils.encode(ann)
                bbox = [int(_) for _ in mutils.toBbox(rle)]
                area = int(mutils.area(rle))

                positive = true_label_id == label_id
                purity_list.append(purity)
                if positive:
                    positive_cnt += 1

                # if area>500:
                #     show_ndarray(ann, putpalette=False)
                # if area < 100:
                #     continue

                dts.append({
                    "segmentation": {
                        "size": [int(_) for _ in rle["size"]],
                        "counts": rle["counts"].decode()},
                    "bbox": [int(_) for _ in bbox], "area": int(area), "category_id": int(label_id),
                    "id": len(dts),
                    "true_label": {'id': int(true_label_id), 'purity': purity, 'positive': int(positive), }
                })
        print("filename:{}\t ins_num:{}\t".format(psd_label_path, len(dts)))
        save_json_path = psd_label_path.replace(result_dir, save_dir).replace(".png", ".json")
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        with open(save_json_path, 'w') as f:
            json.dump(dts, f, indent=2)
        img_cnt += 1
        ins_cnt += len(dts)
        if len(dts) > max_cnt:
            max_cnt = len(dts)
        if len(dts) < min_cnt:
            min_cnt = len(dts)

    print('get data instances mask over\n'
          'img cnt : {}\tins cnt : {}\n'
          'positive_cnt : {}\tpositive_ratio : {}%\n'
          'mean ins : {}\tmax ins : {}\tmin ins : {}\n'
          'purity: \tmax-{}%\tmin-{}%\tmean-{}%'.format(img_cnt, ins_cnt, positive_cnt, round(positive_cnt / ins_cnt * 100, 2),
                                                        round(ins_cnt / img_cnt, 2), max_cnt, min_cnt,
                                                        np.max(purity_list), np.min(purity_list), np.round(np.mean(purity_list), 2))
          )


if __name__ == '__main__':
    import sys

    input_dir, save_dir, root_dir, info_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    transform_mode = 'none'  # you can choose in ['none', 'only_image', 'only_label', 'all']
    # case:
    # input_dir = '/data/yrz/repos/sam/results/stuff_and_thing_pixel_50_vit_h_pix_top_50'
    # save_dir = '/data/yrz/repos/sam/input_json/stuff_and_thing_pixel_50_vit_h_pix_top_50'
    # root_dir = '/data/yrz/repos/sam/data'
    # info_path = '/data/yrz/repos/sam/splits/cityscapes/pix_top25_top_50_image/all_data.txt'

    # gen_ins_mask(result_dir=result_dir, save_dir=save_dir, info_path=info_path, root_path=root_path)
    gen_ins_mask(result_dir=input_dir, save_dir=save_dir, info_path=info_path, root_path=root_dir, transform_19to16=transform_mode)
