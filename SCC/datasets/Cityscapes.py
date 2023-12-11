import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pycocotools.mask as mutils
import json
import cv2
from tqdm import tqdm


class BaseCityscapes(Dataset):
    def __init__(self, root='./data/Cityscapes', root_mask='./results/ins_mask', info_file=None, size=(256, 512),
                 return_idx=False, norm=True, random_mirror=False, transform=None, mode='train'):

        self.root = root
        self.root_mask = root_mask

        self.return_idx = return_idx

        self.img_ids = [i_id.strip().split(' ')[0] for i_id in open(info_file).readlines()]
        # if not max_iters == None:
        # self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.norm = norm
        self.random_mirror = random_mirror
        self.transform = transform
        self.size = size
        self.mode = mode
        if mode == 'val' or mode == 'test':
            self.random_mirror = False
            self.transform = None

        self.files = []
        for name in tqdm(self.img_ids):
            img_file = os.path.join(self.root, "%s" % (name))
            ins_mask_file = os.path.join(self.root_mask, *name.replace('.png', '.json').split('/')[1:])

            for ins_mask_data in json.loads(open(ins_mask_file, 'r').read()):
                label_id = ins_mask_data['category_id']
                ins_mask = np.array(mutils.decode(ins_mask_data['segmentation']), dtype=np.uint8)
                bbox = ins_mask_data['bbox']

                self.files.append({
                    "img": img_file,
                    "ins_mask": ins_mask,
                    'bbox': bbox,
                    "label_id": label_id,
                    "name": os.path.split(name)[-1],
                    "true_label": ins_mask_data['true_label'],
                    'segmentation': ins_mask_data['segmentation'],
                })

    def get_patch_region(self, image_size, bbox, size=(256, 512)):
        # size = (h,w)
        size_h, size_w = size
        img_h, img_w = image_size
        w, h = bbox[2], bbox[3]
        x, y = bbox[0], bbox[1]
        center_x, center_y = int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)
        assert w <= img_w and h <= img_h
        # w, h = w * 1.5, h * 1.5
        if w > size_w or h > size_h:
            ratio = max(h / size_h, w / size_w)
            size_h = int(ratio * size_h)
            size_w = int(ratio * size_w)

        region_temp = [center_y - size_h // 2, center_y + size_h // 2, center_x - size_w // 2, center_x + size_w // 2]

        if region_temp[0] < 0:
            gap = 0 - region_temp[0]
            region_temp[0] += gap
            region_temp[1] += gap
        if region_temp[1] > img_h:
            gap = img_h - region_temp[1]
            region_temp[0] += gap
            region_temp[1] += gap

        if region_temp[2] < 0:
            gap = 0 - region_temp[2]
            region_temp[2] += gap
            region_temp[3] += gap
        if region_temp[3] > img_w:
            gap = img_w - region_temp[3]
            region_temp[2] += gap
            region_temp[3] += gap
        region_temp[0] = max(0, region_temp[0])
        region_temp[1] = min(img_h, region_temp[1])
        region_temp[2] = max(0, region_temp[2])
        region_temp[3] = min(img_w, region_temp[3])

        # patch_w = max(w, size_w)
        # patch_h = max(h, size_h)
        # start_w_id = x if patch_w + x < img_w else patch_w - x
        # start_h_id = y if patch_h + y < img_w else patch_h - y
        return region_temp

    def __len__(self):
        return len(self.files)

    def show_label(self, label, class_color_map=None):

        from matplotlib import pyplot as plt
        label_show = Image.fromarray(label.astype('uint8')).convert('P')
        if class_color_map is not None:
            label_show.putpalette(class_color_map)
        plt.imshow(label_show)
        plt.show()

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]
        ins_mask = datafiles['ins_mask']
        label_id = datafiles['label_id']
        # BGR
        image = Image.open(datafiles["img"]).convert("RGB")
        # image = image.resize(ins_mask.shape[::-1], Image.BICUBIC)
        # transform image
        if self.transform is not None:
            image = self.transform(image)
        image = np.asarray(image, dtype=np.float32)

        if self.size is not None:
            # fx对应h，fy对应w
            fx, fy = np.ceil(image.shape[0] / ins_mask.shape[0]).astype('int'), np.ceil(image.shape[1] / ins_mask.shape[1]).astype('int')
            ins_mask = cv2.resize(ins_mask, dsize=None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
            bbox = datafiles['bbox']
            bbox[0], bbox[2] = fx * bbox[0], fx * bbox[2]
            bbox[1], bbox[3] = fy * bbox[1], fy * bbox[3]
            # self.show_label(image)
            # self.show_label(ins_mask, [0, 0, 0, 255, 255, 255])
            patch_region = self.get_patch_region(image.shape[:2], bbox, size=self.size)
            image = image[patch_region[0]:patch_region[1], patch_region[2]:patch_region[3]]
            ins_mask = ins_mask[patch_region[0]:patch_region[1], patch_region[2]:patch_region[3]]
            image = cv2.resize(image, dsize=self.size[::-1], interpolation=cv2.INTER_LINEAR)
            ins_mask = cv2.resize(ins_mask, dsize=self.size[::-1], interpolation=cv2.INTER_NEAREST)
            # self.show_label(image)
            # self.show_label(ins_mask, [0, 0, 0, 255, 255, 255])

        image = image[:, :, ::-1]  # RGB2BGR->
        if self.norm:
            image = image / 255.0
            image -= np.array([0.485, 0.456, 0.406])
            image = image / np.array([0.229, 0.224, 0.225])
        else:
            image = image - np.array([122.67892, 116.66877, 104.00699])

        if self.random_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :]
            ins_mask = ins_mask[:, ::flip]

        image = image.transpose((2, 0, 1)).astype(np.float32)
        ins_mask = ins_mask[np.newaxis, :, :]

        if self.mode == 'val' or self.mode == 'test':
            return image.copy(), ins_mask.copy(), label_id, datafiles['true_label'], datafiles['segmentation'], name

        # if self.return_idx:
        #     return image.copy(), ins_mask.copy(), label_id, name, index
        # else:
        #     return image.copy(), ins_mask.copy(), label_id, name
        if self.return_idx:
            return image.copy(), ins_mask.copy(), label_id, datafiles['true_label'], datafiles['segmentation'], name, index
        else:
            return image.copy(), ins_mask.copy(), label_id, datafiles['true_label'], datafiles['segmentation'], name


if __name__ == '__main__':
    root = './data/cityscapes'
    root_mask = '/data/yrz/repos/BETA/data/input_json/seco_vit_h_seco'
    info_file = "./data/splits/cityscapes/pix_top25_top_50_image/all_data.txt"

    dataset = BaseCityscapes(root, root_mask, info_file)

    for index in range(dataset.__len__()):
        data = dataset.__getitem__(index)
