import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from torch import nn
from shapely.geometry import Polygon
import pyclipper
import errno
import sys
sys.path.append("..")
from PIL import Image
import torch.nn.functional as F

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

def text_save(filename, data):
    with open(filename,'w') as file:
        for i in range(len(data)):
            s = str(data[i]).replace('[','').replace(']','')
            s = s.replace("'",'').replace(',','') +'\n'   
            file.write(s)
    print("{} save successful !".format(filename)) 
        
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
def Cal_iou(mask_a, mask_b):
    i = np.sum(mask_a * mask_b)
    u = np.sum(mask_a) + np.sum(mask_b) - i
    return i / (u+0.1)

def get_color_pallete(npimg, dataset='voc'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    zero_pad = 256 * 3 - len(cityspallete)
    for i in range(zero_pad):
        cityspallete.append(255)
    out_img.putpalette(cityspallete)
    return out_img


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
        
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_cityscape_points(coords, labels, ax, marker_size=200):
    for point,label in zip(coords,labels):
        show_color = np.array([x/255 for x in cityspallete[label*3: label*3+3] ] + [1])
        ax.scatter(point[0], point[1], color=show_color, marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


            
def unclip(box, unclip_ratio=1.25):
    poly = Polygon(box)
    if poly.length < 0.01:
        return None
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def cal_four_para_bbox(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int16), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int16), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int16), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int16), 0, h - 1)
    
    return int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)

def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1])

class SpatialPurity(nn.Module):
    def __init__(self, in_channels=19, padding_mode='zeros', size=3):
        super(SpatialPurity, self).__init__()
        assert size % 2 == 1, "error size"
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=size,
                              stride=1, padding=int(size / 2), bias=False, padding_mode=padding_mode,
                              groups=in_channels)
        a = torch.ones((size, size), dtype=torch.float32)
        a = a.unsqueeze(dim=0).unsqueeze(dim=0)
        a = a.repeat([in_channels, 1, 1, 1])
        a = nn.Parameter(a)
        self.conv.weight = a
        self.conv.requires_grad_(False)

    def forward(self, x):
        summary = self.conv(x)
        # summary: (b, 19, h, w)
        count = torch.sum(summary, dim=1, keepdim=True)
        # count: (b, 1, h, w)
        dist = summary / count
        # dist: (b, 19, h, w)
        spatial_purity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True)
        # (b, 1, h, w), normally b = 1, (1, 1, h, w)
        return spatial_purity

def Calculate_purity(psd_label, randius = 8, class_num = 10):
    cal_purity = SpatialPurity(in_channels=class_num+1, size=2 * randius + 1).cuda()
    one_hot = torch.from_numpy(psd_label)
    one_hot[one_hot>class_num]=class_num
    one_hot = F.one_hot(one_hot.long(), num_classes=class_num+1).float()
    one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0).cuda()
    purity = cal_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
    purity = purity.cpu().numpy()
    purity[psd_label==class_num]=np.max(purity)
    return purity
        