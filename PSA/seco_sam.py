import os
import numpy as np
import cv2

from tqdm import tqdm
from skimage.measure import label as sklabel
from utils import mkdir, unclip, cal_four_para_bbox, get_mini_boxes, Calculate_purity, get_color_pallete, text_save
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import argparse

# python seco_sam.py --id-list-path  './splits/cityscapes/HRDA_seco/labeled.txt'
# --data
#   --psd label
#   --leftImg8bit

# save to HRDA_seco/

parser = argparse.ArgumentParser(description='Semantic Connectivity-Driven Pseudo-labeling for Cross-domain Segmentation.')
parser.add_argument('--id-list-path', type=str, required=True)
parser.add_argument('--class-num', type=int, default=19, required=True)
parser.add_argument('--root-path', type=str, default='./data/')
parser.add_argument('--sam-checkpoint', type=str, default="./pretrain/sam_vit_h_4b8939.pth")
parser.add_argument('--sam-type', type=str, default="vit_h")

def main():
    args = parser.parse_args()
    ## init sam
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    SAM_GRIDS = False
    if SAM_GRIDS:
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
    else:
        mask_generator = SamAutomaticMaskGenerator(model=sam)
    
    
    with open(args.id_list_path, 'r') as f:
        list_ids = f.read().splitlines()
    new_id_list = []
    for id in tqdm(list_ids):
        ### read data
        exp_name = id.split(' ')[1].split('/')[0]
        exp_name_new = '{}_sam_{}'.format(exp_name, args.sam_type)
        exp_name_new = id.split(' ')[1].replace(exp_name, exp_name_new)
        save_path = os.path.join(args.root_path, exp_name_new)
        if os.path.exists(save_path):
            continue
        image = Image.open(os.path.join(args.root_path, id.split(' ')[0])).convert('RGB')
        psd_label_ori = Image.fromarray(np.array(Image.open(os.path.join(args.root_path, id.split(' ')[1]))))
        image = image.resize(psd_label_ori.size)
        psd_label = np.array(psd_label_ori)
        image = np.array(image)

        ### init label sets
        if args.class_num==19:
            STUFF_LABEL = [0,1,2,3,4,8,9,10]                      # semantic alignment
            THING_LABEL = [5,6,7,11,12,13,14,15,16,17,18]         # box + point prompt
            
        elif args.class_num==16:
            STUFF_LABEL = [0,1,2,3,4,8,9]                           # semantic alignment
            THING_LABEL = [5,6,7,10,11,12,13,14,15]         # box + point prompt            
            
        
        unique_stuff_list = []
        unique_thing_list = []
        for cid in np.unique(psd_label):
            if cid in STUFF_LABEL:
                unique_stuff_list.append(cid)
            if cid in THING_LABEL:
                unique_thing_list.append(cid)
                
        ### init return vars
        all_point = []
        all_label = []
        all_boxes = []
        final_mask = np.ones_like(psd_label) * 255
        filled_mask = np.zeros_like(psd_label)

        ### STUFF  ---->  semantic alignment
        predictor.set_image(image)
        masks = mask_generator.generate(image)
        stuff_align_thr = 0.20
        for mask_ in masks:
            mask = mask_['segmentation']
            Proportion = []
            for cid in unique_stuff_list: 
                Proportion.append(np.mean(psd_label[(mask>0) * (psd_label!=255)] == cid))       
            try:
                max_idx, max_pro = np.argmax(np.array(Proportion)), np.max(np.array(Proportion))
                if max_pro > stuff_align_thr:
                    final_mask[mask>0] = unique_stuff_list[max_idx]
            except Exception as e:
                continue
                
        ###  THING_LABEL_SMALL (instance )  ---->  box + point prompt
        minum_psd_nums = 10
        purity=Calculate_purity(psd_label, class_num = args.class_num)
        for _, label_id in enumerate(unique_thing_list):
            mask = psd_label==label_id
            
            if np.sum(mask)>minum_psd_nums:
                masknp = mask.astype(int) 
                seg, forenum = sklabel(masknp, background=0, return_num=True, connectivity=2)
                for i in range(forenum):
                    instance_id = i+1
                    if np.sum(seg==instance_id) < minum_psd_nums:
                        continue
                    ins_mask = seg==instance_id
                    cont, hierarchy = cv2.findContours(ins_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
                    cont.sort(key=lambda c: cv2.contourArea(c), reverse=True)

                    points, _ = get_mini_boxes(cont[0])
                    points = np.array(points)
                    box = unclip(points, unclip_ratio=1.25)
                    if len(box) == 0: continue
                    x, y, w, h = cal_four_para_bbox(ins_mask, box.reshape(-1, 2))
                    input_box = np.array([x, y,  x+w, y+h])
                    all_boxes.append([x, y,  x+w, y+h])
        
                    label_where = np.where(seg==instance_id)
                    top_points_nums = 2
                    select_point, select_lbl = [], []
                    point_, score_, = [], []
                    divided = len(label_where[1]) // top_points_nums
                    cur_points_nums = 1
                    for idx, xy in enumerate(zip(label_where[1], label_where[0])):
                        point_.append(xy)
                        score_.append(purity[xy[1], xy[0]])
                        if idx == divided * cur_points_nums:
                            min_indices_h = np.argmin(np.array(score_))
                            
                            select_point.append(point_[min_indices_h])
                            select_lbl.append(1)
                            
                            cur_points_nums += 1
                            score_ = []
                            point_ = []
                            
                    if len(select_point) > 0:
                        masks, score, _ = predictor.predict(
                            point_coords= np.array(select_point),
                            point_labels= np.array(select_lbl),
                            box=input_box,
                            multimask_output=False,
                        )
                    
                        all_point += select_point
                        all_label += [x* label_id for x in select_lbl]
                        
                        final_mask[masks[0]] = label_id
                            
        final_mask[filled_mask>1] = 255
        
        all_point = np.array(all_point)
        all_label = np.array(all_label)
    
        mask = get_color_pallete(final_mask, "city")
        exp_name = id.split(' ')[1].split('/')[0]
        exp_name_new = '{}_sam_{}'.format(exp_name, args.sam_type)
        exp_name_new = id.split(' ')[1].replace(exp_name, exp_name_new)
        save_path = os.path.join(args.root_path, exp_name_new)
        mkdir(os.path.dirname(save_path))
        mask.save(save_path)

        new_id_list.append(id.split(' ')[0] + ' ' + exp_name_new)
    
    list_name = args.id_list_path.split('/')[-2]
    list_name_new = '{}_sam_{}'.format(list_name, args.sam_type)
    list_name_new = args.id_list_path.replace(list_name, list_name_new)
    mkdir(os.path.dirname(list_name_new))
    text_save(list_name_new, new_id_list)
    


if __name__ == '__main__':
    main()
