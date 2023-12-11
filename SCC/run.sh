# train model
CUDA_VISIBLE_DEVICES=4 python train_src.py configs/Cityscapes/train_src_base.py \
    --work-dir /data/yrz/repos/BETA/checkpoints/Cityscapes/DTST_synthia_sam_vit_h_16
# eval model
CUDA_VISIBLE_DEVICES=4 python evaluate.py configs/Cityscapes/val_src_base.py \
    --load /data/yrz/repos/BETA/checkpoints/Cityscapes/seco_vit_h_seco/iter_5000.pth \
#    --work-dir /data/yrz/repos/BETA/data/output_json/seco_vit_h_seco_i5000

# bdd train
CUDA_VISIBLE_DEVICES=4 python train_src.py configs/bdd/train_src_base.py \
    --work-dir /data/yrz/repos/BETA/checkpoints/bdd/DTST_bdd_sam_vit_h

# bdd eval
CUDA_VISIBLE_DEVICES=5 python evaluate.py configs/bdd/val_src_base.py \
    --load /data/yrz/repos/BETA/checkpoints/bdd/DTST_bdd_sam_vit_h/iter_75000.pth

