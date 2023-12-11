# train_src.py

# model
num_classes = 19
model = dict(type='ResNet', depth=101, num_classes=num_classes, pretrained=True)
loss = dict(
    train=dict(type='SmoothCE'),
    val=dict(type='CrossEntropyLoss'),
)

# data
root = './data/cityscapes'
root_mask = '/data/yrz/repos/BETA/data/input_json/sam_v6_vit_h_pix_top_25'
info_file = "./data/splits/cityscapes/pix_top25_top_50_image/all_data.txt"

batch_size = 64
num_workers = 4

eps = 0.1  # label smoothing
data = dict(
    train=dict(
        ds_dict=dict(
            type='BaseCityscapes',
            root=root,
            root_mask=root_mask,
            info_file=info_file,
            random_mirror=True,
            mode='train'
        ),
        trans_dict=dict(type=None),

    ),
    val=dict(
        ds_dict=dict(
            type='BaseCityscapes',
            root=root,
            root_mask=root_mask,
            info_file=info_file,
            random_mirror=False,
            mode='val'
        ),
        trans_dict=dict(type=None),
    ),
)

# training optimizer & scheduler
local_rank = 0
device_ids = [3, 4, 5, 6]
distributed = False
epochs = 20
lr = 0.01
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

# log & save
log_interval = 100
# work_dir = 'checkpoints/Cityscapes/stuff_and_thing_pixel_50_vit_h_pix_top_50_i1000'
work_dir = '/data/yrz/repos/BETA/checkpoints/Cityscapes/seco_vit_h_seco'
# work_dir = '/data/yrz/repos/BETA/checkpoints/Cityscapes/DTST_synthia_sam_vit_h_16'
resume = None
load = None
port = 10001
save_interval = 1000
