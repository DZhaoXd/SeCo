# SeCo part 2: Semantic ConnectivityCorrection

代码结构借鉴[BETA]()，但是我们没有使用BETA进行训练优化，仅使用src进行训练

## Data structure

## Use Flow

step 1: 将png格式图片输出转换为input json格式的输入，用于后续训练   
step 2: 使用input json格式的输入进行模型训练  
step 3: 使用训练好的模型进行推理得到output json格式的结果  
step 4: 将output json格式结果转化为类似input json格式的结果  
step 5: 将input json格式结果进行可视化，转换为png格式图片，完成流程闭环

注意：

1. step 5可以直接接在step1之后，用于判断input json格式的数据转换是否有误
2. step2和step3脚本使用案例你都可以再run.sh当中找到
3. 在使用之前一定要确保使用的数据结构没有问题，这样才能保证完整复现出

## step 1

这一步的目的是完成将png格式图片输出转换为input json格式的输入，用于后续训练。    
对于使用cityscapes数据集和bdd数据集使用不同的脚本，对应脚本分别为`gen_ins_mask.py`和`gen_ins_mask_bdd.py`,
两者均使用了函数`gen_ins_mask`，不同点在于数据读取的写法不同。

- cityscapes数据集

```shell
python gen_ins_mask.py [input_dir] [save_dir] [root_dir] [info_path]
```

| 参数        | 说明                  |
|-----------|---------------------|
| input_dir | 伪标签结果的输入路径          |  
| save_dir  | 转换json格式实例伪标签的保存路径  |
| root_dir  | 原始数据的路径，为了读取对应的gt标签 |  
| info_path | 用于训练的数据信息文件路径       |

注意：在`gen_ins_mask.py`当中有案例可以进行参考设置

- bdd数据集

和cityscapes数据集类似，不过使用脚本更改为`gen_ins_mask_bdd.py`

```shell
python gen_ins_mask_bdd.py [input_dir] [save_dir] [root_dir] [info_path]
```

## step 2

### 校对config文件

这一步是网络模型的训练步骤，在训练之前你需要校对config文件是否设置正确。

- cityscapes  
  配置文件路径：`./configs/CityScapes/train_src_base.py`
  需要校对的参数：

| 参数        | 说明                             |
|-----------|--------------------------------|
| root      | 数据集数据路径,对应上一步`root_dir`        |
| root_mask | json格式的实例伪标签路径，对应上一步`save_dir` |
| info_file | 用于训练的数据信息文件路径，对应上一步`info_path` |

- bdd  
  配置文件路径：`./configs/bdd/train_src_base.py`
  需要校对的参数：同上

### 启动训练

```shell
CUDA_VISIBLE_DEVICES=0 python train_src.py [trian config path] \
    --work-dir [model save dir]
```

注意：所有的启动脚本范例都可以在[`run.sh`](./run.sh)找到

## step 3

### 校对config文件

- cityscapes
  配置文件路径：`./configs/CityScapes/val_src_base.py`
  需要校对的参数：同step2
- bdd
  配置文件路径：`./configs/bdd/val_src_base.py`
  需要校对的参数：同step2

### 启动验证

```shell
CUDA_VISIBLE_DEVICES=0 python evaluate.py [val config path] \
    --load [val model path]
```

注意：所有的启动脚本范例都可以在[`run.sh`](./run.sh)找到

## step 4

将推理得到的output json格式结果转化为类似input json格式的结果

```shell
python convert_json_2input.py [output json dir] [save dir]
```

| 参数              | 说明                                                                           |
|-----------------|------------------------------------------------------------------------------|
| output json dir | 模型推理得到的output json文件路径，一般为`./checkpoints/[dataset name]/val_src_base/*.json` |
| save dir        | 类似input json格式结果的保存路径                                                        |

## step 5

将input json格式结果进行可视化，转换为png格式图片，完成流程闭环

```shell
python vis_json.py [data_path] [save_dir] [filter_mode] 
```

| 参数说明        |                    |
|-------------|--------------------|
| data_path   | 类似input json格式数据路径 |
| save_dir    | 可视化结果展示路径          |
| filter_mode | 对于output结果，使用的过滤模式 |

