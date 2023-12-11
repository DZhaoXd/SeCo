# SeCo Part 2: Semantic Connectivity Correction

The code structure of SeCo is inspired by [BETA](https://github.com/xyupeng/BETA), but we did not use BETA for training optimization. We only used the "src" (source code) for training.

## Data Structure

## How to Use

1. **Step 1:** Convert input PNG images to JSON format for subsequent training.

2. **Step 2:** Train the model using the input data in JSON format.

3. **Step 3:** Use the trained model for inference and obtain output results in JSON format.

4. **Step 4:** Convert the output JSON results back to a format similar to the input JSON format.

5. **Step 5:** Visualize the input JSON format results, convert them to PNG images, and complete the workflow.

**Note:**

- Step 5 can be directly connected after Step 1 to verify the correctness of the input data conversion in JSON format.
- Examples of running the scripts for Step 2 and Step 3 can be found in the "run.sh" file.
- Before using the code, make sure the data structure is correct to ensure the complete reproduction of SeCo.

Please follow the above steps and considerations to use SeCo effectively.

## Step 1

The purpose of this step is to convert PNG format images into input JSON format for subsequent training. Different scripts are used for the Cityscapes dataset and BDD dataset, namely `gen_ins_mask.py` and `gen_ins_mask_bdd.py`,
respectively. Both scripts utilize the `gen_ins_mask` function, with the difference lying in the way data is read.

- For the Cityscapes dataset:

```shell
python gen_ins_mask.py [input_dir] [save_dir] [root_dir] [info_path]
```

| Parameter | Description                                     |
|-----------|-------------------------------------------------|
| input_dir | Path to the input directory of pseudo-labels    |
| save_dir  | Path to save the converted JSON format labels   |
| root_dir  | Path to the original data for reading GT labels |
| info_path | Path to the data information file for training  |

Note: There are examples in `gen_ins_mask.py` that can be referred to for configuration.

- For the BDD dataset:

The process is similar to the Cityscapes dataset, but the script is changed to `gen_ins_mask_bdd.py`.

```shell
python gen_ins_mask_bdd.py [input_dir] [save_dir] [root_dir] [info_path]
```

## Step 2

### Verify Config File

This step involves training the network model. Before training, you need to verify the correctness of the config file.

- For Cityscapes:
  Config file path: `./configs/CityScapes/train_src_base.py`
  Parameters to verify:

| Parameter | Description                                                                  |
|-----------|------------------------------------------------------------------------------|
| root      | Path to the dataset data, corresponding to `root_dir`                        |
| root_mask | Path to the JSON format instance pseudo-labels, corresponding to `save_dir`  |
| info_file | Path to the data information file for training, corresponding to `info_path` |

- For BDD:
  Config file path: `./configs/bdd/train_src_base.py`
  Parameters to verify: Same as above.

### Start Training

```shell
CUDA_VISIBLE_DEVICES=0 python train_src.py [train config path] \
    --work-dir [model save dir]
```

Note: Examples of all the startup scripts can be found in the [`run.sh`](./run.sh) file.

## Step 3

### Verify Config File

- For Cityscapes:
  Config file path: `./configs/CityScapes/val_src_base.py`
  Parameters to verify: Same as Step 2.

- For BDD:
  Config file path: `./configs/bdd/val_src_base.py`
  Parameters to verify: Same as Step 2.

### Start Validation

```shell
CUDA_VISIBLE_DEVICES=0 python evaluate.py [val config path] \
    --load [val model path]
```

Note: Examples of all the startup scripts can be found in the [`run.sh`](./run.sh) file.

## Step 4

Convert the output JSON format results obtained from inference into a format similar to the input JSON format.

```shell
python convert_json_2input.py [output json dir] [save dir]
```

| Parameter       | Description                                                                                                               |
|-----------------|---------------------------------------------------------------------------------------------------------------------------|
| output json dir | Path to the output JSON files obtained from model inference, typically `./checkpoints/[dataset name]/val_src_base/*.json` |
| save dir        | Path to save the results in a format similar to the input JSON format                                                     |

## Step 5

Visualize the input JSON format results, convert them to PNG format images, and complete the workflow.

```shell
python vis_json.py [data_path] [save_dir] [filter_mode]
```

| Parameter   | Description                            |
|-------------|----------------------------------------|
| data_path   | Path to the input JSON format data     |
| save_dir    | Path to save the visualization results |
| filter_mode | Filter mode for the output results     |
