import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os


def get_acc(pred, ps_label, label, mask=None, return_num=False):
    true_pred = pred == label
    ps_true_pred = pred == ps_label
    samples = np.ones(len(pred))
    if mask is not None:
        true_pred = true_pred[mask]
        ps_true_pred = ps_true_pred[mask]
        samples = samples[mask]
    acc = round(true_pred.sum() / samples.sum(), 6)
    ps_acc = round(ps_true_pred.sum() / samples.sum(), 6)

    if return_num:
        return acc, ps_acc, samples.sum()
    else:
        return acc, ps_acc

def generate_correlation_heatmap(data, variable_names=None, name=None):
    # 将数据转换为DataFrame格式，并设置变量名称
    if variable_names is not None:
        df = pd.DataFrame(data, columns=variable_names)
    else:
        df = pd.DataFrame(data)

    # 计算相关性矩阵
    corr_matrix = df.corr()

    # 绘制热力图
    plt.imshow(corr_matrix, cmap='Blues', aspect='auto')
    plt.colorbar()

    if variable_names is not None:
        plt.xticks(range(len(variable_names)), variable_names, rotation='vertical')
        plt.yticks(range(len(variable_names)), variable_names)

    plt.title('Correlation Matrix Heatmap\n{}'.format(name))

    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            correlation = corr_matrix.iloc[i, j]
            color = 'red' if abs(correlation) > 0.5 else 'black'
            plt.text(j, i, "{:.2f}".format(correlation), ha='center', va='center', color=color, fontsize=8)
    plt.show()


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    variable_names = list(data[0].keys())  # 获取变量名列表

    samples = []  # 存储样本数据的二维数组

    for sample in data:
        sample_data = [sample[key] for key in variable_names]
        samples.append(sample_data)

    # return np.array(samples), variable_names
    return samples, variable_names


def remove_string_columns(samples, variable_names):
    string_columns = []

    # 检查每一列的数据类型
    for i in range(len(samples[0])):
        column_data = [sample[i] for sample in samples]

        # 检查是否所有值都是字符串类型
        if all(isinstance(value, str) for value in column_data):
            string_columns.append(i)

        # 检查是否所有值都是布尔类型
        if all(isinstance(value, bool) for value in column_data):
            # 将布尔类型转换为整数类型
            samples = [[int(sample[j]) if j == i else sample[j] for j in range(len(sample))] for sample in samples]

    # 删除对应的列和变量名
    samples = [[sample[j] for j in range(len(sample)) if j not in string_columns] for sample in samples]
    variable_names = [variable_names[i] for i in range(len(variable_names)) if i not in string_columns]

    return samples, variable_names


def remove_non_numeric_columns(samples, variable_names):
    non_numeric_columns = []

    # 检查每一列的数据类型
    for i in range(len(samples[0])):
        column_data = [sample[i] for sample in samples]

        # 检查是否所有值都是布尔类型
        if all(isinstance(value, bool) for value in column_data):
            # 将布尔类型转换为整数类型
            samples = [[int(sample[j]) if j == i else sample[j] for j in range(len(sample))] for sample in samples]

        # 检查是否所有值都是非数值类型
        if not all(isinstance(value, (int, float)) for value in column_data):
            non_numeric_columns.append(i)

    # 删除非数值类型的列和变量名
    samples = [[sample[j] for j in range(len(sample)) if j not in non_numeric_columns] for sample in samples]
    variable_names = [variable_names[i] for i in range(len(variable_names)) if i not in non_numeric_columns]

    return samples, variable_names


def add_variable(data, variable_values, variable_names, variable_name):
    # 将数据转换为DataFrame格式
    df = pd.DataFrame(data, columns=variable_names)

    if len(variable_values) != len(df):
        raise ValueError("Variable values must have the same length as the number of rows in the data.")

    # 添加新的变量列
    df[variable_name] = variable_values

    # 添加变量名到变量名列表
    variable_names.append(variable_name)

    # 返回更新后的数据和变量名列表
    return df.values.tolist(), variable_names


def determine_threshold(loss_list, target_ratio):
    """
    # case
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


if __name__ == '__main__':
    import sys


    """参数"""
    file_path = sys.argv[1]
    # case
    # file_path = '/data/yrz/repos/BETA/checkpoints/VAL/val_Test_iter_5000_20231019_233236.json'
    ratio = 0.5

    """read"""
    samples, variable_names = read_json_file(file_path)

    """calculate correlation heatmap"""
    name = os.path.basename(file_path)
    samples, variable_names = remove_non_numeric_columns(samples, variable_names)
    print('variable_names:\t', variable_names)
    generate_correlation_heatmap(samples, variable_names=variable_names, name=name)

    """get threshold"""
    samples = np.array(samples)
    variable_dict = {variable_name: value for variable_name, value in zip(variable_names, samples.T)}
    print('loss:\tmax:{:.6f},\tmin:{:.6f},\tmean:{:.6f}'.format(
        np.max(variable_dict['loss']),
        np.min(variable_dict['loss']),
        np.mean(variable_dict['loss']))
    )
    loss_threshold = determine_threshold(variable_dict['loss'], ratio)
    print('score:\tmax:{:.6f},\tmin:{:.6f},\tmean:{:.6f}'.format(
        np.max(variable_dict['score']),
        np.min(variable_dict['score']),
        np.mean(variable_dict['score']))
    )
    score_threshold = determine_threshold(variable_dict['score'], 1 - ratio)
    print()
    local_loss_threshold = np.zeros_like(variable_dict['loss'])
    class_id_set = np.unique(variable_dict['pred_id'])
    class_id_loss_threshold_list = []
    for class_id in class_id_set:
        class_id_index = variable_dict['pred_id'] == class_id
        class_loss_threshold = determine_threshold(variable_dict['loss'][class_id_index], ratio)
        local_loss_threshold[class_id_index] = class_loss_threshold
        class_id_loss_threshold_list.append(class_loss_threshold)
    print('local_loss_threshold:\tmax:{:.6f},\tmin:{:.6f},\tmean:{:.6f}'.format(
        np.max(local_loss_threshold),
        np.min(local_loss_threshold),
        np.mean(local_loss_threshold))
    )
    print('class loss threshold set:\n{}'.format(class_id_loss_threshold_list))
    local_score_threshold = np.zeros_like(variable_dict['score'])
    class_id_set = np.unique(variable_dict['pred_id'])
    class_id_score_threshold_list = []
    for class_id in class_id_set:
        class_id_index = variable_dict['pred_id'] == class_id
        class_score_threshold = determine_threshold(variable_dict['score'][class_id_index], 1 - ratio)
        local_score_threshold[class_id_index] = class_score_threshold
        class_id_score_threshold_list.append(class_score_threshold)
    print('local_score_threshold:\tmax:{:.6f},\tmin:{:.6f},\tmean:{:.6f}'.format(
        np.max(local_score_threshold),
        np.min(local_score_threshold),
        np.mean(local_score_threshold))
    )
    print('class score threshold set:\n{}'.format(class_id_score_threshold_list))

    """filter sample"""
    samples_ids = np.ones(samples.shape[0])
    true_pred_ids = variable_dict['pred_id'] == variable_dict['true_id']
    ps_true_pred_ids = variable_dict['pred_id'] == variable_dict['psl_id']

    sample_num = samples_ids.sum()
    true_num = true_pred_ids.sum()
    ps_true_num = ps_true_pred_ids.sum()

    assert true_num == (variable_dict['TP'] + variable_dict['TN']).sum()

    print('name:{}'.format(os.path.basename(file_path)))

    print('--------------no rule------------------')
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], return_num=True)
    print('no mask num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(num, acc, ps_acc))

    print('--------------global rule------------------')
    mask = (variable_dict['loss'] < loss_threshold)
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('loss mask th:{}\tnum:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(loss_threshold, num, acc, ps_acc))

    mask = (variable_dict['score'] > score_threshold)
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('score mask th:{}\t num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(score_threshold, num, acc, ps_acc))
    mask = (variable_dict['loss'] < loss_threshold) * (variable_dict['score'] > score_threshold)
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('both & :\t num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(num, acc, ps_acc))

    mask = (variable_dict['loss'] < loss_threshold) + (variable_dict['score'] > score_threshold)
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('both | :\t num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(num, acc, ps_acc))

    mask = (variable_dict['loss'] < np.mean(local_loss_threshold))
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('loss mask th:{}\tnum:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(np.mean(local_loss_threshold), num, acc, ps_acc))

    mask = (variable_dict['score'] > np.mean(local_score_threshold))
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('score mask th:{}\t num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(np.mean(local_score_threshold), num, acc, ps_acc))
    mask = (variable_dict['loss'] < np.mean(local_loss_threshold)) * (variable_dict['score'] > np.mean(local_score_threshold))
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('both & :\t num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(num, acc, ps_acc))

    mask = (variable_dict['loss'] < np.mean(local_loss_threshold)) + (variable_dict['score'] > np.mean(local_score_threshold))
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('both | :\t num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(num, acc, ps_acc))

    print('--------------local rule------------------')
    mask = (variable_dict['loss'] < local_loss_threshold)
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('local loss:\t num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(num, acc, ps_acc))

    mask = (variable_dict['score'] > local_score_threshold)
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('local score:\t num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(num, acc, ps_acc))

    print('--------------both rule------------------')

    mask = (variable_dict['loss'] < local_loss_threshold) + (variable_dict['loss'] < np.mean(local_loss_threshold))
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('local and global loss:\t num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(num, acc, ps_acc))

    mask = (variable_dict['score'] > local_score_threshold) + (variable_dict['score'] > np.mean(local_score_threshold))
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('local and global score:\t num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(num, acc, ps_acc))

    mask_1 = (variable_dict['loss'] < local_loss_threshold) + (variable_dict['loss'] < np.mean(local_loss_threshold))
    mask_2 = (variable_dict['score'] > local_score_threshold) + (variable_dict['score'] > np.mean(local_score_threshold))
    mask = mask_1 * mask_2
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('local and global both &:\t num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(num, acc, ps_acc))

    mask_1 = (variable_dict['loss'] < local_loss_threshold) + (variable_dict['loss'] < np.mean(local_loss_threshold))
    mask_2 = (variable_dict['score'] > local_score_threshold) + (variable_dict['score'] > np.mean(local_score_threshold))
    mask = mask_1 + mask_2
    acc, ps_acc, num = get_acc(variable_dict['pred_id'], variable_dict['psl_id'], variable_dict['true_id'], mask=mask, return_num=True)
    print('local and global both |:\t num:{}\n'
          'true acc: {}\t'
          'ps   acc: {}\n'
          ''.format(num, acc, ps_acc))
