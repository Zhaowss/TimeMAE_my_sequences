import numpy as np
import torch
from scipy.io import arff
import os
import pandas as pd
from sklearn.model_selection import train_test_split
def padding_varying_length(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :][np.isnan(data[i, j, :])] = 0
    return data


def load_UCR(Path='data/', folder='Cricket'):
    train_path = Path + folder + '/' + folder + '_TRAIN.arff'
    test_path = Path + folder + '/' + folder + '_TEST.arff'
    TRAIN_DATA = []
    TRAIN_LABEL = []
    label_dict = {}
    label_index = 0
    with open(train_path, encoding='UTF-8', errors='ignore') as f:
        data, meta = arff.loadarff(f)
        f.close()
    if type(data[0][0]) == np.ndarray:  # multivariate
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            raw_data_list = raw_data.tolist()
            # print(raw_data_list)
            TRAIN_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            TEST_LABEL.append(label_dict[raw_label])
            raw_data_list = raw_data.tolist()
            TEST_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        index = np.arange(0, len(TRAIN_DATA))
        np.random.shuffle(index)

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        return [np.array(TRAIN_DATA)[index], np.array(TRAIN_LABEL)[index]], \
               [np.array(TRAIN_DATA)[index], np.array(TRAIN_LABEL)[index]], \
               [np.array(TEST_DATA), np.array(TEST_LABEL)]

    else:  # univariate
        for index in range(data.shape[0]):
            raw_data = np.array(list(data[index]))[:-1]
            raw_label = data[index][-1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            TRAIN_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = np.array(list(data[index]))[:-1]
            raw_label = data[index][-1]
            TEST_LABEL.append(label_dict[raw_label])
            TEST_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
            np.array(TEST_DATA), np.array(TEST_LABEL)]


def load_HAR(Path='data/HAR/'):
    train = torch.load(Path + 'train.pt')
    val = torch.load(Path + 'val.pt')
    test = torch.load(Path + 'test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    TRAIN_LABEL = train['labels'].long()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    VAL_LABEL = val['labels'].long()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    TEST_LABEL = test['labels'].long()

    ALL_TRAIN_DATA = torch.cat([TRAIN_DATA, VAL_DATA])
    ALL_TRAIN_LABEL = torch.cat([TRAIN_LABEL, VAL_LABEL])
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]


# 新增自制的数据集的加载的代码
# 主要的功能:加载指定的目录的下的文件
# 遍历三个不同类别的数据集的所有的CSv的文件读取其中的数据


# 后续需要新增类别的代码，需要更改此处的代码，将label替换为每个数据对应的类别

def load_mydata(Path='data/Mydata/'):
    # 获取文件夹中的子文件夹列表
    subfolders = [f.path for f in os.scandir(Path) if f.is_dir()]
    # 遍历每个子文件夹
    first_column_list = []
    for subfolder in subfolders:
        # 获取子文件夹中的CSV文件列表
        csv_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith('.csv')]
        # 遍历每个CSV文件
        for csv_file in csv_files:
            # 读取CSV文件
            df = pd.read_csv(csv_file)

            # 提取第一列数据
            first_column = df.iloc[:, 1].to_numpy()

            # 将第一列数据存储在列表中
            first_column_list.append(first_column)
    result_array = np.concatenate((np.ones(len(first_column_list)//2), np.full(len(first_column_list) - len(first_column_list)//2, 2)))
    # 此处新增将数据划分为训练集和测试集，并返回

    train_data, test_data, train_labels, test_labels = train_test_split(np.expand_dims(np.array(first_column_list), axis=2), result_array,
                                                                        test_size=0.2, random_state=42)


    TRAIN_DATA = train_data
    TRAIN_LABEL = train_labels
    VAL_DATA =test_data
    VAL_LABEL = test_labels
    TEST_DATA =test_data
    TEST_LABEL =test_labels

    ALL_TRAIN_DATA = torch.cat([torch.tensor(TRAIN_DATA), torch.tensor(VAL_DATA)])
    ALL_TRAIN_LABEL = torch.cat([torch.tensor(TRAIN_LABEL), torch.tensor(VAL_LABEL)])
    print('data loaded')

    return [np.array(ALL_TRAIN_DATA), np.array(ALL_TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]


def load_mat(Path='data/AUSLAN/'):
    if 'UWave' in Path:
        train = torch.load(Path + 'train_new.pt')
        test = torch.load(Path + 'test_new.pt')
    else:
        train = torch.load(Path + 'train.pt')
        test = torch.load(Path + 'test.pt')
    TRAIN_DATA = train['samples'].float()
    TRAIN_LABEL = (train['labels'] - 1).long()
    TEST_DATA = test['samples'].float()
    TEST_LABEL = (test['labels'] - 1).long()
    print('data loaded')

    return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
        np.array(TEST_DATA), np.array(TEST_LABEL)]
