import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 指定包含CSV文件的文件夹路径
folder_path = 'data/Mydata'  # 将路径替换为你的文件夹路径

# 获取文件夹中的子文件夹列表
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
# 遍历每个子文件夹
first_column_list=[]
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
        print(first_column.shape)

print(np.array(first_column_list).shape)

# # 将数据划分为训练集和测试集
train_data, test_data, train_labels, test_labels = train_test_split(np.array(first_column_list),
                                                                    np.ones(len(first_column_list)),
                                                                    test_size=0.2, random_state=42)

TRAIN_DATA = train_data
TRAIN_LABEL = train_labels
VAL_DATA = test_data
VAL_LABEL = test_labels
TEST_DATA = test_data
TEST_LABEL = test_labels
print(TEST_DATA.shape)
print(TRAIN_LABEL.shape)