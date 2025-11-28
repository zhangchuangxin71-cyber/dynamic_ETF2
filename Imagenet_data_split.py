import os
import shutil
import pandas as pd

# 数据集路径
dataset_dir = r"F:\CV\Dataset\miniimagenet\images"  # 存储所有图像的目录
output_dir = r"F:\CV\Dataset\miniimagenet\imagenet100"  # 输出的根目录
train_csv = r"F:\CV\Dataset\miniimagenet\train.csv"  # 训练集 CSV 文件路径
test_csv = r"F:\CV\Dataset\miniimagenet\test.csv"  # 测试集 CSV 文件路径

# 加载 CSV 文件
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

# 创建输出目录
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


def organize_data(data, split_dir):
    """
    将数据按照类别组织到指定目录。
    """
    for _, row in data.iterrows():
        image_name, label = row['filename'], row['label']
        source_path = os.path.join(dataset_dir, image_name)
        class_dir = os.path.join(split_dir, label)

        # 创建类别文件夹
        os.makedirs(class_dir, exist_ok=True)

        # 将图像移动到对应的类别文件夹中
        dest_path = os.path.join(class_dir, image_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)  # 复制图像


# 组织训练集和测试集
organize_data(train_data, train_dir)
organize_data(test_data, test_dir)

print("Dataset organized successfully!")
