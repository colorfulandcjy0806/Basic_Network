import os
import shutil
from sklearn.model_selection import train_test_split

# 设定源文件夹路径
data_dir = 'data'
images_dir = os.path.join(data_dir, 'images')
masks_dir = os.path.join(data_dir, 'masks')

# 获取文件名列表
images_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
masks_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])

# 确保图像和掩模文件一一对应
assert len(images_files) == len(masks_files) and all(img.split('.')[0] == mask.split('.')[0] for img, mask in zip(images_files, masks_files)), "Files mismatch"

# 划分数据集
train_images, test_images, train_masks, test_masks = train_test_split(images_files, masks_files, test_size=0.2, random_state=42)

# 创建所需的文件夹结构
for folder in ['train/images', 'train/masks', 'test/images', 'test/masks']:
    os.makedirs(os.path.join(data_dir, folder), exist_ok=True)

# 定义一个函数来复制文件
def copy_files(files, src_dir, dest_dir):
    for file in files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

# 复制文件到相应的文件夹
copy_files(train_images, images_dir, os.path.join(data_dir, 'train/images'))
copy_files(train_masks, masks_dir, os.path.join(data_dir, 'train/masks'))
copy_files(test_images, images_dir, os.path.join(data_dir, 'test/images'))
copy_files(test_masks, masks_dir, os.path.join(data_dir, 'test/masks'))
