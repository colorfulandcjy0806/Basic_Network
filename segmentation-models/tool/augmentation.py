import os
import cv2
import numpy as np
from albumentations import Compose, OneOf, HorizontalFlip, VerticalFlip, Rotate, RandomScale, ColorJitter, RandomBrightnessContrast, RandomCrop, ShiftScaleRotate, GaussNoise

def create_augmentations():
    # 使用OneOf确保每次仅应用一种增强
    return Compose([
        OneOf([
            HorizontalFlip(p=1),
            VerticalFlip(p=1),
            Rotate(limit=90, p=1),
            RandomScale(scale_limit=0.1, p=1),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=1),
        ], p=1)
    ], additional_targets={'mask': 'image'})

# 初始化数据增强
augmentations = create_augmentations()

# 图像和掩码的文件夹路径
image_dir = 'image'
mask_dir = 'mask'
# 增强后存放的文件夹路径
augmented_image_dir = 'augmented_images'
augmented_mask_dir = 'augmented_masks'

# 创建存放增强后图像和掩码的文件夹
os.makedirs(augmented_image_dir, exist_ok=True)
os.makedirs(augmented_mask_dir, exist_ok=True)

# 循环次数
n_iterations = 5

# 遍历图像文件夹
for img_file in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_file)
    mask_file = img_file.replace('.jpg', '.png')
    mask_path = os.path.join(mask_dir, mask_file)

    # 读取图像和掩码
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    for i in range(n_iterations):
        # 应用增强
        augmented = augmentations(image=img, mask=mask)
        aug_img, aug_mask = augmented['image'], augmented['mask']

        # 将掩码二值化，前景类像素点设置为1，背景类像素点设置为0
        _, aug_mask = cv2.threshold(aug_mask, 1, 1, cv2.THRESH_BINARY)

        # 构建增强后的文件名
        augmented_img_file = f'{os.path.splitext(img_file)[0]}_aug{i}{os.path.splitext(img_file)[1]}'
        augmented_mask_file = f'{os.path.splitext(mask_file)[0]}_aug{i}{os.path.splitext(mask_file)[1]}'

        # 保存增强后的图像和掩码
        cv2.imwrite(os.path.join(augmented_image_dir, augmented_img_file), aug_img)
        cv2.imwrite(os.path.join(augmented_mask_dir, augmented_mask_file), aug_mask)
