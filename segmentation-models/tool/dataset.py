import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'images/*.jpg'))
        self.transform = transform

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        label_path = image_path.replace('images', 'masks')
        # 读取训练图片和标签图片
        image = Image.open(image_path).convert("L")
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            # 对图像应用变换
            image_transform = self.transform
            image = image_transform(image)

            # 对标签应用相同的变换，但首先转换为PIL图像
            label_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 确保和图像变换尺寸一致
                transforms.ToTensor(),
            ])
            label = Image.fromarray(label)
            label = label_transform(label)

        # 由于transforms.ToTensor()会将单通道图像转换成1xHxW的张量，因此直接返回转换后的label即可
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为 (224, 224)
        transforms.ToTensor(),  # 将图像转换为 Tensor 格式
        # transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
        # transforms.RandomVerticalFlip(),  # 随机垂直翻转图像
    ])

    mydataset = MyDataset("data/train/", transform=transform)

    print("数据个数：", len(mydataset))
    train_loader = torch.utils.data.DataLoader(dataset=mydataset,
                                               batch_size=1,
                                               shuffle=True)
    for image, label in train_loader:
        print("image的尺寸: ", image.shape)
        print("label的尺寸: ", label.shape)
        break # 只看一对的，所以提前break终止了
