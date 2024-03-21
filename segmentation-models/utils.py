import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.optim.lr_scheduler import LambdaLR
import os
import math
import cv2
import numpy as np

# 定义IoU指标好
def iou_score(preds, labels):
    """
    计算IoU分数
    参数:
        preds: 模型预测输出，形状为[B, 1, H, W]，B是批量大小，H和W是高度和宽度
        labels: 真实标签，形状与preds相同
    返回:
        IoU分数的平均值
    """
    # 将预测输出转换为二值图像，即将概率值大于0.5的设置为1，其余为0
    preds_bin = preds > 0.5
    labels_bin = labels > 0.5

    # 计算交集和并集
    intersection = (preds_bin & labels_bin).float().sum((2, 3))  # 对高度和宽度维度求和
    union = (preds_bin | labels_bin).float().sum((2, 3))

    # 避免除以0，添加一个小的常数epsilon
    epsilon = 1e-6

    # 计算IoU分数
    iou = (intersection + epsilon) / (union + epsilon)

    # 返回批量中所有IoU分数的平均值
    return iou.mean()

def dice_score(output, target, threshold=0.5):
    """
    计算Dice系数。
    参数:
    - output: 模型的输出，维度为[B, C, H, W]。
    - target: 真实的标签，维度与output相同。
    - threshold: 阈值，用于将output的概率值转换为二值图像。
    返回:
    - dice的平均值。
    """
    with torch.no_grad():
        # 将输出激活并应用阈值
        output = torch.sigmoid(output) > threshold
        output = output.float()  # 确保转换为浮点数，以便进行后续计算
        
        # 确保target也是浮点数
        target = target.float()
        
        # 计算交集和各自的和
        intersection = (output * target).sum((2, 3))  # 对每个批次的每个类别，沿高和宽维度求和
        union = output.sum((2, 3)) + target.sum((2, 3))
        
        # 计算Dice系数
        dice = (2. * intersection + 1e-6) / (union + 1e-6)  # 添加平滑项以避免除以0
        
        # 返回所有批次和类别的平均Dice系数
        return dice.mean()