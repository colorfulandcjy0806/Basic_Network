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
import time
from torchprofile import profile_macs

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
    preds = torch.sigmoid(preds)  # 应用sigmoid函数将logits转换为概率值
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

def dice_score(preds, labels):
    """
    计算Dice系数
    参数:
        preds: 模型预测输出，形状为[B, 1, H, W]，B是批量大小，H和W是高度和宽度
        labels: 真实标签，形状与preds相同
    返回:
        Dice系数的平均值
    """
    # 将预测输出转换为二值图像，即将概率值大于0.5的设置为1，其余为0
    preds = torch.sigmoid(preds)  # 应用sigmoid函数将logits转换为概率值
    preds_bin = preds > 0.5
    labels_bin = labels > 0.5
    
    # 计算交集
    intersection = (preds_bin & labels_bin).float().sum((2, 3))
    
    # 计算预测和真实标签的元素总和
    preds_sum = preds_bin.float().sum((2, 3))
    labels_sum = labels_bin.float().sum((2, 3))
    
    # 避免除以0，添加一个小的常数epsilon
    epsilon = 1e-6
    
    # 计算Dice系数
    dice = (2. * intersection + epsilon) / (preds_sum + labels_sum + epsilon)
    
    # 返回批量中所有Dice系数的平均值
    return dice.mean()

def mae_score(preds, labels):
    """
    计算平均绝对误差 (MAE)
    参数:
        preds: 模型预测输出，形状为[B, 1, H, W]
        labels: 真实标签，形状与preds相同
    返回:
        MAE的平均值
    """
    preds = torch.sigmoid(preds)  # 将logits转换为概率值
    mae = torch.abs(preds - labels).mean()
    return mae

def precision_recall_score(preds, labels):
    """
    计算精确度和召回率
    参数:
        preds: 模型预测输出，形状为[B, 1, H, W]
        labels: 真实标签，形状与preds相同
    返回:
        精确度和召回率的平均值
    """
    preds_bin = torch.sigmoid(preds) > 0.5
    labels_bin = labels > 0.5

    tp = (preds_bin & labels_bin).float().sum((2, 3))
    fp = (preds_bin & ~labels_bin).float().sum((2, 3))
    fn = (~preds_bin & labels_bin).float().sum((2, 3))

    epsilon = 1e-6
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    return precision.mean(), recall.mean()


def inference_speed(model, device='cuda'):
    """
    计算模型的推理速度。
    参数:
        model: 要评估的模型
        input_tensor: 输入张量
        device: 计算设备，默认为'cpu'
    返回:
        单次推理所需的平均时间（秒）
    """
    input_tensor = torch.randn(1, 3, 640, 640)
    model.eval()  # 将模型设置为评估模式
    model.to(device)
    input_tensor = input_tensor.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # 计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_tensor)
    end_time = time.time()

    avg_time = (end_time - start_time) / 100
    return avg_time

def calculate_flops(model, device='cuda'):
    """
    计算模型的FLOPs。
    参数:
        model: 要评估的模型
        input_tensor: 输入张量
        device: 计算设备，默认为'cpu'
    返回:
        模型的FLOPs（浮点运算次数）
    """
    input_tensor = torch.randn(1, 3, 640, 640)
    model.eval()  # 将模型设置为评估模式
    model.to(device)
    input_tensor = input_tensor.to(device)
    flops = profile_macs(model, input_tensor)
    return flops

def count_parameters(model):
    """
    计算模型的参数量。
    参数:
        model: 要评估的模型
    返回:
        模型的总参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
