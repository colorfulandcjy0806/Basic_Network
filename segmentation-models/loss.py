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

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1e-5
        input = torch.sigmoid(input)
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        dice_coeff = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        # 这里我们返回 1 - dice_coefficient 作为损失；因为我们想要最小化损失，而dice_coefficient是我们希望最大化的
        return 1 - dice_coeff

class EnhancedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0):
        super().__init__()
        self.dice_loss_fn = DiceLoss()
        self.ce_loss_fn = nn.BCEWithLogitsLoss()
        self.alpha = alpha  # Dice 损失的权重
        self.beta = beta  # 交叉熵损失的权重
        self.gamma = gamma  # 实例数量不匹配损失的权重

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss_fn(inputs, targets)
        ce_loss = self.ce_loss_fn(inputs, targets)

        # 模型输出后进行sigmoid激活，并二值化
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_bin = (inputs_sigmoid > 0.5).float()

        # 计算预测的连通区域个数与真实标签中连通区域个数的不匹配度
        num_pred_regions = self.count_regions(inputs_bin)
        num_target_regions = self.count_regions(targets)
        region_mismatch_loss = (num_pred_regions - num_target_regions).abs().float()

        # 损失函数结合了三部分：Dice损失、交叉熵损失和实例数量不匹配损失
        total_loss = self.alpha * dice_loss + self.beta * ce_loss + self.gamma * region_mismatch_loss.mean()
        return total_loss

    @staticmethod
    def count_regions(binary_mask):
        """
        计算给定二值掩码中的连通区域数量
        """
        num_regions = []
        for mask in binary_mask:
            mask_np = mask.squeeze().cpu().numpy()
            num_labels, labels_im = cv2.connectedComponents(mask_np.astype(np.uint8))
            num_regions.append(torch.tensor(num_labels - 1))  # 减去背景的连通区域
        return torch.stack(num_regions)
