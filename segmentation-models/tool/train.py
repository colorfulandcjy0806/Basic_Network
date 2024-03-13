import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataset import MyDataset # 引入自己设计的数据集读取
from UNet import UNet # 引入网络结构
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import os
import math
import cv2
import numpy as np
# ==================各种超参数==================
num_class = 1 # 因为我们做的任务有1个类别
learning_rate = 0.01
max_lr = 1e-2   # 最大学习率，预热阶段结束时的学习率
warmup_epochs = 5 # 预热epoch 可以不动
num_epochs = 100
best_iou = 0.0
batch_size = 3 # 根据自己的GPU显存来，尽量大
input_channels = 3 # 医学影像通常是灰度图，通道是1，如果是彩色的就改成3 RGB三个通道
# ==================各种超参数==================

# tensorboard引入日志记录
writer = SummaryWriter('runs/unet_experiment')


# 学习率调整函数
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (max_lr - learning_rate) / warmup_epochs * epoch + learning_rate
    else:
        return max_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

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

# 定义Dice指标
def dice_score(output, target):
    with torch.no_grad():
        smooth = 1e-5
        output = torch.sigmoid(output) > 0.5
        output = output.float()
        target = target.float()
        intersection = (output * target).sum((1, 2))
        dice = (2. * intersection + smooth) / (output.sum((1, 2)) + target.sum((1, 2)) + smooth)
    return dice.mean()


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

# 数据加载
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
dataset = MyDataset("data", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型、优化器和损失函数
model = UNet(n_channels=input_channels, n_classes=num_class)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = EnhancedLoss()
# 应用学习率调度器
scheduler = LambdaLR(optimizer, lr_lambda)

# 检查是否有可用的检查点————断点训练用的
checkpoint_path = 'checkpoint.pth'
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state']).to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    best_iou = checkpoint['best_iou']
    start_epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {start_epoch}.")
else:
    start_epoch = 0
    print("No checkpoint found.")


# 使用GPU如果可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    tqdm_loop = tqdm(dataloader, leave=True, total=len(dataloader))
    for images, masks in tqdm_loop:
        images, masks = images.to(device), masks.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(outputs)
        # print(masks)
        # 计算指标
        iou = iou_score(outputs, masks)
        dice = dice_score(outputs, masks)
        # iou = edge_iou_score(outputs, masks, simple_edge_detector)
        # dice = edge_dice_score(outputs, masks, simple_edge_detector)

        tqdm_loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        tqdm_loop.set_postfix(loss=loss.item(), iou=iou.item(), dice=dice.item())

        # 保存最好的模型
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), "best_model.pth")
            print("已保存最好的模型")
        # 使用TensorBoard记录
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('IoU/train', iou.item(), epoch)
        writer.add_scalar('Dice/train', dice.item(), epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # 每个epoch保存检查点
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_iou': best_iou,
        }, checkpoint_path)

        scheduler.step()


writer.close()  # 关闭TensorBoard写入器
print(f"训练完成. 最好的IoU: {best_iou:.4f}")