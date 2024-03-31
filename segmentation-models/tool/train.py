import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataset import MyDataset # 引入自己设计的数据集读取
# from UNet import UNet # 引入网络结构
from segformer_backbone import mit_b1
from segformer import SegFormer
from resunet import Unet
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import os
import math
import cv2
import numpy as np
from utils import iou_score, dice_score
from loss import *

# 学习率调整函数
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (max_lr - learning_rate) / warmup_epochs * epoch + learning_rate
    else:
        return max_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
# ==================各种超参数==================
num_class = 1 # 因为我们做的任务有1个类别
learning_rate = 0.01
max_lr = 1e-2   # 最大学习率，预热阶段结束时的学习率
warmup_epochs = 10 # 预热epoch，可以不动
num_epochs = 200
best_iou = 0.0
batch_size = 1 # 根据自己的GPU显存来，尽量大
input_channels = 3 # 医学影像通常是灰度图，通道是1，如果是彩色的就改成3 RGB三个通道
writer = SummaryWriter('../../tf-logs') # tensorboard引入日志记录
train_pth = "train" # 训练集的路径
test_pth = "test" # 测试集的路径
# ==================各种超参数==================

# 数据加载
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
dataset = MyDataset(train_pth, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_dataset = MyDataset(test_pth, transform=transform)  # 确保路径正确
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# 模型、优化器和损失函数
# model = UNet(n_channels=input_channels, n_classes=num_class)
# model = Unet(num_classes=1, pretrained=True, backbone="resnet50")
model = SegFormer(num_classes=1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = EnhancedLoss()
# 应用学习率调度器
scheduler = LambdaLR(optimizer, lr_lambda)

def evaluate_model(model, dataloader, device):
    model.eval()  # 设置为评估模式
    total_iou, total_dice, total_loss = 0.0, 0.0, 0.0
    with torch.no_grad():  # 在评估时不计算梯度
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            iou = iou_score(outputs, masks)
            dice = dice_score(outputs, masks)
            total_loss += loss.item()
            total_iou += iou.item()
            total_dice += dice.item()
    # 计算平均指标
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    return avg_loss, avg_iou, avg_dice

# 使用GPU如果可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 检查是否有可用的检查点————断点训练用的
checkpoint_path = 'checkpoint_best_test_iou.pth'
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    best_test_iou = checkpoint['best_test_iou']
    start_epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {start_epoch}.")
else:
    start_epoch = 0
    print("No checkpoint found.")
      
best_test_iou = 0.0
best_epoch = 0

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
        
        # 计算指标
        iou = iou_score(outputs, masks)
        dice = dice_score(outputs, masks)

        tqdm_loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        tqdm_loop.set_postfix(loss=loss.item(), iou=iou.item(), dice=dice.item())

        # 使用TensorBoard记录
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('IoU/train', iou.item(), epoch)
        writer.add_scalar('Dice/train', dice.item(), epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
                             
    # 在训练循环结束时
    test_loss, test_iou, test_dice = evaluate_model(model, test_dataloader, device)
    print(f"Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}, Test Dice: {test_dice:.4f}")

    # 使用TensorBoard记录测试指标
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('IoU/test', test_iou, epoch)
    writer.add_scalar('Dice/test', test_dice, epoch)
    
    # 检查是否为最佳模型
    if test_iou > best_test_iou:
        print(f"New best Test IoU: {test_iou:.4f} at epoch {epoch+1}")
        best_test_iou = test_iou
        best_epoch = epoch + 1
        # 保存最好的模型状态
        torch.save(model.state_dict(), "best_model_test_iou.pth")
        # 保存检查点
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_test_iou': best_test_iou,
        }, 'checkpoint_best_test_iou.pth')
        
    # 更新学习率
    # scheduler.step()

writer.close()  # 关闭TensorBoard写入器
print(f"训练完成. 最好的测试IoU: {best_test_iou:.4f} 在 epoch {best_epoch}")