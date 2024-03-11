import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataset import MyDataset # 引入自己设计的数据集读取
from UNet import UNet # 引入网络结构

# ==================各种超参数==================
num_class = 1 # 因为我们做的任务有1个类别
learning_rate = 0.001
num_epochs = 20
best_iou = 0.0
batch_size = 3 # 根据自己的GPU显存来，尽量大
input_channels = 1 # 医学影像通常是灰度图，通道是1，如果是彩色的就改成3 RGB三个通道
# ==================各种超参数==================

# 定义IoU指标
def iou_score(output, target):
    with torch.no_grad():
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        target = (target > 0.5).float()

        intersection = (output * target).sum((1, 2))
        union = output.sum((1, 2)) + target.sum((1, 2)) - intersection
        smooth = 1e-6  # 添加平滑项，避免除以零
        iou = (intersection + smooth) / (union + smooth)
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce_with_logits_loss(inputs, targets)
        targets = targets.type(torch.long)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# 数据加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
dataset = MyDataset("data/train/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型、优化器和损失函数
model = UNet(n_channels=input_channels, n_classes=num_class)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

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

        # 计算指标
        iou = iou_score(outputs, masks)
        dice = dice_score(outputs, masks)

        tqdm_loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        tqdm_loop.set_postfix(loss=loss.item(), iou=iou.item(), dice=dice.item())

        # 保存最好的模型
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), "best_model.pth")

print(f"训练完成. 最好的IoU: {best_iou:.4f}")
