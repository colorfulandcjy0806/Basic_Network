import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import MyDataset  # 引入自己设计的数据集读取类
from UNet import UNet  # 引入UNet模型
# ==================测试参数设置==================
batch_size = 3
input_channels = 1
model_path = "best_model.pth"
test_path = "data/test/"
# ==================测试参数设置==================
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


# 数据加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
test_dataset = MyDataset(test_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载模型
model = UNet(n_channels=input_channels, n_classes=1)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# 使用GPU如果可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 设置为评估模式

# 初始化评估指标
total_iou = 0.0
total_dice = 0.0
num_samples = 0

# 不需要计算梯度
with torch.no_grad():
    for images, masks in tqdm(test_dataloader):
        images, masks = images.to(device), masks.to(device)

        # 前向传播
        outputs = model(images)

        # 计算指标
        iou = iou_score(outputs, masks)
        dice = dice_score(outputs, masks)

        total_iou += iou.item() * images.size(0)
        total_dice += dice.item() * images.size(0)
        num_samples += images.size(0)

# 计算平均指标
average_iou = total_iou / num_samples
average_dice = total_dice / num_samples

print(f"测试完成. 平均IoU: {average_iou:.4f}, 平均Dice: {average_dice:.4f}")
