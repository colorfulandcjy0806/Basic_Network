import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from UNet import UNet  # 引入网络结构


# 定义预测函数
def predict(model, img_path, device='cpu'):
    # 加载模型权重并设置为评估模式
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 读取和变换图像
    image = Image.open(img_path).convert("L")
    img = transform(image).unsqueeze(0)  # 增加批次维度

    # 预测
    img = img.to(device)
    with torch.no_grad():
        output = model(img)
        probs = torch.sigmoid(output)  # 将logits转换为概率
        pred_mask = probs > 0.5  # 应用阈值以获得二值掩码
        pred_mask = pred_mask.squeeze().cpu().numpy()  # 转换为numpy数组

    return pred_mask


# 初始化模型
model = UNet(n_channels=1, n_classes=1)

# 使用预测函数
img_path = 'data/train/images/0000.jpg'  # 预测图像路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pred_mask = predict(model, img_path, device)

# 可视化预测掩码
plt.imshow(pred_mask, cmap='gray')
plt.show()
