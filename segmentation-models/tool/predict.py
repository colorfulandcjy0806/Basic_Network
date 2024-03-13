import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from UNet import UNet  # 引入网络结构
import numpy as np

# 定义预测函数
def predict(model, img_path, mask_path, device='cpu'):
    model.to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    image = Image.open(img_path).convert('RGB')
    img_transformed = transform(image).unsqueeze(0)  # 图像预处理并增加批次维度

    img_transformed = img_transformed.to(device)
    with torch.no_grad():
        output = model(img_transformed)
        probs = torch.sigmoid(output)  # 将logits转换为概率
        pred_mask = probs > 0.5  # 应用阈值以获得二值掩码

        pred_mask = pred_mask.squeeze().cpu().numpy()  # 转换为numpy数组

    orig_mask = Image.open(mask_path).convert('L')
    orig_mask = orig_mask.resize((640, 640))
    orig_mask = np.array(orig_mask)
    orig_mask = orig_mask > 128

    image_resized = image.resize([640, 640])
    return image_resized, orig_mask, pred_mask


# 初始化模型
model = UNet(n_channels=3, n_classes=1)

# 使用预测函数
img_path = 'data/images/IMG-152310-0027.png'
mask_path = 'data/masks/IMG-152310-0027.png'  # 假设mask图像路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image, orig_mask, pred_mask = predict(model, img_path, mask_path, device)


# 可视化原mask图
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title("Original Mask")
plt.imshow(orig_mask)

# 可视化预测图
plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(pred_mask)

# 可视化分割图
segmented_img = np.array(image) * np.repeat(pred_mask[:, :, np.newaxis], 3, axis=2)
plt.subplot(1, 3, 3)
plt.title("Segmented Image")
plt.imshow(segmented_img)

plt.show()
