import torch
import torch.nn as nn
import torch.nn.functional as F
from ACmix import ACmix
# 定义双卷积模块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# 定义下采样模块
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

"""特征融合模块，用于融合和加权不同层次的特征图"""
class FeatureFusionModule(nn.Module):
    def __init__(self, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = None
        self.fc2 = None
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        if self.fc1 is None or self.fc2 is None:
            self.fc1 = nn.Linear(c, c // self.reduction).to(x.device)
            self.fc2 = nn.Linear(c // self.reduction, c).to(x.device)
        y = self.global_avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.acmix = ACmix(in_channels, out_channels)
        self.feature_fusion = FeatureFusionModule(in_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2 = self.feature_fusion(x2)
        x = torch.cat([x2, x1], dim=1)
        return self.acmix(x), x

# 定义输出卷积模块
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Sequential(
            nn.Conv2d(1024 + 512 + 256 + 128, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        # self.outc = OutConv(64, n_classes)

    def forward(self, x):
        features = []  # 用于收集特征图的列表
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # print("x4.shape: ", x4.shape)
        x5 = self.down4(x4)
        # print("x5.shape: ", x5.shape)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # 上采样过程
        x, up_feat1 = self.up1(x5, x4)
        x, up_feat2 = self.up2(x, x3)
        x, up_feat3 = self.up3(x, x2)
        x, up_feat4 = self.up4(x, x1)
        up_feat1 = F.interpolate(up_feat1, size=(640, 640), mode='bilinear', align_corners=True)
        up_feat2 = F.interpolate(up_feat2, size=(640, 640), mode='bilinear', align_corners=True)
        up_feat3 = F.interpolate(up_feat3, size=(640, 640), mode='bilinear', align_corners=True)
        features.append(up_feat1)
        features.append(up_feat2)
        features.append(up_feat3)
        features.append(up_feat4)
        # Concat特征图
        final_feat = torch.cat(features, dim=1)
        logits = self.outc(final_feat)
        return logits

if __name__ == '__main__':
    random_input = torch.randn(1, 3, 640, 640)
    model = UNet(n_channels=3, n_classes=1)
    output = model(random_input)
    print("模型输出尺寸：", output.size())
    # print(net)
