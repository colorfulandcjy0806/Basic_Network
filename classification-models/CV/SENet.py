import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.functional as F

# Squeeze and Excitation Block Module
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)  # Squeeze
        w = self.fc(x)
        w, b = w.split(w.data.size(1) // 2, dim=1)  # Excitation
        w = torch.sigmoid(w)

        return x * w + b  # Scale and add bias


# Residual Block with SEBlock
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.conv_lower = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.conv_upper = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.se_block = SEBlock(channels)

    def forward(self, x):
        path = self.conv_lower(x)
        path = self.conv_upper(path)

        path = self.se_block(path)

        path = x + path
        return F.relu(path)


# Network Module
class SENet(nn.Module):
    def __init__(self, in_channel, out_channel, blocks, num_classes):
        super(SENet, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(*[ResBlock(out_channel) for _ in range(blocks - 1)])

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channel, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)

        x = self.out_conv(x)
        x = F.adaptive_avg_pool2d(x, 1)

        x = x.view(x.data.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
if __name__ == '__main__':
    # 实例化模型
    model = SENet(in_channel=3, out_channel=128, blocks=10, num_classes=10)

    input_tensor = torch.randn(1, 3, 224, 224)  # (batch_size, channels, height, width)

    # 前向传播
    output = model(input_tensor)

    # 打印输出的形状
    print(output.shape)