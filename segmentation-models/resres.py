import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from ACmix import ACmix

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

        self.acmix = ACmix(in_channels, out_channels)  # Assuming ACmix is defined elsewhere
        self.feature_fusion = FeatureFusionModule(in_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2 = self.feature_fusion(x2)
        x = torch.cat([x2, x1], dim=1)
        return self.acmix(x), x


class UNetResNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, pretrained=True):
        super(UNetResNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Load ResNet50 pretrained model
        self.resnet = resnet50(pretrained=pretrained)

        # Disable gradient updates for the ResNet layers to freeze it
        for param in self.resnet.parameters():
            param.requires_grad = False

        # ResNet feature blocks
        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # Up-sampling layers
        self.up1 = Up(2048 + 1024, 512, bilinear)
        self.up2 = Up(512 + 512, 256, bilinear)
        self.up3 = Up(256 + 256, 128, bilinear)
        self.up4 = Up(128 + 64, 64, bilinear)

        # Final convolution
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Upsampling + skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)

