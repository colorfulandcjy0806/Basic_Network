import torch
import torch.nn as nn
from regnet import RegNet
import torchvision.models as models

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

# out shape: torch.Size([1, 32, 320, 320])
# out shape: torch.Size([1, 256, 160, 160])
# out shape: torch.Size([1, 512, 80, 80])
# out shape: torch.Size([1, 896, 40, 40])
# out shape: torch.Size([1, 2048, 20, 20])
class Unet(nn.Module):
    def __init__(self, num_classes=1, pretrained=False, backbone='resnet50'):
        super(Unet, self).__init__()
        self.regnet = RegNet('RegNetX-16GF',
                   in_channels=3,
                   num_classes=1,
                   pretrained=True
                   )
        in_filters = [160, 512, 1024, 2944]
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # if backbone == 'resnet50':
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.regnet.forward(inputs)
        print(feat5.shape)
        print(feat4.shape)
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

if __name__ == '__main__':
    random_input = torch.randn(1, 3, 640, 640)
    model = Unet(num_classes=1)
    output = model(random_input)
    print("模型输出尺寸：", output.size())
    # print(net)