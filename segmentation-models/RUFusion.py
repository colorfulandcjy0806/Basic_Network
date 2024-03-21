import torch
import torch.nn as nn
from regnet import RegNet
import torchvision.models as models
import torch.nn.functional as F
from ACmix import ACmix
def align_and_fuse_features(feat_group1, feat_group2):
    # 将 feat1 上采样至 feat2 的空间尺寸
    upsampled_feat1 = F.interpolate(feat_group1, size=feat_group2.shape[2:], mode='bilinear', align_corners=False)
    # 逐元素相加融合
    fused_feat = upsampled_feat1 + feat_group2
    return fused_feat

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        # Point-wise feedforward layers
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # print("x.shape: ", x.shape)
        batch_size, C, width, height = x.size()
        # Generate query, key, value tensors
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)

        # Calculate attention
        attention = self.softmax(torch.bmm(query, key))
        # attention = self.softmax(query @ key)
        # print(attention.shape)
        # print(value.permute(0, 2, 1).shape)
        # print(attention.permute(0, 2, 1).shape)
        out = torch.bmm(value.permute(0, 2, 1), attention)
        out = out.view(batch_size, C, width, height)
        # print("out: ", out.shape)
        return out

class EncoderAttention(nn.Module):
    def __init__(self, channel_sizes):
        super(EncoderAttention, self).__init__()
        self.self_attention = SelfAttention(sum(channel_sizes))
        self.fusion_convs = nn.ModuleList([nn.Conv2d(c * 2, c, kernel_size=1) for c in channel_sizes])
    def forward(self, features):
        upsampled_features = [F.interpolate(f, size=features[-1].shape[2:], mode='bilinear', align_corners=False) for f
                              in features]
        # print(upsampled_features[0].shape)
        # print(upsampled_features[1].shape)
        # print(upsampled_features[2].shape)
        # print(upsampled_features[3].shape)
        # print(upsampled_features[4].shape)
        concat_features = torch.cat(upsampled_features, dim=1)
        # print(concat_features.shape)
        attention_out = self.self_attention(concat_features)

        split_attention_out = torch.split(attention_out, [f.size(1) for f in features], dim=1)
        # print(split_attention_out[0].shape)
        # print(split_attention_out[1].shape)
        # print(split_attention_out[2].shape)
        # print(split_attention_out[3].shape)
        # print(split_attention_out[4].shape)
        f0 = align_and_fuse_features(split_attention_out[0], features[0])
        f1 = align_and_fuse_features(split_attention_out[1], features[1])
        f2 = align_and_fuse_features(split_attention_out[2], features[2])
        f3 = align_and_fuse_features(split_attention_out[3], features[3])
        f4 = align_and_fuse_features(split_attention_out[4], features[4])
        # fused_features = [self.fusion_convs[i](torch.cat((features[i], split_attention_out[i]), dim=1)) for i in
        #                   range(len(features))]
        fused_features = [f0, f1, f2, f3, f4]
        return fused_features

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        # self.conv1 = ACmix(in_size, out_size)
        # self.conv1 = ACmix(out_size, out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.gelu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.gelu(outputs)
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
        self.enhan = EncoderAttention([32, 256, 512, 896, 2048])
        self.backbone = backbone

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.regnet.forward(inputs)
        enhanced_features = self.enhan([feat1, feat2, feat3, feat4, feat5])

        # print(feat5.shape)
        # print(feat4.shape)
        # print(feat3.shape)
        # print(feat2.shape)
        # print(feat1.shape)
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