import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MobileViT_Deeplabv3plus.deeplab_mobilenet import mobilenetv2
from models.MobileViT_Deeplabv3plus.MobileViT import mobile_vit_small
from models.MobileViT_Deeplabv3plus.ACmix import ACmix
class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x

#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#

# class ASPP(nn.Module):
#     def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
#         super(ASPP, self).__init__()
#         self.branch1 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
#
#         self.branch2 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
#         self.branch3 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
#         self.branch4 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
#         self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
#         self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
#         self.branch5_relu = nn.ReLU(inplace=True)
#         self.conv_cat = nn.Sequential(
#             ACmix(dim_out*5, dim_out),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.GELU()
#         )
#         '''self.conv_cat = nn.Sequential(
# 				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
# 		 		nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 		 		nn.ReLU(inplace=True),
# 		 )'''
#
#     def forward(self, x):
#         [b, c, row, col] = x.size()
#         #-----------------------------------------#
#         #   一共五个分支
#         #-----------------------------------------#
#         conv1x1 = self.branch1(x)
#         conv3x3_1 = self.branch2(x)
#         conv3x3_2 = self.branch3(x)
#         conv3x3_3 = self.branch4(x)
#         #-----------------------------------------#
#         #   第五个分支，全局平均池化+卷积
#         #-----------------------------------------#
#         global_feature = torch.mean(x,2,True)
#         global_feature = torch.mean(global_feature,3,True)
#         global_feature = self.branch5_conv(global_feature)
#         global_feature = self.branch5_bn(global_feature)
#         global_feature = self.branch5_relu(global_feature)
#         global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
#
#         #-----------------------------------------#
#         #   将五个分支的内容堆叠起来
#         #   然后1x1卷积整合特征。
#         #-----------------------------------------#
#         feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#         result = self.conv_cat(feature_cat)
#         return result
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, norm_layer=nn.BatchNorm2d):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rates=[1, 6, 12, 18], bn_mom=0.1, acmix_channels=128):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        current_dim = dim_in

        for rate in rates:
            conv = DepthwiseSeparableConv(current_dim, dim_out, kernel_size=3, padding=rate, dilation=rate,
                                          norm_layer=lambda channels: nn.BatchNorm2d(channels, momentum=bn_mom))
            self.convs.append(conv)
            current_dim += dim_out

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.GELU()
        )

        # Assuming ACmix is a predefined layer or replace with a suitable nn.Conv2d
        # Here we're not defining ACmix as its implementation is not provided
        self.acmix = ACmix(current_dim + dim_out,
                           acmix_channels)  # You may need to adjust the ACmix layer or replace it with a suitable alternative

        self.final_conv = nn.Conv2d(acmix_channels, dim_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.GELU()

    def forward(self, x):
        [b, c, row, col] = x.size()
        features = [x]

        for conv in self.convs:
            new_feature = conv(torch.cat(features, dim=1))
            features.append(new_feature)

        global_feature = self.global_avg_pool(x)
        global_feature = F.interpolate(global_feature, (row, col), mode='bilinear', align_corners=True)
        features.append(global_feature)

        # Apply ACmix after concatenating all features
        acmix_feature = self.acmix(torch.cat(features, dim=1))
        result = self.final_conv(acmix_feature)
        result = self.bn(result)
        result = self.relu(result)

        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobileViT", pretrained=True, downsample_factor=8):
        super(DeepLab, self).__init__()
        if backbone=="mobileViT":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = mobile_vit_small()
            in_channels = 640
            low_level_channels = 64
        elif backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        # self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        self.aspp = ASPP(dim_in=in_channels, dim_out=256)
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.GELU()
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

if __name__ == '__main__':
    model = DeepLab(num_classes=1, pretrained=False)
    # random_input = torch.randn(1, 3, 640, 640)
    # output = model(random_input)
    # # 如果辅助分支未启用
    # print("模型输出尺寸：", output.shape)
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 在评估模式下不跟踪梯度
        random_input = torch.randn(1, 3, 640, 640)
        output = model(random_input)
        # 如果辅助分支未启用
        print("模型输出尺寸：", output.shape)