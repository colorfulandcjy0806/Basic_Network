import torch
import torch.nn as nn
from torch.utils import model_zoo


class BlockX(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck_ratio=1.0,
                 group_width=1,
                 bn_eps=1e-5,
                 bn_momentum=0.1,
                 relu_inplace=False):

        super(BlockX, self).__init__()

        # projection convolution
        if (in_channels != out_channels) or (stride != 1):
            self.project = True
        else:
            self.project = False

        if self.project:
            self.project_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=1,
                                          stride=stride,
                                          padding=0,
                                          bias=False)
            self.project_bn = nn.BatchNorm2d(num_features=out_channels,
                                             eps=bn_eps,
                                             momentum=bn_momentum)

        out_bottleneck = int(round(out_channels * bottleneck_ratio))
        groups = out_bottleneck // group_width

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_bottleneck,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        self.conv_1_bn = nn.BatchNorm2d(num_features=out_bottleneck,
                                        eps=bn_eps,
                                        momentum=bn_momentum)
        self.conv_1_relu = nn.ReLU(inplace=relu_inplace)

        self.conv_2 = nn.Conv2d(in_channels=out_bottleneck,
                                out_channels=out_bottleneck,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=groups,
                                bias=False)
        self.conv_2_bn = nn.BatchNorm2d(num_features=out_bottleneck,
                                        eps=bn_eps,
                                        momentum=bn_momentum)
        self.conv_2_relu = nn.ReLU(inplace=relu_inplace)

        self.conv_3 = nn.Conv2d(in_channels=out_bottleneck,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        self.conv_3_bn = nn.BatchNorm2d(num_features=out_channels,
                                        eps=bn_eps,
                                        momentum=bn_momentum)
        self.conv_3_bn.final_bn = True

        self.relu = nn.ReLU(inplace=relu_inplace)
        return

    def forward(self, x):
        if self.project:
            x_inp = self.project_bn(self.project_conv(x))
        else:
            x_inp = x

        x = self.conv_1(x)
        x = self.conv_1_bn(x)
        x = self.conv_1_relu(x)

        x = self.conv_2(x)
        x = self.conv_2_bn(x)
        x = self.conv_2_relu(x)

        x = self.conv_3(x)
        x = self.conv_3_bn(x)

        x = x_inp + x

        x = self.relu(x)
        return x


class BlockY(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck_ratio=1.0,
                 group_width=1,
                 se_ratio=0.25,
                 bn_eps=1e-5,
                 bn_momentum=0.1,
                 relu_inplace=False):

        super(BlockY, self).__init__()

        # projection convolution
        if (in_channels != out_channels) or (stride != 1):
            self.project = True
        else:
            self.project = False

        if self.project:
            self.project_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=1,
                                          stride=stride,
                                          padding=0,
                                          bias=False)
            self.project_bn = nn.BatchNorm2d(num_features=out_channels,
                                             eps=bn_eps,
                                             momentum=bn_momentum)

        out_bottleneck = int(round(out_channels * bottleneck_ratio))
        groups = out_bottleneck // group_width

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_bottleneck,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        self.conv_1_bn = nn.BatchNorm2d(num_features=out_bottleneck,
                                        eps=bn_eps,
                                        momentum=bn_momentum)
        self.conv_1_relu = nn.ReLU(inplace=relu_inplace)

        self.conv_2 = nn.Conv2d(in_channels=out_bottleneck,
                                out_channels=out_bottleneck,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=groups,
                                bias=False)
        self.conv_2_bn = nn.BatchNorm2d(num_features=out_bottleneck,
                                        eps=bn_eps,
                                        momentum=bn_momentum)
        self.conv_2_relu = nn.ReLU(inplace=relu_inplace)

        # squeeze-and-excitation after 3x3 conv
        w_se = int(round(in_channels * se_ratio))
        self.se_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se_reduce = nn.Conv2d(out_bottleneck, w_se, 1, bias=True)
        self.se_reduce_act = nn.ReLU(inplace=relu_inplace)
        self.se_expand = nn.Conv2d(w_se, out_bottleneck, 1, bias=True)
        self.se_expand_act = nn.Sigmoid()

        self.conv_3 = nn.Conv2d(in_channels=out_bottleneck,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        self.conv_3_bn = nn.BatchNorm2d(num_features=out_channels,
                                        eps=bn_eps,
                                        momentum=bn_momentum)
        self.conv_3_bn.final_bn = True

        self.relu = nn.ReLU(inplace=relu_inplace)
        return

    def forward(self, x):
        if self.project:
            x_inp = self.project_bn(self.project_conv(x))
        else:
            x_inp = x

        x = self.conv_1(x)
        x = self.conv_1_bn(x)
        x = self.conv_1_relu(x)

        x = self.conv_2(x)
        x = self.conv_2_bn(x)
        x = self.conv_2_relu(x)

        x_se = x
        x = self.se_avg_pool(x)
        x = self.se_reduce(x)
        x = self.se_reduce_act(x)
        x = self.se_expand(x)
        x = self.se_expand_act(x)
        x = x_se * x

        x = self.conv_3(x)
        x = self.conv_3_bn(x)

        x = x_inp + x

        x = self.relu(x)
        return x


def stage_x(num_blocks, in_channels, out_channels, stride, bottleneck_ratio,
            group_width, bn_eps, bn_momentum, relu_inplace):
    blocks = []
    for i in range(num_blocks):
        b_stride = stride if i == 0 else 1
        b_in_channels = in_channels if i == 0 else out_channels
        blocks.append(BlockX(in_channels=b_in_channels,
                             out_channels=out_channels,
                             stride=b_stride,
                             bottleneck_ratio=bottleneck_ratio,
                             group_width=group_width,
                             bn_eps=bn_eps,
                             bn_momentum=bn_momentum,
                             relu_inplace=relu_inplace))
    return nn.Sequential(*blocks)


def stage_y(num_blocks, in_channels, out_channels, stride, bottleneck_ratio,
            group_width, se_ratio, bn_eps, bn_momentum, relu_inplace):
    blocks = []
    for i in range(num_blocks):
        b_stride = stride if i == 0 else 1
        b_in_channels = in_channels if i == 0 else out_channels
        blocks.append(BlockY(in_channels=b_in_channels,
                             out_channels=out_channels,
                             stride=b_stride,
                             bottleneck_ratio=bottleneck_ratio,
                             group_width=group_width,
                             se_ratio=se_ratio,
                             bn_eps=bn_eps,
                             bn_momentum=bn_momentum,
                             relu_inplace=relu_inplace))
    return nn.Sequential(*blocks)


MODELS = {
    'RegNetX-200MF': {'init_out_channels': 32,
                      'num_blocks': [1, 1, 4, 7],
                      'out_channels': [24, 56, 152, 368],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [8, 8, 8, 8],
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905981/RegNetX-200MF_dds_8gpu.pyth',
                      },

    'RegNetX-400MF': {'init_out_channels': 32,
                      'num_blocks': [1, 2, 7, 12],
                      'out_channels': [32, 64, 160, 384],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [16, 16, 16, 16],
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth',
                      },

    'RegNetX-600MF': {'init_out_channels': 32,
                      'num_blocks': [1, 3, 5, 7],
                      'out_channels': [48, 96, 240, 528],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [24, 24, 24, 24],
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906442/RegNetX-600MF_dds_8gpu.pyth',
                      },

    'RegNetX-800MF': {'init_out_channels': 32,
                      'num_blocks': [1, 3, 7, 5],
                      'out_channels': [64, 128, 288, 672],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [16, 16, 16, 16],
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth',
                      },
    'RegNetX-1.6GF': {'init_out_channels': 32,
                      'num_blocks': [2, 4, 10, 2],
                      'out_channels': [72, 168, 408, 912],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [24, 24, 24, 24],
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth',
                      },
    'RegNetX-3.2GF': {'init_out_channels': 32,
                      'num_blocks': [2, 6, 15, 2],
                      'out_channels': [96, 192, 432, 1008],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [48, 48, 48, 48],
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906139/RegNetX-3.2GF_dds_8gpu.pyth',
                      },
    'RegNetX-4.0GF': {'init_out_channels': 32,
                      'num_blocks': [2, 5, 14, 2],
                      'out_channels': [80, 240, 560, 1360],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [40, 40, 40, 40],
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906383/RegNetX-4.0GF_dds_8gpu.pyth',
                      },
    'RegNetX-6.4GF': {'init_out_channels': 32,
                      'num_blocks': [2, 4, 10, 1],
                      'out_channels': [168, 392, 784, 1624],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [56, 56, 56, 56],
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161116590/RegNetX-6.4GF_dds_8gpu.pyth',
                      },
    'RegNetX-8.0GF': {'init_out_channels': 32,
                      'num_blocks': [2, 5, 15, 1],
                      'out_channels': [80, 240, 720, 1920],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [80, 120, 120, 120],
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161107726/RegNetX-8.0GF_dds_8gpu.pyth',
                      },
    'RegNetX-12GF': {'init_out_channels': 32,
                     'num_blocks': [2, 5, 11, 1],
                     'out_channels': [224, 448, 896, 2240],
                     'strides': [2, 2, 2, 2],
                     'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                     'group_widths': [112, 112, 112, 112],
                     'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906020/RegNetX-12GF_dds_8gpu.pyth',
                     },
    'RegNetX-16GF': {'init_out_channels': 32,
                     'num_blocks': [2, 6, 13, 1],
                     'out_channels': [256, 512, 896, 2048],
                     'strides': [2, 2, 2, 2],
                     'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                     'group_widths': [128, 128, 128, 128],
                     'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/158460855/RegNetX-16GF_dds_8gpu.pyth',
                     },
    'RegNetX-32GF': {'init_out_channels': 32,
                     'num_blocks': [2, 7, 13, 1],
                     'out_channels': [336, 672, 1344, 2520],
                     'strides': [2, 2, 2, 2],
                     'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                     'group_widths': [168, 168, 168, 168],
                     'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/158188473/RegNetX-32GF_dds_8gpu.pyth',
                     },

    'RegNetY-200MF': {'init_out_channels': 32,
                      'num_blocks': [1, 1, 4, 7],
                      'out_channels': [24, 56, 152, 368],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [8, 8, 8, 8],
                      'se_ratio': 0.25,
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/176245422/RegNetY-200MF_dds_8gpu.pyth',
                      },
    'RegNetY-400MF': {'init_out_channels': 32,
                      'num_blocks': [1, 3, 6, 6],
                      'out_channels': [48, 104, 208, 440],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [8, 8, 8, 8],
                      'se_ratio': 0.25,
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906449/RegNetY-400MF_dds_8gpu.pyth',
                      },
    'RegNetY-600MF': {'init_out_channels': 32,
                      'num_blocks': [1, 3, 7, 4],
                      'out_channels': [48, 112, 256, 608],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [16, 16, 16, 16],
                      'se_ratio': 0.25,
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160981443/RegNetY-600MF_dds_8gpu.pyth',
                      },
    'RegNetY-800MF': {'init_out_channels': 32,
                      'num_blocks': [1, 3, 8, 2],
                      'out_channels': [64, 128, 320, 768],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [16, 16, 16, 16],
                      'se_ratio': 0.25,
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906567/RegNetY-800MF_dds_8gpu.pyth',
                      },
    'RegNetY-1.6GF': {'init_out_channels': 32,
                      'num_blocks': [2, 6, 17, 2],
                      'out_channels': [48, 120, 336, 888],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [24, 24, 24, 24],
                      'se_ratio': 0.25,
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906681/RegNetY-1.6GF_dds_8gpu.pyth',
                      },
    'RegNetY-3.2GF': {'init_out_channels': 32,
                      'num_blocks': [2, 5, 13, 1],
                      'out_channels': [72, 216, 576, 1512],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [24, 24, 24, 24],
                      'se_ratio': 0.25,
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906834/RegNetY-3.2GF_dds_8gpu.pyth',
                      },
    'RegNetY-4.0GF': {'init_out_channels': 32,
                      'num_blocks': [2, 6, 12, 2],
                      'out_channels': [128, 192, 512, 1088],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [64, 64, 64, 64],
                      'se_ratio': 0.25,
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906838/RegNetY-4.0GF_dds_8gpu.pyth',
                      },
    'RegNetY-6.4GF': {'init_out_channels': 32,
                      'num_blocks': [2, 7, 14, 2],
                      'out_channels': [144, 288, 576, 1296],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [72, 72, 72, 72],
                      'se_ratio': 0.25,
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160907112/RegNetY-6.4GF_dds_8gpu.pyth',
                      },
    'RegNetY-8.0GF': {'init_out_channels': 32,
                      'num_blocks': [2, 4, 10, 1],
                      'out_channels': [168, 448, 896, 2016],
                      'strides': [2, 2, 2, 2],
                      'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                      'group_widths': [56, 56, 56, 56],
                      'se_ratio': 0.25,
                      'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161160905/RegNetY-8.0GF_dds_8gpu.pyth',
                      },
    'RegNetY-12GF': {'init_out_channels': 32,
                     'num_blocks': [2, 5, 11, 1],
                     'out_channels': [224, 448, 896, 2240],
                     'strides': [2, 2, 2, 2],
                     'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                     'group_widths': [112, 112, 112, 112],
                     'se_ratio': 0.25,
                     'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160907100/RegNetY-12GF_dds_8gpu.pyth',
                     },
    'RegNetY-16GF': {'init_out_channels': 32,
                     'num_blocks': [2, 4, 11, 1],
                     'out_channels': [224, 448, 1232, 3024],
                     'strides': [2, 2, 2, 2],
                     'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                     'group_widths': [112, 112, 112, 112],
                     'se_ratio': 0.25,
                     'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161303400/RegNetY-16GF_dds_8gpu.pyth',
                     },
    'RegNetY-32GF': {'init_out_channels': 32,
                     'num_blocks': [2, 5, 12, 1],
                     'out_channels': [232, 696, 1392, 3712],
                     'strides': [2, 2, 2, 2],
                     'bottleneck_ratios': [1.0, 1.0, 1.0, 1.0],
                     'group_widths': [232, 232, 232, 232],
                     'se_ratio': 0.25,
                     'weights_url': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161277763/RegNetY-32GF_dds_8gpu.pyth',
                     },
}


class RegNet(nn.Module):

    def __init__(self,
                 name,
                 in_channels=3,
                 num_classes=1000,
                 bn_eps=1e-5,
                 bn_momentum=0.1,
                 relu_inplace=False,
                 pretrained=False,
                 progress=False,
                 ):
        super(RegNet, self).__init__()

        config = MODELS.get(name)
        if config is None:
            raise ValueError('Invalid model name: %s' % name)

        self.init_conv = nn.Conv2d(in_channels=in_channels,
                                   out_channels=config['init_out_channels'],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   bias=False)

        self.init_bn = nn.BatchNorm2d(num_features=config['init_out_channels'],
                                      eps=bn_eps,
                                      momentum=bn_momentum)
        self.init_relu = nn.ReLU(inplace=relu_inplace)

        prev_out_channels = config['init_out_channels']

        self.stages = nn.ModuleList()
        if 'RegNetX' in name:
            for i in range(len(config['num_blocks'])):
                stage = stage_x(num_blocks=config['num_blocks'][i],
                                in_channels=prev_out_channels,
                                out_channels=config['out_channels'][i],
                                stride=config['strides'][i],
                                bottleneck_ratio=config['bottleneck_ratios'][i],
                                group_width=config['group_widths'][i],
                                bn_eps=bn_eps,
                                bn_momentum=bn_momentum,
                                relu_inplace=relu_inplace)
                self.stages.append(stage)
                prev_out_channels = config['out_channels'][i]
        elif 'RegNetY' in name:
            for i in range(len(config['num_blocks'])):
                stage = stage_y(num_blocks=config['num_blocks'][i],
                                in_channels=prev_out_channels,
                                out_channels=config['out_channels'][i],
                                stride=config['strides'][i],
                                bottleneck_ratio=config['bottleneck_ratios'][i],
                                group_width=config['group_widths'][i],
                                se_ratio=config['se_ratio'],
                                bn_eps=bn_eps,
                                bn_momentum=bn_momentum,
                                relu_inplace=relu_inplace)
                self.stages.append(stage)
                prev_out_channels = config['out_channels'][i]

        self.avpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=prev_out_channels,
                            out_features=num_classes,
                            bias=True)

        if pretrained:
            self._load_state(name, in_channels, num_classes, progress)

        return

    def _load_state(self, name, in_channels, num_classes, progress):
        state_dict = model_zoo.load_url(
            MODELS[name]['weights_url'], progress=progress, check_hash=False)

        state_dict = state_dict['model_state']
        self_state_dict = self.state_dict()
        assert len(state_dict) == len(
            self_state_dict), f"{len(state_dict)} {len(self_state_dict)}"

        strict = True

        with torch.no_grad():
            for key, self_key in zip(list(state_dict.keys()), list(self_state_dict.keys())):
                if key in ['head.fc.weight', 'head.fc.bias']:
                    if num_classes != 1000:
                        strict = False
                        continue
                elif key == 'stem.conv.weight':
                    if in_channels != 3:
                        strict = False
                        continue
                    else:
                        # original models were trained with images in BGR format,
                        # swap B and R filters in initial convolution weights in order to
                        # apadt model to RGB images
                        t = state_dict[key].detach().clone()
                        state_dict[key][:, 0, :, :] = state_dict[key][:, 2, :, :]
                        state_dict[key][:, 2, :, :] = t[:, 0, :, :]

                assert state_dict[key].shape == self_state_dict[
                    self_key].shape, f"{key}, {self_key}, {state_dict[key].shape}, {self_state_dict[self_key].shape}"

                self_state_dict[self_key] = state_dict[key]

        self.load_state_dict(self_state_dict, strict=strict)
        return

    def get_features(self, x):
        """
        Computer and return only intermediate features, no final fc layer
        """
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_relu(x)

        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out

    def forward(self, x):
        """
        forward pass
        """
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_relu(x)

        # for stage in self.stages:
        #     x = stage(x)
        x0 = x
        x1 = self.stages[0](x)
        x2 = self.stages[1](x1)
        x3 = self.stages[2](x2)
        x4 = self.stages[3](x3)

        # x = self.avpool(x)
        # x = torch.flatten(x, 1)
        #
        # x = self.fc(x)

        return x0, x1, x2, x3, x4


if __name__ == "__main__":
    model = RegNet('RegNetX-16GF',
                   in_channels=3,
                   num_classes=1,
                   pretrained=True
                   )
    # inp - tensor of shape [batch_size, in_channels, image_height, image_width]
    inp = torch.randn([1, 3, 640, 640])

    # to get predictions:
    x0, x1,x2,x3,x4 = model(inp)
    print('out shape:', x0.shape)
    print('out shape:', x1.shape)
    print('out shape:', x2.shape)
    print('out shape:', x3.shape)
    print('out shape:', x4.shape)

    # # to extract features:
    # features = model.get_features(inp)
    # for i, feature in enumerate(features):
    #     print('feature %d shape:' % i, feature.shape)