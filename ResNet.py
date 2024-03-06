# # 模型选择和实现
# '''
# 深度残差网络（ResNet）的主要好处包括：
#
# 解决了梯度消失和梯度爆炸问题： 通过引入残差连接（residual connections），即将输入直接添加到输出中，可以在深层网络中更有效地传播梯度，减少了梯度消失和梯度爆炸的问题，有助于训练非常深的神经网络。
#
# 加速了训练收敛： 由于残差连接的引入，使得网络更容易优化，加速了训练收敛的速度，同时也有助于避免了训练过程中的退化问题（degradation problem）。
#
# 提高了网络性能： 深度残差网络通常能够学习到更好的特征表示，从而在许多视觉任务中取得了更好的性能。由于残差块的堆叠，网络可以学习到更复杂的特征，提高了模型的表达能力。
#
# 模型更深，但却更容易训练： 残差块的设计使得网络可以非常深，例如，ResNet-152拥有152层，但它却相对容易训练和优化，同时还能够在训练集上获得更低的错误率。
#
# 更好地适应不同的数据集和任务： 由于残差块的通用性和强大的特征学习能力，深度残差网络在各种视觉任务中表现良好，包括图像分类、目标检测、图像分割等。
# '''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

class Bottleneck(nn.Module):
    '''
    包含三种卷积层
    conv1-压缩通道数
    conv2-提取特征
    conv3-扩展通道数
    这种结构可以更好的提取特征，加深网络，并且可以减少网络的参数量。
    '''
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # Corrected line
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        '''
        这块实现了残差块结构

        ResNet50有两个基本的块，分别名为Conv Block和Identity Block，renet50就是利用了这两个结构堆叠起来的。
        它们最大的差距是残差边上是否有卷积。

        Identity Block是正常的残差结构，其残差边没有卷积，输入直接与输出相加；
        Conv Block的残差边加入了卷积操作和BN操作（批量归一化），其作用是可以通过改变卷积操作的步长通道数，达到改变网络维度的效果。

        也就是说
        Identity Block输入维度和输出维度相同，可以串联，用于加深网络的；
        Conv Block输入和输出的维度是不一样的，所以不能连续串联，它的作用是改变网络的维度。
        :param
        x:输入数据
        :return:
        out:网络输出结果
        '''
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        # -----------------------------------#
        #   假设输入进来的图片是3，224，224
        # -----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        用于构造Conv Block 和 Identity Block的堆叠
        :param block:就是上面的Bottleneck，用于实现resnet50中最基本的残差块结构
        :param planes:输出通道数
        :param blocks:残差块重复次数
        :param stride:步长
        :return:
        构造好的Conv Block 和 Identity Block的堆叠网络结构
        '''
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#

        # 边（do构建Conv Block的残差wnsample）
        if stride != 1 or self.inplanes != planes * block.expansion:# block.expansion=4
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [] # 用于堆叠Conv Block 和 Identity Block
        # 添加一层Conv Block
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 添加完后输入维度变了，因此改变inplanes（输入维度）
        self.inplanes = planes * block.expansion
        # 添加blocks层 Identity Block
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        x = self.avgpool(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # 生成随机输入
    random_input = torch.randn(1, 3, 224, 224)  # 批量大小为 3，通道数为 3，图像尺寸为 224x224
    model = ResNet(Bottleneck, [3, 4, 6, 3])

    # 加载预训练权重
    pretrained_dict = torch.load("resnet50.pth")

    # 移除最后一层的权重
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ["fc.weight", "fc.bias"]}

    # 更新模型权重
    model.load_state_dict(pretrained_dict, strict=False)

    # 替换最后一层以匹配新的分类任务
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)


    # 运行模型
    output = model(random_input)
    print("模型输出尺寸：", output.size())