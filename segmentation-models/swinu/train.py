import torch
from torch import nn
from config import get_config
from swin_unet.vision_transformer import SwinUnet as ViT_seg

if __name__ == '__main__':
    random_input = torch.randn(1, 3, 640, 640)
    model = ViT_seg(num_classes=1)
    output = model(random_input)
    print("模型输出尺寸：", output.size())