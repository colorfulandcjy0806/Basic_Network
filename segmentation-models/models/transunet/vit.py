import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num  # 头的数量，决定了注意力机制的复杂度和能力
        self.dk = (embedding_dim // head_num) ** (1 / 2)  # 每个头的维度，根据embedding_dim和head_num计算得到

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        # 查询（query）、键（key）、值（value）的线性变换层，输入维度是embedding_dim，
        # 输出维度是embedding_dim * 3，因为需要分别生成查询、键、值
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # 输出注意力的线性变换层，输入维度是embedding_dim，输出维度是embedding_dim

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)  # 对输入x进行查询、键、值的线性变换，将输入x映射到查询、键、值的空间

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        # 重排数据，将得到的查询、键、值重新排列正确的维度顺序，以便进行注意力计算
        # 'b t (d k h)'表示原始数据的维度顺序，k=3表示'查询、键、值'这三个部分，h=self.head_num表示头的数量
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk
        # 通过查询和键的内积计算注意力能量分数，使用**einsum**函数按照矩阵乘法规则进行计算
        # 结果的维度为"batch_size × head_num × sequence_length × sequence_length"

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)  # 如果有mask，则将对应位置的注意力分数设为负无穷，以实现对输入的屏蔽

        attention = torch.softmax(energy, dim=-1)  # 对能量分数进行softmax操作，得到注意力权重，使其相加等于1

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)
        # 根据注意力权重对值进行加权求和，使用**einsum**函数按照矩阵乘法规则进行计算
        # 得到每个位置的加权值

        x = rearrange(x, "b h t d -> b t (h d)")  # 重新排列输出的维度顺序，将头的维度和值的维度连接在一起
        x = self.out_attention(x)  # 通过线性变换层将加权值映射到最终输出的维度

        return x


class MLP(nn.Module):   # 多层感知机
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),  # 第一层线性变换，将输入维度embedding_dim映射到mlp_dim
            nn.GELU(),  # GELU激活函数，用于增加模型的非线性能力
            nn.Dropout(0.1),  # Dropout层，以0.1的概率随机将输入设置为0，用于防止过拟合
            nn.Linear(mlp_dim, embedding_dim),  # 第二层线性变换，将维度mlp_dim映射回embedding_dim
            nn.Dropout(0.1)  # 再次使用Dropout层，以0.1的概率随机将输入设置为0
        )

    def forward(self, x):
        x = self.mlp_layers(x)  # 将输入x经过MLP的各层操作，得到输出

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        # embedding_dim输入特征的维度，head_num头的数量，mlp_dim是MLP（多层感知机）中间隐藏层的维度
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)  # 将输入进行自注意力计算
        self.mlp = MLP(embedding_dim, mlp_dim)  # 多层感知机，其中embedding_dim是输入特征的维度，mlp_dim是MLP中间隐藏层的维度。

        self.layer_norm1 = nn.LayerNorm(embedding_dim)  # 创建了一个层归一化模块，其中embedding_dim是输入特征的维度
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 通过多头注意力机制计算注意力，然后将结果与输入特征进行加和操作，并进行丢弃层和层归一化操作
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        # 通过MLP进行非线性变换，再次将结果与之前的输出进行加和，并进行层归一化操作
        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):  # 编码器部分
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):  # block_num是编码器中包含的编码器块的数量，默认为12
        super().__init__()

        # 遍历12个编码器
        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=True, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim  # 用于存储图像块的大小
        self.classification = classification  # 判断是否分类
        self.num_tokens = (img_dim // patch_dim) ** 2  # 计算图像块数量，用于输入到transformer编码器中
        self.token_dim = in_channels * (patch_dim ** 2)  # 计算每个图像块的大小，输入图像的通道数乘以图像块的面积

        self.projection = nn.Linear(self.token_dim, embedding_dim)  # 将图像快嵌入到指定维度中

        # 可学习的参数，表示位置嵌入矩阵，用于表示输入序列中的位置信息
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))  # 可学习的参数，表示类别标记的嵌入向量

        self.dropout = nn.Dropout(0.1)  # 防止过拟合

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

        # 如果classification为True，则创建一个线性层self.mlp_head，用于将Transformer输出映射到类别数量的维度
        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # print(x.shape)
        # 将输入图像分割成图像块，并重新排列维度，以便将其输入到线性投影层
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)
        # print(img_patches.shape)
        batch_size, tokens, _ = img_patches.shape  # 获取图像块的批量大小、token 数量以及每个 token 的维度

        project = self.projection(img_patches)
        # print("img_patches.shape:", img_patches.shape)
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size)
        # print("token: ", token.shape)
        patches = torch.cat([token, project], dim=1)  # 将类别标记和投影后的图像块拼接在一起，形成完整的输入序列
        # print("patches.shape: ", patches.shape)
        # print("self.embedding: ", self.embedding.shape)
        # print("self.embedding[:tokens + 1, :]: ", self.embedding[:tokens + 1, :].shape)
        patches += self.embedding[:tokens + 1, :]  # 将位置嵌入矩阵加到输入序列中，以表示每个 token 的位置信息

        x = self.dropout(patches)
        x = self.transformer(x)

        # 如果用于分类任务，则将 Transformer 的输出通过线性层进行分类；否则，直接返回 Transformer 的输出
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        return x


if __name__ == '__main__':
    vit = ViT(img_dim=512,  #输入图像的维度
              in_channels=1,  #输入图像的通道数
              patch_dim=16,  #图像分割成块的块的大小
              embedding_dim=512,  #嵌入向量的维度，编码器的输出维度
              block_num=6,  #transformer中编码器数量
              head_num=4,  #多头注意力的头的数量
              mlp_dim=1024)  #隐藏层的维度
    print(sum(p.numel() for p in vit.parameters()))
    print(vit(torch.rand(1, 1, 640, 640)).shape)
