import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, dropout, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv2d(1, num_filters, (kernel_sizes[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, num_filters, (kernel_sizes[1], embedding_dim))
        self.conv3 = nn.Conv2d(1, num_filters, (kernel_sizes[2], embedding_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # 增加通道维度方便卷积处理

        conv1_out = F.relu(self.conv1(x)).squeeze(3)
        pooled1 = F.max_pool1d(conv1_out, conv1_out.size(2)).squeeze(2)

        conv2_out = F.relu(self.conv2(x)).squeeze(3)
        pooled2 = F.max_pool1d(conv2_out, conv2_out.size(2)).squeeze(2)

        conv3_out = F.relu(self.conv3(x)).squeeze(3)
        pooled3 = F.max_pool1d(conv3_out, conv3_out.size(2)).squeeze(2)

        x = torch.cat((pooled1, pooled2, pooled3), 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    vocab_size = 5000  # 词汇表大小
    embedding_dim = 300  # 词向量维度
    num_filters = 100  # 每个卷积核的输出通道数
    kernel_sizes = [3, 4, 5]  # 卷积核大小
    dropout = 0.5  # dropout概率
    num_classes = 2  # 分类类别数

    model = TextCNN(vocab_size, embedding_dim, num_filters, kernel_sizes, dropout, num_classes)
    input_data = torch.randint(0, vocab_size, (32, 50))  # 生成随机输入数据 (batch_size, seq_length)
    output = model(input_data)
    print(output.shape)  # 输出形状应该是 (batch_size, num_classes)
