import torch
import torch.nn as nn
import math


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, dropout, num_classes):
        super(TextTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(x.size(-1))
        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        x = self.dropout(x.mean(dim=1))  # Average pooling over sequence length

        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == '__main__':
    vocab_size = 5000  # 词汇表大小
    embedding_dim = 512  # 词向量维度
    num_heads = 8  # 注意力头数
    num_layers = 6  # Transformer层数
    hidden_dim = 2048  # 隐藏层维度
    dropout = 0.1  # dropout概率
    num_classes = 2  # 分类类别数

    model = TextTransformer(vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, dropout, num_classes)
    input_data = torch.randint(0, vocab_size, (32, 50))  # 生成随机输入数据 (batch_size, seq_length)
    output = model(input_data)
    print(output.shape)  # 输出形状应该是 (batch_size, num_classes)
