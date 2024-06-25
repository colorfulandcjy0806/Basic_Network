import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(keys).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)

        out = self.fc_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_dim, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, dropout, num_classes):
        super(TextTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(x.size(-1))
        x = self.pos_encoding(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, x, x, mask)

        x = self.dropout(x.mean(dim=1))  # Average pooling over sequence length

        x = self.fc(x)
        return x

if __name__ == '__main__':
    vocab_size = 5000  # 词汇表大小
    embed_size = 512  # 词向量维度
    num_heads = 8  # 注意力头数
    num_layers = 6  # Transformer层数
    hidden_dim = 2048  # 隐藏层维度
    dropout = 0.1  # dropout概率
    num_classes = 2  # 分类类别数

    model = TextTransformer(vocab_size, embed_size, num_heads, num_layers, hidden_dim, dropout, num_classes)
    input_data = torch.randint(0, vocab_size, (32, 50))  # 生成随机输入数据 (batch_size, seq_length)
    output = model(input_data)
    print(output.shape)  # 输出形状应该是 (batch_size, num_classes)
