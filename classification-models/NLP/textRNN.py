import torch
import torch.nn as nn

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_classes):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)

        rnn_out, hidden = self.rnn(x)

        x = self.dropout(rnn_out[:, -1, :])
        x = self.fc(x)
        return x

if __name__ == '__main__':
    vocab_size = 5000  # 词汇表大小
    embedding_dim = 300  # 词向量维度
    hidden_dim = 256  # 隐藏层维度
    num_layers = 2  # RNN层数
    dropout = 0.5  # dropout概率
    num_classes = 2  # 分类类别数

    model = TextRNN(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_classes)
    input_data = torch.randint(0, vocab_size, (32, 50))  # 生成随机输入数据 (batch_size, seq_length)
    output = model(input_data)
    print(output.shape)  # 输出形状应该是 (batch_size, num_classes)
