import torch
import torch.nn as nn

class CustomRNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomRNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.i2h = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.tanh(self.i2h(combined))
        output = self.h2o(hidden)
        return output, hidden


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_classes):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn1 = CustomRNNLayer(embedding_dim, hidden_dim)
        self.rnn2 = CustomRNNLayer(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        batch_size, seq_len, _ = x.shape
        hidden1 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        hidden2 = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for t in range(seq_len):
            input_t = x[:, t, :]
            hidden1, _ = self.rnn1(input_t, hidden1)
            hidden2, _ = self.rnn2(hidden1, hidden2)

        x = self.dropout(hidden2)
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
