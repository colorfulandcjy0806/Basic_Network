import torch
import torch.nn as nn

class CustomLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.i2f = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.i2i = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.i2c = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.i2o = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        f_t = self.sigmoid(self.i2f(combined))
        i_t = self.sigmoid(self.i2i(combined))
        c_tilde_t = self.tanh(self.i2c(combined))
        c_t = f_t * cell + i_t * c_tilde_t
        o_t = self.sigmoid(self.i2o(combined))
        h_t = o_t * self.tanh(c_t)
        return h_t, c_t


class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_classes):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm1 = CustomLSTMLayer(embedding_dim, hidden_dim)
        self.lstm2 = CustomLSTMLayer(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        batch_size, seq_len, _ = x.shape
        hidden1 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        cell1 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        hidden2 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        cell2 = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for t in range(seq_len):
            input_t = x[:, t, :]
            hidden1, cell1 = self.lstm1(input_t, hidden1, cell1)
            hidden2, cell2 = self.lstm2(hidden1, hidden2, cell2)

        x = self.dropout(hidden2)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    vocab_size = 5000  # 词汇表大小
    embedding_dim = 300  # 词向量维度
    hidden_dim = 256  # 隐藏层维度
    num_layers = 2  # LSTM层数
    dropout = 0.5  # dropout概率
    num_classes = 2  # 分类类别数

    model = TextLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_classes)
    input_data = torch.randint(0, vocab_size, (32, 50))  # 生成随机输入数据 (batch_size, seq_length)
    output = model(input_data)
    print(output.shape)  # 输出形状应该是 (batch_size, num_classes)
