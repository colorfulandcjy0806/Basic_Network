import torch
import torch.nn as nn

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_classes):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        batch_size, seq_len, _ = x.shape
        h_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)

        x, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        x = self.dropout(h_n[-1])

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
