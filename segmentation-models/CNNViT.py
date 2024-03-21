import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, C, N = x.size()
        print(x.shape)
        Q = self.queries(x)
        K = self.keys(x)
        V = self.values(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention, V)
        return out

class CNNWithAttention(nn.Module):
    def __init__(self):
        super(CNNWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Adjust embed_size for 14x14 input, after pooling it becomes 7x7
        self.attention1 = SelfAttention(embed_size=196)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Adjust the input size of the fully connected layer accordingly
        self.fc = nn.Linear(16*7*7, 10)  # Adjusted for 7x7 feature map size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), x.size(1), -1)
        x = self.attention1(x)
        # Adjust the reshape dimensions according to the new pooled feature map size
        x = x.view(x.size(0), 16, 14, 14)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    input_img = torch.randn(1, 3, 14, 14)
    model = CNNWithAttention()
    output = model(input_img)
    print(output.shape)
