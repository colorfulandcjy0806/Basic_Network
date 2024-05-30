import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import get_dataloaders
from model import TextCNN

def train_model(model, train_loader, test_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                predictions = model(data)
                loss = criterion(predictions, labels.long())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.update(1)
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch + 1}, Loss: {avg_loss}')

        evaluate_model(model, test_loader, device)


def evaluate_model(model, test_loader, device):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            predictions = model(data)
            predicted_labels = torch.argmax(predictions, dim=1)
            total_correct += (predicted_labels == labels).sum().item()

    accuracy = total_correct / len(test_loader.dataset)
    print(f'Accuracy: {accuracy}')


if __name__ == "__main__":
    train_dir = 'aclImdb/train'
    test_dir = 'aclImdb/test'
    vocab_file = 'aclImdb/imdb.vocab'
    subset_ratio = 0.1  # 使用 10% 的数据
    train_loader, test_loader = get_dataloaders(train_dir, test_dir, vocab_file, subset_ratio=subset_ratio)
    vocab_size = len(open(vocab_file, 'r').readlines()) + 1  # 包括 <UNK> 标记
    embedding_dim = 100
    num_classes = 2
    kernel_sizes = [3, 4, 5]
    num_filters = 100
    dropout = 0.5
    epochs = 10
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextCNN(vocab_size, embedding_dim, num_classes, kernel_sizes, num_filters, dropout)
    train_model(model, train_loader, test_loader, epochs, lr, device)
