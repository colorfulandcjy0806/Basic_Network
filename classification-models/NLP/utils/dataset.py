import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class IMDBDataset(Dataset):
    def __init__(self, data_dir, vocab_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.vocab = self.load_vocab(vocab_file)

        for label_type in ['pos', 'neg']:
            dir_name = os.path.join(data_dir, label_type)
            for fname in os.listdir(dir_name):
                if fname.endswith('.txt'):
                    with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                        self.data.append(f.read())
                        self.labels.append(1 if label_type == 'pos' else 0)

    def load_vocab(self, vocab_file):
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                vocab[word] = idx + 1
        vocab['<UNK>'] = 0
        return vocab

    def text_to_sequence(self, text):
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        else:
            sample = torch.tensor(self.text_to_sequence(sample), dtype=torch.long)
        return sample, torch.tensor(label, dtype=torch.float)

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_texts, labels

def get_dataloaders(train_dir, test_dir, vocab_file, batch_size=32):
    train_dataset = IMDBDataset(train_dir, vocab_file)
    test_dataset = IMDBDataset(test_dir, vocab_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader

if __name__ == '__main__':
    train_dir = 'aclImdb/train'
    test_dir = 'aclImdb/test'
    vocab_file = 'aclImdb/imdb.vocab'
    train_loader, test_loader = get_dataloaders(train_dir, test_dir, vocab_file)

    for i, (data, labels) in enumerate(train_loader):
        print(f'Batch {i+1}:')
        print(f'Data shape: {data.shape}')
        print(f'Labels shape: {labels.shape}')
        print(f'Sample data: {data[0][:10]}')
        print(f'Sample label: {labels[0]}')
        if i == 1:
            break

    for i, (data, labels) in enumerate(test_loader):
        print(f'Test Batch {i+1}:')
        print(f'Data shape: {data.shape}')
        print(f'Labels shape: {labels.shape}')
        print(f'Sample data: {data[0][:10]}')
        print(f'Sample label: {labels[0]}')
        if i == 1:
            break
