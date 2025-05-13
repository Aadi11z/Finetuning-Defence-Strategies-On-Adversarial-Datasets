import torch
import torch.nn as nn
import pandas as pd 
from functions import GaussianNoise, tSigmoid, BReLU
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from collections import Counter
import re

class RobustTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, t=0.15, noise_std=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.noise = GaussianNoise(noise_std)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.activation = BReLU(t)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        emb = self.embedding(x).mean(dim=1)  # simple mean pooling
        emb = self.noise(emb)  # add Gaussian noise during training
        x = self.fc1(emb)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in self.texts[idx]]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
    
def tokenize(text):
    return word_tokenize(text)

def main(train_df, test_df):
    print(torch.device.type)
    
    train_df['tokens'] = train_df['hypothesis'].apply(tokenize)
    test_df['tokens'] = test_df['hypothesis'].apply(tokenize)

    label_encoder = LabelEncoder()
    train_df['label_enc'] = label_encoder.fit_transform(train_df['label'])
    test_df['label_enc'] = label_encoder.transform(test_df['label'])

    counter = Counter(tok for tokens in train_df['tokens'] for tok in tokens)
    vocab = {word: i + 2 for i, (word, _) in enumerate(counter.items())}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    train_dataset = TextDataset(train_df['tokens'].tolist(), train_df['label_enc'].tolist(), vocab)
    val_dataset = TextDataset(test_df['tokens'].tolist(), test_df['label_enc'].tolist(), vocab)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)

    num_classes = len(label_encoder.classes_)
    model = RobustTextClassifier(len(vocab), 100, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = [x.to(device) for x in batch]
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        print(f"Validation Accuracy: {correct / total:.4f}")


def collate_fn(batch):
    inputs, labels = zip(*batch)
    lengths = [len(seq) for seq in inputs]
    max_len = max(lengths)
    padded_inputs = [F.pad(seq, (0, max_len - len(seq)), value=0) for seq in inputs]
    return torch.stack(padded_inputs), torch.tensor(labels)


if __name__ == "__main__":
    splits = {
        'train_r1': 'anli/plain_text/train_r1-00000-of-00001.parquet', 
        'dev_r1': 'anli/plain_text/dev_r1-00000-of-00001.parquet', 
        'test_r1': 'anli/plain_text/test_r1-00000-of-00001.parquet', 
        'train_r2': 'anli/plain_text/train_r2-00000-of-00001.parquet', 
        'dev_r2': 'anli/plain_text/dev_r2-00000-of-00001.parquet', 
        'test_r2': 'anli/plain_text/test_r2-00000-of-00001.parquet', 
        'train_r3': 'anli/plain_text/train_r3-00000-of-00001.parquet', 
        'dev_r3': 'anli/plain_text/dev_r3-00000-of-00001.parquet', 
        'test_r3': 'anli/plain_text/test_r3-00000-of-00001.parquet'
    }
    train_df = pd.read_parquet(splits["train_r1"])
    test_df = pd.read_parquet(splits['test_r1'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    main(train_df, test_df)