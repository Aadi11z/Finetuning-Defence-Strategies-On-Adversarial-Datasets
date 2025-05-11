import torch
import torch.nn as nn
import pandas as pd 
from functions import GaussianNoise, BReLU
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import re

class RobustTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, t=6.0, noise_std=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.noise = GaussianNoise(noise_std)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.brelu = BReLU(t)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        emb = self.embedding(x).mean(dim=1)  # simple mean pooling
        emb = self.noise(emb)  # add Gaussian noise during training
        x = self.fc1(emb)
        x = self.brelu(x)
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
        return re.findall(r"\b\w+\b", text.lower())

def main():
    # 1. Load and preprocess data
    df['tokens'] = df['hypothesis'].apply(tokenize)  # assuming you use 'hypothesis' column
    label_encoder = LabelEncoder()
    df['label_enc'] = label_encoder.fit_transform(df['label'])

    # 2. Build vocabulary
    counter = Counter(tok for tokens in df['tokens'] for tok in tokens)
    vocab = {word: i+2 for i, (word, _) in enumerate(counter.items())}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    # 3. Create dataset and dataloaders
    X_train, X_val, y_train, y_val = train_test_split(df['tokens'], df['label_enc'], test_size=0.1, random_state=42)
    train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), vocab)
    val_dataset = TextDataset(X_val.tolist(), y_val.tolist(), vocab)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)

    # 4. Model setup
    model = RobustTextClassifier(len(vocab), 100, len(label_encoder.classes_)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 5. Training loop
    for epoch in range(50):  # or however many epochs you want
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

        # Optional: validation
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
        'train_r1': 'plain_text/train_r1-00000-of-00001.parquet', 
        'dev_r1': 'plain_text/dev_r1-00000-of-00001.parquet', 
        'test_r1': 'plain_text/test_r1-00000-of-00001.parquet', 
        'train_r2': 'plain_text/train_r2-00000-of-00001.parquet', 
        'dev_r2': 'plain_text/dev_r2-00000-of-00001.parquet', 
        'test_r2': 'plain_text/test_r2-00000-of-00001.parquet', 
        'train_r3': 'plain_text/train_r3-00000-of-00001.parquet', 
        'dev_r3': 'plain_text/dev_r3-00000-of-00001.parquet', 
        'test_r3': 'plain_text/test_r3-00000-of-00001.parquet'
    }
    df = pd.read_parquet("hf://datasets/facebook/anli/" + splits["train_r1"])
    df.describe()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    main()