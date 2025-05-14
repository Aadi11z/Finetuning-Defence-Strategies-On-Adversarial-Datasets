from infobert import *
from freelb import *
from ranmask import *
import sys

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
def tokenize(text):
    return word_tokenize(text)

def main(train_df, test_df):
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("kweinmeister/distilbert-anli")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_encoder = LabelEncoder()
    train_df['label_enc'] = label_encoder.fit_transform(train_df['label'])
    test_df['label_enc'] = label_encoder.transform(test_df['label'])

    train_texts = train_df['hypothesis'].tolist()
    train_labels = train_df['label_enc'].tolist()
    test_texts = test_df['hypothesis'].tolist()
    test_labels = test_df['label_enc'].tolist()

    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    num_classes = len(label_encoder.classes_)

    if sys.argv[1] == "1": 
        print("Running INFOBert")
        model = InfoBERTModel(num_labels=num_classes, t=6)
    elif sys.argv[1] == "2":
        print("Running FreeLB")
        model = FreeLBModel(num_classes, t=6)
    elif sys.argv[1] == "3":
        print("Running RanMASK")
        model = RanMASKModel(num_classes, t=6)
    else:
        model = AutoModelForSequenceClassification.from_pretrained("kweinmeister/distilbert-anli")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
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
    main(train_df, test_df)