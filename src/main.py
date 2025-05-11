import torch
from transformers import pipeline
import pandas as pd

def main():
    classifier = pipeline(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        torch_dtype=torch.float16,
        device=0
    )

    result = classifier("I love using Hugging Face Transformers!")
    print(result)

if __name__ == "__main__":
    splits = {'train_r1': 'plain_text/train_r1-00000-of-00001.parquet', 'dev_r1': 'plain_text/dev_r1-00000-of-00001.parquet', 'test_r1': 'plain_text/test_r1-00000-of-00001.parquet', 'train_r2': 'plain_text/train_r2-00000-of-00001.parquet', 'dev_r2': 'plain_text/dev_r2-00000-of-00001.parquet', 'test_r2': 'plain_text/test_r2-00000-of-00001.parquet', 'train_r3': 'plain_text/train_r3-00000-of-00001.parquet', 'dev_r3': 'plain_text/dev_r3-00000-of-00001.parquet', 'test_r3': 'plain_text/test_r3-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/facebook/anli/" + splits["train_r1"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(f"Device: {device}")
    main()