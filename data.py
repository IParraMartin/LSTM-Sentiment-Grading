import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from collections import Counter

class FGDataset(Dataset):
    def __init__(self, path, vocab=None, max_length=None):
        data = pd.read_csv(path)
        self.x = data.iloc[:, 0].tolist()
        self.y = data.iloc[:, -2].tolist()
        self.max_length = max_length

        if vocab is None:
            self.vocab = self.create_vocab(self.x)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        text_tensor = self.text_to_tensor(self.x[idx])
        if self.max_length is not None:
            if len(text_tensor) < self.max_length:
                padding = torch.zeros(self.max_length - len(text_tensor), dtype=torch.long)
                text_tensor = torch.cat((text_tensor, padding), dim=0)
            else:
                text_tensor = text_tensor[:self.max_length]

        label_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return text_tensor, label_tensor

    @staticmethod
    def tokenize(text):
        if isinstance(text, str):
            return text.split()
        else:
            return []

    def create_vocab(self, texts):
        tokens = [token for text in texts for token in self.tokenize(text)]
        token_counts = Counter(tokens)
        return {token: index for index, (token, _) in enumerate(token_counts.items(), start=1)}

    def text_to_tensor(self, text):
        return torch.tensor([self.vocab[token] for token in self.tokenize(text) if token in self.vocab], dtype=torch.long)

if __name__ == "__main__":
    data = FGDataset("Family_Guy_Final_NRC_AFINN_BING.csv", max_length=50)
    print(data[45])