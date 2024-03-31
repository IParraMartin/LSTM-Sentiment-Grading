import torch.nn as nn
import torch
from torchsummary import summary

import torch.nn as nn


class LSTMSentiment(nn.Module):
    def __init__(self, classes, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.lstm1 = nn.LSTM(embedding_size, 64, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True, dropout=0.2, num_layers=2)
        self.lstm3 = nn.LSTM(128, 256, batch_first=True, num_layers=1)
        self.lstm4 = nn.LSTM(256, 128, batch_first=True, dropout=0.2, num_layers=2)
        self.lstm5 = nn.LSTM(128, 64, batch_first=True, num_layers=1)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(64, classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out, _ = self.lstm3(lstm_out)
        lstm_out, _ = self.lstm4(lstm_out)
        lstm_out, _ = self.lstm5(lstm_out)
        lstm_out = lstm_out[:, -1, :]           # Use the output corresponding to the last input time step
        out = self.dropout(lstm_out)
        logits = self.fc(out).squeeze()
        return logits

if __name__ == "__main__":
    model = LSTMSentiment(1, 10000, 300)