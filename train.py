from data import FGDataset
from model import LSTMSentiment
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

BATCH_SIZE = 128
EPOCHS = 1
DATA_PATH = "Family_Guy_Final_NRC_AFINN_BING.csv"
LEARNING_RATE = 0.001

def get_dataloader(data, batch_size):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_train_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        y_hat = model(inputs).squeeze()
        train_loss = criterion(y_hat, targets.float())
        total_train_loss += train_loss

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    return total_train_loss / len(dataloader)

def train(epochs, model, dataloader, criterion, optimizer, device):
    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        print(f"{epoch+1} / {epochs}")
        train_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Train L: {train_loss.item()}")
        print("----------------------")
    print(f"Training Complete")

if __name__ == "__main__":
    dataset = FGDataset(DATA_PATH, vocab=None, max_length=50)
    train_dataloader = get_dataloader(dataset, batch_size=BATCH_SIZE)
    model = LSTMSentiment(classes=1, vocab_size=75000, embedding_size=300)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(
        epochs=1,
        model=model,
        dataloader=train_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )