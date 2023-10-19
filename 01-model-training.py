import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Constants
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
PATIENCE = 3

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False

    def step(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    valid_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, valid_loader

def main():
    train_loader, valid_loader = load_data()

    model = SimpleMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping = EarlyStopping(patience=PATIENCE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data.view(-1, INPUT_SIZE))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for data, target in valid_loader:
                output = model(data.view(-1, INPUT_SIZE))
                loss = criterion(output, target)
                validation_loss += loss.item()

        print(f'Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {validation_loss/len(valid_loader)}')
        early_stopping.step(validation_loss, model)
        if early_stopping.stop:
            print('Early stopping')
            break

if __name__ == '__main__':
    main()
