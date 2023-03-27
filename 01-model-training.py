import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import datasets, transforms

class EarlyStopping:
    def __init__(self, patience, patience_increase, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.patience_increase = patience_increase
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def save_checkpoint(val_loss, model, early_stopping):
    """Saves model when validation loss decrease."""
    if early_stopping.verbose:
        print(f'Validation loss decreased ({early_stopping.val_loss_min:.6f} --> {val_loss:.6f}).  Saving checkpoint ...')
    torch.save(model.state_dict(), early_stopping.path)
    early_stopping.val_loss_min = val_loss

def early_stopping_step(val_loss, model, early_stopping):
    score = -val_loss

    if early_stopping.best_score is None:
        early_stopping.best_score = score
        save_checkpoint(val_loss, model, early_stopping)
    elif score < early_stopping.best_score:
        early_stopping.counter += 1
        if early_stopping.counter >= early_stopping.patience:
            if early_stopping.verbose:
                print(f'Validation loss did not improve enough after {early_stopping.patience} epochs. '
                f'Stopping the training.')
            early_stopping.early_stop = True
    else:
        early_stopping.best_score = score
        early_stopping.patience = early_stopping.patience + early_stopping.patience_increase
        save_checkpoint(val_loss, model, early_stopping)
        early_stopping.counter = 0
    
    return early_stopping

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('Epoch: {}\tTraining Loss: {:.6f}'.format(epoch, train_loss))

    return train_loss

def validate(model, device, val_loader, criterion, epoch, patience, early_stopping):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()

        val_loss /= len(val_loader)
        print(f'Validation loss: {val_loss:.4f}')

        if early_stopping is not None:
            early_stopping_step(val_loss, model, early_stopping)

    return val_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * accuracy))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Split the train set into train and validation sets.
    train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    patience = 10
    patience_increase=5
    num_epochs = 100
    early_stopping = EarlyStopping(patience, patience_increase, verbose=True)

    for epoch in range(1, num_epochs + 1):
      train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
      val_loss = validate(model, device, val_loader, criterion, epoch, patience, early_stopping)
      early_stopping = early_stopping_step(val_loss, model, early_stopping)

      if early_stopping.early_stop:
          break

    # Load the best checkpoint.
    model.load_state_dict(torch.load('checkpoint.pt'))

    # Test the model on the test set.
    test(model, device, test_loader)

if __name__ == '__main__':
    main()

