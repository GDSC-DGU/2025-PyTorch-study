import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, filepath):
        df = pd.read_csv(filepath)
        self.x = df.iloc[:, 0].values
        self.y = df.iloc[:, 1].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index] ** 2, self.x[index]])
        y = torch.FloatTensor(self.y[index])
        return x, y

    def __len__(self):
        return self.length


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

train_dataset = CustomDataset(filepath)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

# GPU 가속
device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"

model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(10000):
    cost = 0.0
    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    cost = cost / len(train_dataloader)
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch: {epoch + 1:4d}, Model: {list(model.parameters())}, Cost: {cost:.3f}")

