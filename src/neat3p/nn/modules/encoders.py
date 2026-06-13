import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class SimpleEncoder(nn.Module):
    def __init__(self, input_dim: int, encoded_dim: int, hidden_dim: int = 64, device: str = "cuda:0"):
        super(SimpleEncoder, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, encoded_dim))
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.encoder(x)

    def train_model(self, dataloader, epochs=10, lr=0.001):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                output = self.forward(x_batch)
                loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()

    def create_dataloader(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32) -> DataLoader:
        tensor_x = torch.tensor(x, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
