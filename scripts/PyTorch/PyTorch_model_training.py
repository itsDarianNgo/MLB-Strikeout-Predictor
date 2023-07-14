import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import numpy as np

# Load and preprocess data
df = pd.read_csv("/mnt/data/final_data_with_features.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)

# Split data into training, validation, and test sets
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

# Train is the first 70% of the dataset
train_data = df[: int(train_ratio * len(df))].copy()

# Valid is the next 15% of the dataset
valid_data = df[int(train_ratio * len(df)) : int((train_ratio + valid_ratio) * len(df))].copy()

# Test is the last 15% of the dataset
test_data = df[int((train_ratio + valid_ratio) * len(df)) :].copy()

# Convert pandas DataFrame to PyTorch tensor
train_data_tensor = torch.tensor(train_data.values, dtype=torch.float32)
valid_data_tensor = torch.tensor(valid_data.values, dtype=torch.float32)
test_data_tensor = torch.tensor(test_data.values, dtype=torch.float32)


# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


# Hyperparameters grid
param_grid = {
    "lr": [0.01, 0.001, 0.0001],
    "hidden_dim": [32, 64, 128],
    "layer_dim": [1, 2, 3],
    "batch_size": [16, 32, 64],
    "n_epochs": [10, 20, 50],
}

# Grid search
for params in ParameterGrid(param_grid):
    model = LSTM(input_dim=train_data_tensor.shape[1], hidden_dim=params["hidden_dim"], layer_dim=params["layer_dim"], output_dim=1)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)

    train_losses = []
    valid_losses = []

    # Training loop
    for epoch in range(params["n_epochs"]):
        model.train()
        optimizer.zero_grad()
        train_predictions = model(train_data_tensor)
        train_loss = criterion(train_predictions, train_data_tensor)
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_predictions = model(valid_data_tensor)
            valid_loss = criterion(valid_predictions, valid_data_tensor)
            valid_losses.append(valid_loss.item())

        scheduler.step(valid_loss)

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(test_data_tensor)
    test_loss = mean_squared_error(test_data_tensor.numpy(), test_predictions.numpy())

    print(f"Params: {params}, Test loss: {test_loss}")

# Save the model
torch.save(model.state_dict(), "model.pth")
