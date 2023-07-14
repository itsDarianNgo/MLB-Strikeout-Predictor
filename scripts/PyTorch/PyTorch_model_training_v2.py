import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import copy
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device


# Define a PyTorch Dataset
class StrikeoutDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        features_df = sequence.drop(columns=["SO_y"])  # Drop 'Player' column from sequences here
        features = torch.tensor(features_df.values[:-1], dtype=torch.float32).to(device)  # Move features to device
        target = torch.tensor(sequence["SO_y"].values[-1], dtype=torch.float32).to(device)  # Move target to device
        return features, target


# Function to create sequences
def create_sequences(df, window_size):
    sequence_list = []
    for player in df["Player"].unique():
        player_df = df[df["Player"] == player]
        for i in range(window_size, len(player_df)):
            sequence = player_df.iloc[i - window_size : i + 1].copy()
            sequence_list.append(sequence)
    return sequence_list


# Define the LSTM model architecture
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(-1)


# Load data
df = pd.read_csv("./data/final_data_with_features.csv")

# Load unseen data
unseen_df = pd.read_csv("./data/final_data_with_features_2023.csv")

# Combine the player names from the training data and unseen data
all_players = pd.concat([df['Player'], unseen_df['Player']]).unique()

# Split the 'Date' column into 'Year', 'Month', and 'Day' columns
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Sort values
df.sort_values("Date", inplace=True)

# Preserve the 'Player' column before one-hot encoding
players = df["Player"]

# Define a one-hot encoder object for the 'Player' feature with all players as categories
player_encoder = OneHotEncoder(categories=[all_players], sparse=False)

# Apply one hot encoding to the 'Player' column
player_encoded_columns = player_encoder.fit_transform(df[["Player"]])
player_encoded_df = pd.DataFrame(player_encoded_columns, columns=player_encoder.get_feature_names_out(["Player"]))

# Define a one-hot encoder object for the other categorical features
other_encoder = OneHotEncoder(sparse=False)

# Apply one hot encoding to the other categorical columns
other_encoded_columns = other_encoder.fit_transform(df[["Team", "venue", "Opposing_Team", "home_away"]])
other_encoded_df = pd.DataFrame(other_encoded_columns, columns=other_encoder.get_feature_names_out(["Team", "venue", "Opposing_Team", "home_away"]))

# Drop the original 'Date', 'Team', 'Player', 'venue', 'Opposing_Team', 'home_away' columns
df = df.drop(columns=["Date", "Team", "Player", "venue", "Opposing_Team", "home_away", "GameID"])

# Concatenate the original dataframe with the one-hot encoded dataframes
df = pd.concat([df, player_encoded_df, other_encoded_df], axis=1)

# Create a separate DataFrame for the purpose of creating sequences
df_with_players = df.copy()
df_with_players["Player"] = players

# Create sequences using df_with_players and drop 'Player' column from each sequence
window_size = 10
sequence_list = create_sequences(df_with_players, window_size)
for sequence in sequence_list:
    sequence.drop(columns=["Player"], inplace=True)

# Split data into train/valid/test sets
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15
train_end = int(train_ratio * len(sequence_list))
valid_end = int((train_ratio + valid_ratio) * len(sequence_list))
train_sequences = sequence_list[:train_end]
valid_sequences = sequence_list[train_end:valid_end]
test_sequences = sequence_list[valid_end:]

# Convert sequence list to tensors
train_dataset = StrikeoutDataset(train_sequences)
valid_dataset = StrikeoutDataset(valid_sequences)
test_dataset = StrikeoutDataset(test_sequences)

# Create dataloaders
batch_size = 1  # Since we are now working with sequences, it's common to use a batch size of 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Set baseline model parameters
input_size = train_sequences[0].shape[1] - 1  # Minus 1 because we removed the 'SO_y' column
hidden_size = 50
num_layers = 1
output_size = 1
learning_rate = 0.001
num_epochs = 10

# Define baseline model
model = LSTM(input_size, hidden_size, num_layers, output_size)
model = model.to(device)  # Move model to device
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the baseline model
for epoch in range(num_epochs):
    loop = tqdm(train_loader)
    for i, (features, targets) in enumerate(loop):
        features = features.to(device)
        targets = targets.to(device)
        model.train()
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

# Validate the model
model.eval()
valid_losses = []
for i, (features, targets) in enumerate(valid_loader):
    features = features.to(device)
    targets = targets.to(device)
    outputs = model(features)
    loss = criterion(outputs, targets)
    valid_losses.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {np.mean(valid_losses):.4f}")

# Optimize model architecture
hidden_size = 100
num_layers = 2
model = LSTM(input_size, hidden_size, num_layers, output_size)
model = model.to(device)  # Move model to device
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Perform hyperparameter tuning
# We use grid search to find the best learning rate and number of epochs.
param_grid = {"lr": [0.1], "num_epochs": [5]}
# param_grid = {"lr": [0.1, 0.01, 0.001], "num_epochs": [10, 20, 30]}
best_params = None
best_loss = np.inf
for params in ParameterGrid(param_grid):
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    for epoch in range(params["num_epochs"]):
        for i, (features, targets) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()
        valid_losses = []
        for i, (features, targets) in enumerate(valid_loader):
            outputs = model(features)
            loss = criterion(outputs, targets)
            valid_losses.append(loss.item())

        avg_valid_loss = np.mean(valid_losses)
        print(f'lr={params["lr"]}, num_epochs={params["num_epochs"]}, Epoch {epoch+1}/{params["num_epochs"]}, Validation Loss: {avg_valid_loss:.4f}')

        # Check if this is the best model
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            best_params = params
            best_model = copy.deepcopy(model.state_dict())

# Train the final model with the best hyperparameters
model.load_state_dict(best_model)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
for epoch in range(best_params["num_epochs"]):
    for i, (features, targets) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validate the model
    model.eval()
    valid_losses = []
    for i, (features, targets) in enumerate(valid_loader):
        outputs = model(features)
        loss = criterion(outputs, targets)
        valid_losses.append(loss.item())

    print(
        f'Final model lr={best_params["lr"]}, num_epochs={best_params["num_epochs"]}, Epoch {epoch+1}/{best_params["num_epochs"]}, Validation Loss: {np.mean(valid_losses):.4f}'
    )

# Evaluate the final model performance on the test set
test_losses = []
for i, (features, targets) in enumerate(test_loader):
    features = features.to(device)  # Move features to the same device as the model
    targets = targets.to(device)  # Move targets to the same device as the model
    outputs = model(features)
    loss = criterion(outputs, targets)
    test_losses.append(loss.item())

print(f"Final Model Test Loss: {np.mean(test_losses):.4f}")

# Save the final model
model = model.to("cpu")  # Ensure the model is on CPU before saving
torch.save(model.state_dict(), "./data/PyTorch_final_model.pth")