import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime


# Define a PyTorch Dataset
class StrikeoutDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        features_df = sequence.drop(columns=["SO_y"])  # Drop 'Player' column from sequences here
        features = torch.tensor(features_df.values[:-1], dtype=torch.float32)  # Move features to device
        target = torch.tensor(sequence["SO_y"].values[-1], dtype=torch.float32)  # Move target to device
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(-1)


# Load the unseen data
unseen_data = pd.read_csv("./data/final_data_with_features_2023.csv")

# Preprocess the unseen data similar to the training data
unseen_data["Date"] = pd.to_datetime(unseen_data["Date"])
unseen_data["Year"] = unseen_data["Date"].dt.year
unseen_data["Month"] = unseen_data["Date"].dt.month
unseen_data["Day"] = unseen_data["Date"].dt.day

players_unseen = unseen_data["Player"]

# Initialize an one hot encoder object
encoder = OneHotEncoder(sparse=False)

# Apply one hot encoding to the specified columns
encoded_columns_unseen = encoder.fit_transform(unseen_data[["Team", "Player", "venue", "Opposing_Team", "home_away"]])
encoded_df_unseen = pd.DataFrame(encoded_columns_unseen, columns=encoder.get_feature_names_out(["Team", "Player", "venue", "Opposing_Team", "home_away"]))

# Drop the original 'Date', 'Team', 'Player', 'venue', 'Opposing_Team', 'home_away' columns
unseen_data = unseen_data.drop(columns=["Date", "Team", "Player", "venue", "Opposing_Team", "home_away", "GameID"])

# Concatenate the original dataframe with the one-hot encoded dataframe
unseen_data = pd.concat([unseen_data, encoded_df_unseen], axis=1)

# Create a separate DataFrame for the purpose of creating sequences
unseen_data_with_players = unseen_data.copy()
unseen_data_with_players["Player"] = players_unseen

# Create sequences using unseen_data_with_players and drop 'Player' column from each sequence
window_size = 10
sequence_list_unseen = create_sequences(unseen_data_with_players, window_size)
for sequence in sequence_list_unseen:
    sequence.drop(columns=["Player"], inplace=True)

# Convert sequence list to tensors
unseen_dataset = StrikeoutDataset(sequence_list_unseen)

# Create dataloaders
batch_size = 1
unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size)

# Load the trained model
input_size = sequence_list_unseen[0].shape[1] - 1  # Minus 1 because we removed the 'SO_y' column
hidden_size = 100
num_layers = 2
output_size = 1

model = LSTM(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load("./data/PyTorch_final_model.pth"))

# Make predictions on unseen data
model.eval()
predictions = []
actual_values = []
for i, (features, targets) in enumerate(unseen_loader):
    outputs = model(features)
    predictions.extend(outputs.detach().numpy())
    actual_values.extend(targets.detach().numpy())

# Compute evaluation metrics
mae = mean_absolute_error(actual_values, predictions)
mse = mean_squared_error(actual_values, predictions)
r2 = r2_score(actual_values, predictions)

# Prepare the unseen data with predictions
unseen_data_with_predictions = unseen_data_with_players.copy()  # Create a copy of the original DataFrame

# Create a new DataFrame to store the final results
final_predictions_df = pd.DataFrame(columns=unseen_data_with_predictions.columns.tolist() + ["Predicted_SO"])

# For each player, get the corresponding predictions and actual data
start_idx = 0
for player in unseen_data_with_players["Player"].unique():
    player_data = unseen_data_with_predictions[unseen_data_with_predictions["Player"] == player]
    
    # Only create sequences for players with at least 'window_size' games
    if len(player_data) >= window_size:
        player_predictions = predictions[start_idx : start_idx + len(player_data) - window_size + 1]
        player_data = player_data.iloc[window_size - 1:]  # Skip the first 'window_size - 1' games for each player

        # Ensure the lengths of player_predictions and player_data are the same before assignment
        if len(player_predictions) == len(player_data):
            player_data["Predicted_SO"] = player_predictions
            final_predictions_df = pd.concat([final_predictions_df, player_data])
        
        start_idx += len(player_data)

# Keep only the necessary columns
final_predictions_df = final_predictions_df[["Date", "Player", "SO_y", "SO_Prop", "Predicted_SO"]]

# Save the DataFrame with predictions to a CSV file
final_predictions_df.to_csv("./data/PyTorch_predictions.csv", index=False)


print(unseen_data_with_predictions)
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R^2: {r2}")
