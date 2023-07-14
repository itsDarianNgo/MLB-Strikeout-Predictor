import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import keras

print(tf.__version__)
print(keras.__version__)

# Load the dataset
data = pd.read_csv("./data/final_data_with_features_2023.csv")

# Convert the 'date' column to a datetime format
data["Date"] = pd.to_datetime(data["Date"])

# Create separate columns for year, month, and day
data["year"] = data["Date"].dt.year
data["month"] = data["Date"].dt.month
data["day"] = data["Date"].dt.day

# Separating the features and target variables
features = data.drop(["SO_y", "SO_Prop", "GameID", "Date"], axis=1)

# Apply one-hot encoding
features = pd.get_dummies(
    features,
    columns=["Team", "Player", "venue", "Opposing_Team", "home_away"],
)

# Replace infinity with NaN
features.replace([np.inf, -np.inf], np.nan, inplace=True)

# Replace NaN values with the mean of the column
features.fillna(features.mean(), inplace=True)

target = data["SO_y"]
betting_lines = data["SO_Prop"]

# Scaling the features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))

# Compiling the LSTM model
lstm_model.compile(optimizer="adam", loss="mean_squared_error")

# Reshape the input data to be 3D for LSTM (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Training the LSTM model
lstm_model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=1)

# Save the LSTM model
lstm_model.save("lstm_model.h5")

# Making LSTM predictions
lstm_predictions = lstm_model.predict(X_test)

# Binary classification model to predict whether the actual strikeouts will be over or under the betting line
binary_features = pd.concat([features, pd.Series(lstm_predictions.flatten(), name="predicted_SO")], axis=1)
binary_target = (target > betting_lines).astype(int)  # 1 if over, 0 if under

# Splitting the data for binary classification model
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(binary_features, binary_target, test_size=0.2, random_state=42)

# Binary classification model
binary_model = Sequential()
binary_model.add(Dense(64, activation="relu", input_dim=X_train_binary.shape[1]))
binary_model.add(Dense(32, activation="relu"))
binary_model.add(Dense(1, activation="sigmoid"))

# Compiling the binary classification model
binary_model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# Training the binary classification model
binary_model.fit(X_train_binary, y_train_binary, batch_size=32, epochs=100, verbose=1)

# Save the binary model
binary_model.save("binary_model.h5")

# Making binary predictions
binary_predictions = binary_model.predict(X_test_binary)
binary_predictions = ["over" if pred > 0.5 else "under" for pred in binary_predictions.flatten()]

# Creating a dataframe with all the relevant information
predictions_df = data.loc[y_test.index]
predictions_df["lstm_predictions"] = lstm_predictions
predictions_df["binary_predictions"] = binary_predictions

# Saving the predictions to a CSV file
predictions_df.to_csv("LSTM_predictions.csv", index=False)
