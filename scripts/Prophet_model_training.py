import pandas as pd
from sklearn.model_selection import train_test_split
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load the data
df = pd.read_csv("./data/final_data_with_features.csv")

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare dataframe for Prophet model
prophet_train_df = train_df[["Date", "SO_y"]].rename(columns={"Date": "ds", "SO_y": "y"})

# Initialize and fit Prophet model
model = Prophet(daily_seasonality=True)
model.fit(prophet_train_df)

# Prepare dataframe for predictions
prophet_test_df = test_df[["Date", "SO_y", "Player", "SO_Prop"]].rename(columns={"Date": "ds", "SO_y": "y"})

# Generate future datestamps for prediction
future = prophet_test_df[["ds"]]

# Predict
forecast = model.predict(future)

# Convert 'ds' index in 'prophet_test_df' back to datetime
prophet_test_df.index = pd.to_datetime(prophet_test_df.index)

# Join predictions with test data
forecast = forecast.join(prophet_test_df[["Player", "y", "SO_Prop"]])

# Reset index before saving to CSV
predictions = forecast.reset_index()[["ds", "Player", "y", "SO_Prop", "yhat"]]

# Save predictions to a CSV file
predictions.to_csv("./data/prophet_predictions.csv", index=False)
print("Predictions saved to ./data/prophet_predictions.csv")

# Evaluate model's performance
y_true = prophet_test_df["y"].values
y_pred = forecast["yhat"][-len(prophet_test_df) :].values

mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
print(f"Root Mean Squared Error (RMSE): {rmse}")
