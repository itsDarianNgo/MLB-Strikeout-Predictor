import pandas as pd
import numpy as np
import os


def load_final_data():
    # Get the absolute path of the directory the script is in
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Define the directory where the final data file is located
    final_data_dir = os.path.join(script_dir, "../data/final_data_with_features.csv")

    # Load the final data from CSV file into a pandas DataFrame
    final_data = pd.read_csv(final_data_dir)

    return final_data


def calculate_weighted_avg_strikeouts(final_data, decay_rate=0.95):
    final_data_sorted = final_data.sort_values(by="Date")

    # Apply a decay factor to the 'SO_y_lag' column
    final_data_sorted["SO_y_lag_decay"] = (
        final_data_sorted.groupby("Player")["SO_y_lag"].apply(lambda x: x.ewm(alpha=1 - decay_rate).mean()).reset_index(level=0, drop=True)
    )

    # Compute the weighted average of strikeouts
    final_data_sorted["Weighted_SO_lag"] = (
        final_data_sorted.groupby("Player")["SO_y_lag_decay"].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    )

    return final_data_sorted


def calculate_strikeout_prop(final_data_sorted):
    # Create strikeout prop based on weighted average, cap it to the provided limits, and round up to the nearest half
    final_data_sorted["SO_Prop"] = np.clip(final_data_sorted["Weighted_SO_lag"], 3.5, 8.5).apply(lambda x: np.ceil(x * 2) / 2)

    return final_data_sorted


def save_props_data(final_data_with_prop):
    # Select the required columns
    props_data = final_data_with_prop[["Date", "Player", "SO_y", "SO_Prop"]]

    # Sort the DataFrame by 'Player' and 'Date'
    props_data.sort_values(by=["Player", "Date"], inplace=True)

    # Ensure the directory where the file will be saved exists
    final_data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(final_data_dir, exist_ok=True)

    # Save the props data to a new CSV file
    props_data.to_csv(os.path.join(final_data_dir, "final_props.csv"), index=False)


if __name__ == "__main__":
    final_data = load_final_data()
    final_data_weighted = calculate_weighted_avg_strikeouts(final_data)
    final_data_with_prop = calculate_strikeout_prop(final_data_weighted)
    save_props_data(final_data_with_prop)
