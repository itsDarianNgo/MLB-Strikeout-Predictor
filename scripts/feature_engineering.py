import pandas as pd
from data_preprocessing import load_data, clean_data, merge_data, save_final_data
import numpy as np


def generate_features(data, feature_columns, stats):
    """Generates features for the data."""

    # Initialize an empty DataFrame to store the final data
    historical_df = pd.DataFrame()

    # For each player in the data
    for player in data["Player"].unique():
        # Get the player's data
        player_df = data[data["Player"] == player].copy()

        # Shift all the features by 1
        player_df[feature_columns] = player_df[feature_columns].shift(1)

        # For each feature
        for feature in feature_columns:
            # Check if the column is numeric
            if np.issubdtype(player_df[feature].dtype, np.number):
                # Calculate the statistics
                for stat in stats:
                    if stat == "mean":
                        player_df[f"{feature}_historical_mean"] = player_df[feature].expanding().mean()
                    elif stat == "median":
                        player_df[f"{feature}_historical_median"] = player_df[feature].expanding().median()
                    elif stat == "std":
                        player_df[f"{feature}_historical_std"] = player_df[feature].expanding().std()

        # Add the player's data to the final DataFrame
        historical_df = pd.concat([historical_df, player_df])

    # Keep all the features, the historical features, and the target 'SO_y'
    all_features = [col for col in historical_df.columns]
    historical_df = historical_df[all_features]

    # Drop rows with missing values
    historical_df.dropna(axis=0, how="any", inplace=True)

    return historical_df


def main():
    # Load, clean, and merge the data
    batting_data, pitching_data, game_data = load_data()
    batting_data, pitching_data, game_data = clean_data(batting_data, pitching_data, game_data)
    final_data = merge_data(batting_data, pitching_data, game_data)

    # Specify the features and statistics you want to calculate
    feature_columns = [
        "AB",
        "R_x",
        "H_x",
        "RBI",
        "BB_x",
        "SO_x",
        "PA",
        "BA",
        "OBP",
        "SLG",
        "OPS",
        "Pit_x",
        "Str_x",
        "WPA_x",
        "aLI_x",
        "WPA+",
        "WPA-",
        "cWPA_x",
        "acLI_x",
        "RE24_x",
        "PO",
        "A",
        "IP",
        "H_y",
        "R_y",
        "ER",
        "BB_y",
        "SO_y",
        "HR",
        "ERA",
        "BF",
        "Pit_y",
        "Str_y",
        "Ctct",
        "StS",
        "StL",
        "GB",
        "FB",
        "LD",
        "Unk",
        "GSc",
        "IR",
        "IS",
        "WPA_y",
        "aLI_y",
        "cWPA_y",
        "acLI_y",
        "RE24_y",
    ]
    stats = ["mean"]  # Add any other statistics you want to calculate here

    # Generate the features
    final_data = generate_features(final_data, feature_columns, stats)

    # Save the final data
    save_final_data(final_data)


if __name__ == "__main__":
    main()
