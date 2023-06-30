import pandas as pd
from data_preprocessing import load_data, clean_data, merge_data, save_final_data


def calculate_cumulative_average_strikeouts(player_data):
    """Calculate the cumulative average strikeouts per year for a single player."""
    # Extract year from date and calculate running sum and count per year
    player_data["Year"] = pd.DatetimeIndex(player_data["Date"]).year
    player_data["SO_y_shifted"] = player_data.groupby("Year")["SO_y"].shift()
    player_data["Cumulative_SO"] = player_data.groupby("Year")["SO_y_shifted"].cumsum()
    player_data["Game_Count"] = player_data.groupby("Year").cumcount()

    # Calculate cumulative average strikeouts, with the first game set to 0
    player_data["Avg_SO_per_year"] = 0
    player_data.loc[player_data["Game_Count"] != 0, "Avg_SO_per_year"] = player_data["Cumulative_SO"] / player_data["Game_Count"]

    # Drop auxiliary columns
    player_data = player_data.drop(columns=["SO_y_shifted", "Cumulative_SO", "Game_Count", "Year"])

    return player_data


def generate_features(data):
    """Generate features for the dataset."""

    players = data["Player"].unique()
    player_dfs = [calculate_cumulative_average_strikeouts(data[data["Player"] == player].copy()) for player in players]

    # Combine all player data
    historical_df = pd.concat(player_dfs, axis=0)

    # Drop rows with missing values
    historical_df.dropna(axis=0, how="any", inplace=True)

    # Reorder columns
    cols = historical_df.columns.tolist()
    cols.insert(1, cols.pop(cols.index("SO_y")))
    cols.insert(2, cols.pop(cols.index("Avg_SO_per_year")))
    historical_df = historical_df[cols]

    return historical_df


def main():
    # Load, clean, and merge the data
    batting_data, pitching_data, game_data = load_data()
    batting_data, pitching_data, game_data = clean_data(batting_data, pitching_data, game_data)
    final_data = merge_data(batting_data, pitching_data, game_data)

    # Generate the features
    final_data = generate_features(final_data)

    # Save the final data
    save_final_data(final_data)


if __name__ == "__main__":
    main()
