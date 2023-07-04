import pandas as pd
from data_preprocessing import load_data, clean_data, merge_data, save_final_data
from sklearn.linear_model import LinearRegression
import numpy as np

LAG_COLUMNS = [
    "SO_y",
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
    "WPA+",
    "WPA-",
    "cWPA_x",
    "acLI_x",
    "PO",
    "A",
    "IP",
    "H_y",
    "R_y",
    "ER",
    "BB_y",
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
    "cWPA_y",
    "acLI_y",
]


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


def generate_rolling_avg_SO(player_data, window=10):
    """Generate rolling average strikeouts for a single player."""
    player_data.sort_values(by="Date", inplace=True)

    # Create a new column with the rolling mean
    player_data["Avg_SO_Rolling"] = player_data["SO_y"].rolling(window=window, min_periods=1).mean().shift()

    return player_data


def generate_rolling_avg_StS(player_data, window=10):
    """Generate rolling average strikes swinging for a single player."""
    player_data.sort_values(by="Date", inplace=True)

    # Create a new column with the rolling mean
    player_data["Avg_StS_Rolling"] = player_data["StS"].rolling(window=window, min_periods=1).mean().shift()

    return player_data


def generate_rolling_avg_Str(player_data, window=10):
    """Generate rolling average strikes for a single player."""
    player_data.sort_values(by="Date", inplace=True)

    # Create a new column with the rolling mean
    player_data["Avg_Str_Rolling"] = player_data["Str_y"].rolling(window=window, min_periods=1).mean().shift()

    return player_data


def generate_lagged_features(player_data):
    """Generate lagged features for a single player."""
    player_data.sort_values(by="Date", inplace=True)
    for column in LAG_COLUMNS:
        player_data[f"{column}_lag"] = player_data[column].shift()

    # Define columns to drop (all lag columns except target)
    columns_to_drop = [col for col in LAG_COLUMNS if col != "SO_y"]

    # Drop original columns excluding the target
    player_data.drop(columns=columns_to_drop, inplace=True)

    # Fill NaN values with 0
    player_data.fillna(0, inplace=True)

    return player_data


def calculate_recent_performance_trend(player_data, games=5):
    """Calculate recent performance trend for a single player."""
    player_data.sort_values(by="Date", inplace=True)

    # Initialize a linear regression model
    model = LinearRegression()

    # Prepare an array to hold the trends
    trends = np.zeros(len(player_data))

    for i in range(games, len(player_data)):
        # Extract the target values from the last games games
        y = player_data.iloc[i - games : i]["SO_y"].values.reshape(-1, 1)

        # Prepare an array of game numbers to use as our feature
        X = np.array(range(games)).reshape(-1, 1)

        # Fit the linear regression model
        model.fit(X, y)

        # The slope of the line is in model.coef_
        trends[i] = model.coef_[0][0]

    # Add the trends to the player_data DataFrame
    player_data["Recent_Performance_Trend"] = trends

    return player_data


def generate_opposing_team(data):
    """Generate opposing team for each player in each game."""
    data["Opposing_Team"] = data.apply(lambda row: row["home_team"] if row["Team"] != row["home_team"] else row["away_team"], axis=1)
    return data


def generate_pitcher_performance_against_teams(player_data):
    """Generate average strikeouts against each team for a single player."""
    player_data.sort_values(by="Date", inplace=True)

    # Create an expanding window mean, grouping by the opposing team
    player_data["Avg_SO_vs_Team"] = player_data.groupby("Opposing_Team")["SO_y"].expanding(min_periods=1).mean().reset_index(level=0, drop=True)

    # Shift the data to avoid data leakage
    player_data["Avg_SO_vs_Team"] = player_data.groupby("Opposing_Team")["Avg_SO_vs_Team"].shift()

    return player_data


def generate_pitcher_fatigue(player_data):
    """Generate a pitcher fatigue feature for a single player."""
    player_data.sort_values(by="Date", inplace=True)

    # Convert the Date column to datetime format if it's not already
    player_data["Date"] = pd.to_datetime(player_data["Date"])

    # Calculate the difference in days between consecutive games
    player_data["Days_Since_Last_Game"] = player_data["Date"].diff().dt.days

    # Fill the NaN values (for the first game) with 0
    player_data["Days_Since_Last_Game"].fillna(0, inplace=True)

    return player_data


def generate_KBB_ratio(player_data):
    """Generate the lagged K/BB ratio for a single player."""
    player_data.sort_values(by="Date", inplace=True)

    # To avoid division by zero, add a small constant to the denominator
    player_data["BB_y_shifted"] = player_data["BB_y"].shift() + 1e-10

    player_data["SO_y_shifted"] = player_data["SO_y"].shift()
    player_data["K/BB_lag"] = player_data["SO_y_shifted"].cumsum() / player_data["BB_y_shifted"].cumsum()

    # Fill NaN values with 0
    player_data.fillna(0, inplace=True)

    # Drop auxiliary columns
    player_data.drop(columns=["BB_y_shifted", "SO_y_shifted"], inplace=True)

    return player_data


def generate_features(data):
    """Generate features for the dataset."""

    # Generate the Opposing_Team column
    data = generate_opposing_team(data)

    players = data["Player"].unique()

    player_dfs = []
    for player in players:
        player_data = data[data["Player"] == player].copy()
        player_data = generate_rolling_avg_SO(player_data)
        player_data = generate_rolling_avg_StS(player_data)
        player_data = generate_rolling_avg_Str(player_data)
        player_data = generate_pitcher_performance_against_teams(player_data)
        player_data = generate_pitcher_fatigue(player_data)
        player_data = generate_KBB_ratio(player_data)
        player_data = generate_lagged_features(player_data)
        player_data = calculate_cumulative_average_strikeouts(player_data)
        player_data = calculate_recent_performance_trend(player_data, games=5)
        player_dfs.append(player_data)

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


def drop_unnecessary_columns(data):
    """Drop unnecessary columns from the DataFrame."""
    columns_to_drop = ["home_score", "away_score", "winner", "venue"]
    data.drop(columns=columns_to_drop, inplace=True)
    return data


def main():
    # Load, clean, and merge the data
    batting_data, pitching_data, game_data = load_data()
    batting_data, pitching_data, game_data = clean_data(batting_data, pitching_data, game_data)
    final_data = merge_data(batting_data, pitching_data, game_data)

    # Drop unnecessary columns
    final_data = drop_unnecessary_columns(final_data)

    # Generate the features
    final_data = generate_features(final_data)

    # Save the final data
    save_final_data(final_data)


if __name__ == "__main__":
    main()
