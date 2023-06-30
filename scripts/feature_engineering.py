import pandas as pd
import numpy as np
from data_preprocessing import load_data, clean_data, merge_data, save_final_data


def get_player_data(data, player):
    """Get and sort player's data."""
    player_df = data[data["Player"] == player].copy()
    player_df.sort_values("Date", inplace=True)
    player_df["Opp"] = np.where(player_df["Team"] == player_df["home_team"], player_df["away_team"], player_df["home_team"])
    return player_df


def calculate_recent_games_stats(player_df, features):
    """Calculate recent games statistics for given features."""
    for feature in features:
        for i in range(1, 6):  # For last 5 games
            player_df[f"{feature}_game_{i}"] = player_df[feature].shift(i)
    return player_df


def calculate_historical_stats(player_df, feature_columns, stats):
    """Calculate historical statistics for the given features."""
    stats_df_list = []

    for feature in feature_columns:
        if np.issubdtype(player_df[feature].dtype, np.number):
            feature_stats = {}
            for stat in stats:
                if stat == "mean":
                    feature_stats[f"{feature}_historical_mean"] = player_df[feature].expanding().mean()
                elif stat == "median":
                    feature_stats[f"{feature}_historical_median"] = player_df[feature].expanding().median()
                elif stat == "std":
                    feature_stats[f"{feature}_historical_std"] = player_df[feature].expanding().std()
                elif stat == "max":
                    feature_stats[f"{feature}_historical_max"] = player_df[feature].expanding().max()
                elif stat == "min":
                    feature_stats[f"{feature}_historical_min"] = player_df[feature].expanding().min()
            stats_df = pd.DataFrame(feature_stats)
            stats_df_list.append(stats_df)
    final_stats_df = pd.concat(stats_df_list, axis=1)
    player_df = pd.concat([player_df, final_stats_df], axis=1)
    return player_df


def calculate_average_strikeouts_vs_team(player_df):
    # Sort the dataframe by Player and Date
    player_df = player_df.sort_values(["Player", "Date"])

    # Group dataframe by Player and Opp, then apply calculations
    player_df = player_df.groupby(["Player", "Opp"]).apply(lambda df: calculate_avg_strikeouts(df)).reset_index(drop=True)

    # Fill -1 for any other missing values in the dataframe
    player_df.fillna(-1, inplace=True)

    return player_df


def calculate_avg_strikeouts(df):
    df["cumulative_strikeouts"] = df["SO_y"].shift().cumsum()
    df["cumulative_games"] = np.arange(len(df))

    # Calculate average strikeouts
    df["average_strikeouts_vs_team"] = df["cumulative_strikeouts"] / df["cumulative_games"]

    # Replace NaN values with zero for the first game of each player against each team
    df["average_strikeouts_vs_team"].fillna(0, inplace=True)

    return df


def calculate_yearly_avg_SO(player_df):
    """Calculate average strikeouts (SO_y) per year for a player."""
    # Extract year from 'Date' column
    player_df["Year"] = pd.DatetimeIndex(player_df["Date"]).year
    # Calculate yearly average strikeouts (SO_y)
    yearly_avg_SO_y = player_df.groupby(["Player", "Year"])["SO_y"].transform("mean")
    player_df = player_df.assign(Yearly_Avg_SO_y=yearly_avg_SO_y)
    return player_df


def prepare_final_df(historical_df):
    historical_features = [col for col in historical_df.columns if "historical" in col or "game" in col]
    historical_df = historical_df[
        historical_features + ["Player", "Date", "Team", "GameID", "SO_y", "average_strikeouts_vs_team", "Opp", "Yearly_Avg_SO_y"]
    ]

    historical_df.dropna(axis=0, how="any", inplace=True)
    historical_df = historical_df[
        ["Player", "Date", "Team", "GameID", "SO_y", "average_strikeouts_vs_team", "Opp", "Yearly_Avg_SO_y"] + historical_features
    ]
    return historical_df


def generate_features(data, feature_columns, stats):
    historical_df = pd.DataFrame()

    # These are the features you calculated for recent games in your original code
    recent_game_features = ["Pit_y", "SO_y", "H_x", "H_y", "BB_x", "BB_y", "HR", "ER"]

    for player in data["Player"].unique():
        player_df = data[data["Player"] == player].copy()
        player_df.sort_values("Date", inplace=True)
        player_df["Opp"] = np.where(player_df["Team"] == player_df["home_team"], player_df["away_team"], player_df["home_team"])

        # Shift all the features by 1
        player_df[feature_columns] = player_df[feature_columns].shift(1)

        player_df = calculate_recent_games_stats(player_df, recent_game_features)
        player_df = calculate_historical_stats(player_df, feature_columns, stats)
        player_df = calculate_average_strikeouts_vs_team(player_df)
        player_df = calculate_yearly_avg_SO(player_df)
        historical_df = pd.concat([historical_df, player_df])

    historical_df = prepare_final_df(historical_df)
    return historical_df


def main():
    batting_data, pitching_data, game_data = load_data()
    batting_data, pitching_data, game_data = clean_data(batting_data, pitching_data, game_data)
    final_data = merge_data(batting_data, pitching_data, game_data)

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
    stats = ["mean", "median", "std"]

    final_data = generate_features(final_data, feature_columns, stats)
    save_final_data(final_data)


if __name__ == "__main__":
    main()
