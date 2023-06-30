import pandas as pd
from data_preprocessing import load_data, clean_data, merge_data, save_final_data


def generate_avg_strikeouts_per_year_feature(data):
    """Generates the average strikeouts per year feature for each player."""
    data["Year"] = pd.DatetimeIndex(data["Date"]).year
    avg_so_per_year = data.groupby(["Player", "Year"])["SO_y"].mean().reset_index()
    avg_so_per_year.rename(columns={"SO_y": "avg_strikeouts_per_year"}, inplace=True)
    return avg_so_per_year


def generate_features(data, feature_generators):
    """Generates features for the data using a list of feature generator functions."""
    # Sort the data by player and date
    data.sort_values(["Player", "Date"], inplace=True)

    features = [feature_generator(data) for feature_generator in feature_generators]
    feature_df = pd.concat(features, axis=1)
    return feature_df


def main():
    # Load, clean, and merge the data
    batting_data, pitching_data, game_data = load_data()
    batting_data, pitching_data, game_data = clean_data(batting_data, pitching_data, game_data)
    final_data = merge_data(batting_data, pitching_data, game_data)

    # Specify the feature generator functions
    feature_generators = [generate_avg_strikeouts_per_year_feature]

    # Generate the features
    features_data = generate_features(final_data, feature_generators)

    # Merge the original data with the generated features
    final_data = pd.merge(final_data, features_data, on=["Player", "Year"], how="left")

    # Sort the data by player and date
    final_data.sort_values(["Player", "Date"], inplace=True)

    # Define the order of columns
    first_cols = ["GameID", "Date", "Team", "SO_y"]
    new_cols = features_data.columns.tolist()  # new feature column names
    last_cols = [col for col in final_data.columns if col not in first_cols + new_cols]

    # Reorder the columns
    final_data = final_data[first_cols + new_cols + last_cols]

    # Save the final data
    save_final_data(final_data)


if __name__ == "__main__":
    main()
