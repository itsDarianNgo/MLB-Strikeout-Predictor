import pandas as pd
import os


def load_data():
    # Get the absolute path of the directory the script is in
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Define the directories where the data files are located
    batting_data_dir = os.path.join(script_dir, "../data/BattingData/")
    pitching_data_dir = os.path.join(script_dir, "../data/PitchingData/")
    game_data_dir = os.path.join(script_dir, "../data/GameData/")

    # Define the list of columns to drop
    columns_to_drop = ["aLI", "RE24"]

    # Load and concatenate the data from all CSV files in each directory
    batting_data = pd.concat(
        [
            pd.read_csv(os.path.join(batting_data_dir, file), encoding="ISO-8859-1").drop(columns=columns_to_drop, errors="ignore")
            for file in os.listdir(batting_data_dir)
            if file.endswith(".csv")
        ],
        ignore_index=True,
    )
    pitching_data = pd.concat(
        [
            pd.read_csv(os.path.join(pitching_data_dir, file), encoding="ISO-8859-1").drop(columns=columns_to_drop, errors="ignore")
            for file in os.listdir(pitching_data_dir)
            if file.endswith(".csv")
        ],
        ignore_index=True,
    )
    game_data = pd.concat(
        [
            pd.read_csv(os.path.join(game_data_dir, file), encoding="ISO-8859-1").drop(columns=columns_to_drop, errors="ignore")
            for file in os.listdir(game_data_dir)
            if file.endswith(".csv")
        ],
        ignore_index=True,
    )

    return batting_data, pitching_data, game_data


def clean_data(batting_data, pitching_data, game_data):
    # Sort the data in chronological order
    batting_data["Date"] = pd.to_datetime(batting_data["Date"], format="%B %d, %Y")
    batting_data.sort_values(by="Date", inplace=True)

    pitching_data["Date"] = pd.to_datetime(pitching_data["Date"], format="%B %d, %Y")
    pitching_data.sort_values(by="Date", inplace=True)

    game_data["Date"] = pd.to_datetime(game_data["Date"], format="%B %d, %Y")
    game_data.sort_values(by="Date", inplace=True)

    # Fill in any missing values in batting_data and pitching_data
    batting_data.fillna(value=0, inplace=True)

    # Keep only the rows in pitching_data where both IR and IS are blank
    pitching_data = pitching_data[pitching_data["IR"].isna() & pitching_data["IS"].isna()]

    # Remove '%s' from 'WPA_y' and 'cWPA_y' columns and convert them to numeric in both batting and pitching data
    for data in [batting_data, pitching_data]:
        for col in ["cWPA", "WPA-"]:
            if col in data.columns:
                data.loc[:, col] = data[col].str.strip().str.replace("[^\d.]", "", regex=True)
                data.loc[:, col] = pd.to_numeric(data[col], errors="coerce")

        data.fillna(value=0, inplace=True)

    # Drop the 'details' column from batting_data
    if "Details" in batting_data.columns:
        batting_data.drop("Details", axis=1, inplace=True)

    return batting_data, pitching_data, game_data


def merge_data(batting_data, pitching_data, game_data):
    # Merge the batting data and pitching data on 'GameID', 'date', and 'player ID'
    merged_data = pd.merge(batting_data, pitching_data, on=["GameID", "Date", "Player", "Team"])

    # Merge the merged data and game data on 'GameID' and 'date'
    final_data = pd.merge(merged_data, game_data, on=["GameID", "Date"])

    return final_data


def save_final_data(final_data):
    # Ensure the directory where the file will be saved exists
    final_data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(final_data_dir, exist_ok=True)

    # Save the final data to a CSV file
    final_data.to_csv(os.path.join(final_data_dir, "final_data_with_features.csv"), index=False)


if __name__ == "__main__":
    batting_data, pitching_data, game_data = load_data()
    batting_data_clean, pitching_data_clean, game_data_clean = clean_data(batting_data, pitching_data, game_data)
    final_data = merge_data(batting_data_clean, pitching_data_clean, game_data_clean)
    save_final_data(final_data)
