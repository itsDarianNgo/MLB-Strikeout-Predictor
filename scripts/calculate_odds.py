import os
import pandas as pd


# Function to calculate odds
def calculate_odds(prediction, betting_line):
    if prediction >= betting_line:
        # Over odds calculation
        over_odds = prediction / (1 - prediction)
        under_odds = 1 - prediction
    else:
        # Under odds calculation
        over_odds = prediction
        under_odds = (1 - prediction) / prediction

    return over_odds, under_odds


# Function to load and preprocess data
def load_and_preprocess_data(filename, date_col):
    df = pd.read_csv(filename)
    df[date_col] = pd.to_datetime(df[date_col]).dt.date  # convert to date format and remove time
    return df


# Function to save data
def save_data(df, filename):
    df.to_csv(filename, index=False)


# Main function
def main():
    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Define the directories where the data files are located
    data_dir = os.path.join(script_dir, "../data/")

    # Load and preprocess data
    predictions = load_and_preprocess_data(os.path.join(data_dir, "predictions.csv"), "Date")
    betting_lines = load_and_preprocess_data(os.path.join(data_dir, "prop_odds_history.csv"), "Date")

    # Merge data on player name and game date
    data = pd.merge(predictions, betting_lines, on=["Player", "Date"])

    # Calculate odds
    data["OverOdds"], data["UnderOdds"] = zip(*data.apply(lambda row: calculate_odds(row["predictedStrikeouts"], row["prop"]), axis=1))

    # Determine whether to bet on over or under
    data["Bet"] = data.apply(lambda row: "Over" if row["predictedStrikeouts"] > row["prop"] else "Under", axis=1)

    # Check if the bet was successful
    data["SuccessfulBet"] = data.apply(
        lambda row: "Yes" if (row["Bet"] == "Over" and row["SO_y"] > row["prop"]) or (row["Bet"] == "Under" and row["SO_y"] < row["prop"]) else "No",
        axis=1,
    )

    # Save the results
    save_data(data, os.path.join(data_dir, "predicted_odds.csv"))

    # Calculate accuracy
    accuracy = (data["SuccessfulBet"] == "Yes").mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")


# Run the main function
if __name__ == "__main__":
    main()
