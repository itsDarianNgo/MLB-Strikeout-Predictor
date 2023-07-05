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


# Function to load data
def load_data(filename):
    return pd.read_csv(filename)


# Function to save data
def save_data(df, filename):
    df.to_csv(filename, index=False)


# Main function
def main():
    # Load data
    predictions = load_data("h2o_prediction.csv")
    betting_lines = load_data("betting_lines.csv")

    # Merge data on player name and game date
    data = pd.merge(predictions, betting_lines, on=["Player", "GameDate"])

    # Calculate odds
    data["OverOdds"], data["UnderOdds"] = zip(*data.apply(lambda row: calculate_odds(row["PredictedStrikeouts"], row["BettingLine"]), axis=1))

    # Determine whether to bet on over or under
    data["Bet"] = data.apply(lambda row: "Over" if row["OverOdds"] > row["UnderOdds"] else "Under", axis=1)

    # Check if the bet was successful
    data["SuccessfulBet"] = data.apply(
        lambda row: "Yes"
        if (row["Bet"] == "Over" and row["ActualStrikeouts"] > row["BettingLine"])
        or (row["Bet"] == "Under" and row["ActualStrikeouts"] < row["BettingLine"])
        else "No",
        axis=1,
    )

    # Save the results
    save_data(data, "output/predicted_odds.csv")


# Run the main function
if __name__ == "__main__":
    main()
