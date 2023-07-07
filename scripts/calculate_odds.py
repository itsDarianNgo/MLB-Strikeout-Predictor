import os
import pandas as pd


# Function to calculate odds
def calculate_odds(prediction, betting_line):
    if prediction >= betting_line:
        over_odds = prediction / (1 - prediction)
        under_odds = 1 - prediction
    else:
        over_odds = prediction
        under_odds = (1 - prediction) / prediction
    return over_odds, under_odds


# Function to load and preprocess data
def load_and_preprocess_data(filename, date_col):
    df = pd.read_csv(filename)
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df


# Function to save data
def save_data(df, filename):
    df.to_csv(filename, index=False)


# Adjust betting decision based on threshold
def adjust_decision(row, threshold):
    if row["predictedStrikeouts"] > row["prop"] * (1 + threshold):
        return "Over"
    elif row["predictedStrikeouts"] < row["prop"] * (1 - threshold):
        return "Under"
    else:
        return "Hold"


# Adjusted calculate_bet_amount function
def calculate_bet_amount(row):
    return 5 if row["Bet"] != "Hold" else 0


# Function to check if the bet was successful
def check_bet(row):
    if row["Bet"] == "Over" and row["SO_y"] > row["prop"]:
        return "Yes"
    elif row["Bet"] == "Under" and row["SO_y"] < row["prop"]:
        return "Yes"
    elif row["Bet"] == "Hold":
        return "Hold"
    else:
        return "No"


# Function to select top predictions and avoid 'hold' decisions
def select_top_predictions(df):
    return df[df["Bet"] != "Hold"].sort_values("predictedStrikeouts", ascending=False).drop_duplicates("Player")


# Function to decide whether to bet over or under based on the higher odds
def decide_bet_type(row):
    if row["OverOdds"] > row["UnderOdds"]:
        return "Over", row["OverOdds"]
    else:
        return "Under", row["UnderOdds"]


# Adjusted to select top six pitchers based on the best odds
def select_top_six_pitchers(df):
    return df.sort_values("BestBetOdds", ascending=False).drop_duplicates("Player").head(6)


# Adjusted to work with six pitchers
def create_rows_for_six_pitchers(df):
    grouped_data = []
    for date, df_date in df.groupby("Date"):
        top_six_pitchers = select_top_six_pitchers(df_date)
        if len(top_six_pitchers) == 6:
            row = {}
            for i, (index, pitcher_row) in enumerate(top_six_pitchers.iterrows()):
                row[f"Player{i+1}"] = pitcher_row["Player"]
                row[f"SO_y{i+1}"] = pitcher_row["SO_y"]
                row[f"predictedStrikeouts{i+1}"] = pitcher_row["predictedStrikeouts"]
                row[f"prop{i+1}"] = pitcher_row["prop"]
                row[f"Bet{i+1}"] = pitcher_row["Bet"]
                row[f"SuccessfulBet{i+1}"] = pitcher_row["SuccessfulBet"]
            row["Date"] = date
            row["BetAmount"] = 5  # Since each parlay bet is always $5
            grouped_data.append(row)
    return pd.DataFrame(grouped_data)


# Adjusted for new payout rules
def calculate_outcome(row):
    successful_bets = [row[f"SuccessfulBet{i+1}"] for i in range(6)]
    successful_bet_count = successful_bets.count("Yes")
    if successful_bet_count == 6:
        return 120  # Net gain of $120 when all bets are successful
    elif successful_bet_count == 5:
        return 5  # Net gain of $5 when 5 bets are successful
    elif successful_bet_count == 4:
        return -3  # Net loss of $3 when 4 bets are successful
    else:
        return -5  # Lose the initial $5 if 3 or fewer bets are successful


# Adjusted main function
def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, "../data/")
    predictions = load_and_preprocess_data(os.path.join(data_dir, "predictions.csv"), "Date")
    betting_lines = load_and_preprocess_data(os.path.join(data_dir, "prop_odds_history.csv"), "Date")

    data = pd.merge(predictions, betting_lines, on=["Player", "Date"])
    data["OverOdds"], data["UnderOdds"] = zip(*data.apply(lambda row: calculate_odds(row["predictedStrikeouts"], row["prop"]), axis=1))
    data["BestBetType"], data["BestBetOdds"] = zip(*data.apply(decide_bet_type, axis=1))
    data["Bet"] = data.apply(lambda row: adjust_decision(row, threshold=0.032), axis=1)
    data["BetAmount"] = data.apply(calculate_bet_amount, axis=1)
    data["SuccessfulBet"] = data.apply(check_bet, axis=1)

    # Select top predictions and avoid 'hold' decisions
    data = data[data["Bet"] != "Hold"].groupby("Date").apply(select_top_six_pitchers).reset_index(drop=True)

    # Create rows for six pitchers
    grouped_data = create_rows_for_six_pitchers(data)
    grouped_data["Outcome"] = grouped_data.apply(calculate_outcome, axis=1)
    total_profit_loss = grouped_data["Outcome"].sum()
    print(f"Total profit/loss: {total_profit_loss}")

    # Calculate accuracy
    grouped_data["SuccessfulParlay"] = grouped_data.apply(
        lambda row: "Yes" if sum([row[f"SuccessfulBet{i+1}"] == "Yes" for i in range(6)]) >= 4 else "No", axis=1
    )
    accuracy = (grouped_data["SuccessfulParlay"] == "Yes").mean()
    print(f"Parlay accuracy: {accuracy * 100:.2f}%")

    # Save the data
    save_data(grouped_data, os.path.join(data_dir, "predicted_odds.csv"))


if __name__ == "__main__":
    main()
