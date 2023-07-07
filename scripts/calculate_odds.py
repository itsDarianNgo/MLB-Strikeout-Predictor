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


# Adjusting pair_pitchers to use BestBetType and BestBetOdds
def pair_pitchers(df):
    df_sorted = df.sort_values(by="BestBetOdds", ascending=False)
    players = df_sorted["Player"].tolist()
    pairs = [(players[i], players[i + 1]) for i in range(0, len(players) - 1, 2)]
    return pairs


def pair_pitchers_and_create_rows(df):
    paired_data = []
    for date, df_date in df.groupby("Date"):
        pairs = pair_pitchers(df_date)
        for pair in pairs:
            row1 = df_date[df_date["Player"] == pair[0]].iloc[0].to_dict()
            row2 = df_date[df_date["Player"] == pair[1]].iloc[0].to_dict()
            row = {}
            row["Date"] = date
            row["Player1"] = pair[0]
            row["Player2"] = pair[1]
            row["SO_y1"] = row1["SO_y"]
            row["SO_y2"] = row2["SO_y"]
            row["predictedStrikeouts1"] = row1["predictedStrikeouts"]
            row["predictedStrikeouts2"] = row2["predictedStrikeouts"]
            row["prop1"] = row1["prop"]
            row["prop2"] = row2["prop"]
            row["Bet1"] = row1["Bet"]
            row["Bet2"] = row2["Bet"]
            row["SuccessfulBet1"] = row1["SuccessfulBet"]
            row["SuccessfulBet2"] = row2["SuccessfulBet"]
            row["BetAmount"] = 5  # Since each parlay bet is always $5
            paired_data.append(row)
    return pd.DataFrame(paired_data), pairs


# Function to check if the parlay was successful
def check_parlay(df):
    if (df["SuccessfulBet"] == "Yes").all():
        return "Yes"
    elif (df["SuccessfulBet"] == "No").any():
        return "No"
    else:
        return "Hold"


def calculate_outcome(row):
    successful_bets = [row["SuccessfulBet1"], row["SuccessfulBet2"]]
    if successful_bets.count("Yes") == 2:
        return 15  # Return 15 when both bets are successful
    else:
        return -5  # Lose the initial $5 if any bet is unsuccessful


# Main function
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
    data = data.groupby("Date").apply(select_top_predictions).reset_index(drop=True)

    # Pair pitchers and create rows
    paired_data, _ = pair_pitchers_and_create_rows(data)
    paired_data["Outcome"] = paired_data.apply(calculate_outcome, axis=1)
    total_profit_loss = paired_data["Outcome"].sum()
    print(f"Total profit/loss: {total_profit_loss}")

    # Calculate parlay accuracy
    paired_data["SuccessfulParlay"] = paired_data.apply(
        lambda row: "Yes" if row["SuccessfulBet1"] == "Yes" and row["SuccessfulBet2"] == "Yes" else "No", axis=1
    )
    accuracy = (paired_data["SuccessfulParlay"] == "Yes").mean()
    print(f"Parlay accuracy: {accuracy * 100:.2f}%")

    # Save the data
    save_data(paired_data, os.path.join(data_dir, "predicted_odds.csv"))


if __name__ == "__main__":
    main()
