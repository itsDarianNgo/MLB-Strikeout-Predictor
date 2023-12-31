import pandas as pd
import numpy as np
import os


def load_final_data():
    # Get the absolute path of the directory the script is in
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Define the directory where the final data file is located
    final_data_dir = os.path.join(script_dir, "../data/final_data_with_features.csv")

    # Load the final data from CSV file into a pandas DataFrame
    final_data = pd.read_csv(final_data_dir)

    return final_data


def calculate_weighted_avg_strikeouts(final_data, decay_rate=0.90):  # Lower decay rate
    final_data_sorted = final_data.sort_values(by="Date")

    # Apply a decay factor to the 'SO_y_lag' column
    final_data_sorted["SO_y_lag_decay"] = (
        final_data_sorted.groupby("Player")["SO_y_lag"].apply(lambda x: x.ewm(alpha=1 - decay_rate).mean()).reset_index(level=0, drop=True)
    )

    # Compute the weighted average of strikeouts
    final_data_sorted["Weighted_SO_lag"] = (
        final_data_sorted.groupby("Player")["SO_y_lag_decay"].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)  # Smaller window size
    )

    return final_data_sorted


def calculate_strikeout_prop(final_data_sorted):
    # Create strikeout prop based on weighted average, cap it to the provided limits, and round up to the nearest half
    final_data_sorted["SO_Prop"] = np.clip(final_data_sorted["Weighted_SO_lag"], 4.0, 8.5).apply(lambda x: np.ceil(x * 2) / 2)

    return final_data_sorted


def save_props_data(final_data_with_prop):
    # Select the required columns
    props_data = final_data_with_prop[["Date", "Player", "SO_y", "SO_Prop"]]

    # Sort the DataFrame by 'Player' and 'Date'
    props_data.sort_values(by=["Player", "Date"], inplace=True)

    # Ensure the directory where the file will be saved exists
    final_data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(final_data_dir, exist_ok=True)

    # Save the props data to a new CSV file
    props_data.to_csv(os.path.join(final_data_dir, "final_props.csv"), index=False)


from sklearn.metrics import mean_squared_error


def load_scraped_data():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    scraped_data_dir = os.path.join(script_dir, "../data/prop_odds_history.csv")
    scraped_data = pd.read_csv(scraped_data_dir)

    return scraped_data


def evaluate_model(predicted_data, actual_data):
    # Merge the predicted and actual data
    merged_data = pd.merge(predicted_data, actual_data, on=["Date", "Player"], how="inner")

    # Calculate the mean squared error between predicted and actual prop
    mse = mean_squared_error(merged_data["SO_Prop"], merged_data["prop"])
    print(f"Mean Squared Error between predicted and actual prop: {mse}")

    # Calculate the mean difference between predicted and actual prop
    mean_diff = (merged_data["SO_Prop"] - merged_data["prop"]).mean()
    print(f"Mean difference between predicted and actual prop: {mean_diff}")

    # Keep only the specified columns
    merged_data = merged_data[["Date", "Player", "SO_y", "SO_Prop", "prop"]]

    # Sort the DataFrame by 'Player' and 'Date'
    merged_data.sort_values(by=["Player", "Date"], inplace=True)

    # Save the merged data to a CSV file
    merged_data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(merged_data_dir, exist_ok=True)
    merged_data.to_csv(os.path.join(merged_data_dir, "merged_props.csv"), index=False)


def unify_date_format(df, date_column):
    # Convert the 'Date' column to datetime format and remove the time part
    df[date_column] = pd.to_datetime(df[date_column]).dt.date
    return df


if __name__ == "__main__":
    final_data = load_final_data()
    final_data_weighted = calculate_weighted_avg_strikeouts(final_data)
    final_data_with_prop = calculate_strikeout_prop(final_data_weighted)
    save_props_data(final_data_with_prop)

    # Load the scraped data and convert the date format
    # scraped_data = load_scraped_data()
    # scraped_data = unify_date_format(scraped_data, "Date")
    # final_data_with_prop = unify_date_format(final_data_with_prop, "Date")

    # # Evaluate the model
    # evaluate_model(final_data_with_prop, scraped_data)
