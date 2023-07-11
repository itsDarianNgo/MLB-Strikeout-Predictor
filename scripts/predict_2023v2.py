import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import os
import numpy as np

# Initialize the H2O environment
h2o.init()


def load_and_merge_data():
    # Load the data
    prop_odds_history = pd.read_csv("data/prop_odds_history.csv")
    final_data_with_features = pd.read_csv("data/final_data_with_features_2023.csv")

    # Convert 'Date' to datetime and extract only the date
    prop_odds_history["Date"] = pd.to_datetime(prop_odds_history["Date"]).dt.date
    final_data_with_features["Date"] = pd.to_datetime(final_data_with_features["Date"]).dt.date

    # Merge the dataframes on 'Date' and 'Player'
    merged_data = pd.merge(prop_odds_history, final_data_with_features, how="inner", on=["Date", "Player"])

    return merged_data


def predict_with_models(final_data_h2o, regression_model_path, classification_model_path):
    aml_regression = h2o.load_model(regression_model_path)
    aml_classification = h2o.load_model(classification_model_path)

    # Predict on the data using the regression model
    preds_regression = aml_regression.predict(final_data_h2o)
    # Add the regression predictions to the data
    final_data_h2o = final_data_h2o.cbind(preds_regression)
    final_data_h2o.set_name(len(final_data_h2o.names) - 1, "SO_y_pred")

    # Create the binary target based on the actual 'SO_y' and 'prop'
    final_data_h2o["SO_y_binary"] = (final_data_h2o["SO_y"] > final_data_h2o["SO_Prop"]).ifelse(1, 0)

    # Predict on the data using the classification model
    preds_classification = aml_classification.predict(final_data_h2o)

    # Convert the probabilities to class labels
    preds_classification["SO_y_binary_pred"] = (preds_classification["p1"] >= 0.5).ifelse(1, 0)

    # Add the binary predictions to the data
    final_data_h2o = final_data_h2o.cbind(preds_classification[["SO_y_binary_pred", "p1"]])
    final_data_h2o.set_name(len(final_data_h2o.names) - 1, "p0")

    # Convert the H2O Frame to pandas DataFrame
    final_data_df = final_data_h2o.as_data_frame()

    # Add a new column to indicate whether the binary prediction was correct
    final_data_df["binary_prediction_correct"] = np.where(
        final_data_df["SO_y"] == final_data_df["SO_Prop"],
        "Equal",
        np.where(final_data_df["SO_y_binary_pred"] == final_data_df["SO_y_binary"], "Yes", "No"),
    )
    return final_data_df


def main():
    # Load and merge the data
    merged_data = load_and_merge_data()

    # Convert the merged data to H2O Frame
    final_data_h2o = h2o.H2OFrame(merged_data)

    # Paths to the saved models (update these paths as necessary)
    regression_model_path = "./model_output/regression/GLM_1_AutoML_1_20230711_40542"
    classification_model_path = "./model_output/classification/GBM_1_AutoML_2_20230711_41655"
    final_data_with_predictions = predict_with_models(final_data_h2o, regression_model_path, classification_model_path)

    # Select only the necessary columns for the final output
    selected_columns = ["Date", "Player", "SO_y", "SO_y_pred", "SO_Prop", "SO_y_binary", "SO_y_binary_pred", "p0", "binary_prediction_correct"]

    final_data_with_predictions = final_data_with_predictions[selected_columns]

    # Sort the data by 'Player' and 'Date'
    final_data_with_predictions = final_data_with_predictions.sort_values(by=["Player", "Date"])

    # Print the percentage of binary predictions that were correct
    print(final_data_with_predictions["binary_prediction_correct"].value_counts(normalize=True) * 100)

    # Save the data with predictions to a CSV file
    final_data_with_predictions.to_csv("./data/merged_data_with_predictions.csv", index=False)


main()
