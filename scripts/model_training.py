import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import os

# Initialize the H2O environment
h2o.init()


def load_preprocessed_data():
    # Get the absolute path of the directory the script is in
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Define the directory where the preprocessed data file is located
    final_data_dir = os.path.join(script_dir, "../data/final_data_with_features.csv")

    # Load the preprocessed data
    final_data = pd.read_csv(final_data_dir, encoding="ISO-8859-1")

    # Convert 'Date' column to datetime format
    final_data["Date"] = pd.to_datetime(final_data["Date"])

    return final_data


def main():
    # Load the preprocessed data
    final_data = load_preprocessed_data()

    # Sort by 'Date' to prevent data leakage
    final_data["Date"] = pd.to_datetime(final_data["Date"], format="%B %d, %Y")

    # Convert the preprocessed data to H2O Frame
    final_data_h2o = h2o.H2OFrame(final_data)

    # Define the target column for regression
    target_regression = "SO_y"

    # Define the predictors
    predictors = final_data_h2o.columns
    predictors.remove(target_regression)

    # Initialize the H2O AutoML model for regression
    aml_regression = H2OAutoML(max_models=1, seed=1, max_runtime_secs=1200)

    # Train the regression model
    aml_regression.train(x=predictors, y=target_regression, training_frame=final_data_h2o)

    # Predict on the entire data using the regression model
    preds_regression = aml_regression.leader.predict(final_data_h2o)

    # Add the predictions to the entire dataset
    final_data_h2o["SO_y_pred"] = preds_regression

    # Create the binary target
    final_data_h2o["SO_y_binary"] = (final_data_h2o["SO_y_pred"] > final_data_h2o["SO_Prop"]).ifelse(1, 0)

    # Define the target column for classification
    target_classification = "SO_y_binary"

    # Calculate the train/test split index again
    split_index = int(final_data_h2o.shape[0] * 0.7)

    # Split the data into training and testing sets again
    train = final_data_h2o[:split_index, :]
    test = final_data_h2o[split_index:, :]

    # Convert the binary target to a factor
    train[target_classification] = train[target_classification].asfactor()
    test[target_classification] = test[target_classification].asfactor()

    # Initialize the H2O AutoML model for classification
    aml_classification = H2OAutoML(max_models=1, seed=1, max_runtime_secs=1200)

    # Train the classification model
    aml_classification.train(x=predictors, y=target_classification, training_frame=train)

    # Predict on the test data using the classification model
    preds_classification = aml_classification.leader.predict(test)

    # Rename the 'predict' column to 'SO_y_binary_pred'
    preds_classification.set_names(["SO_y_binary_pred", "p0", "p1"])

    # Convert 'SO_y_binary_pred' from factor to integer
    preds_classification["SO_y_binary_pred"] = preds_classification["SO_y_binary_pred"].asnumeric()

    # Add the binary predictions and class probabilities to the test set
    test = test.cbind(preds_classification)

    # Create the 'binary_prediction_correct' column
    test["binary_prediction_correct"] = (
        (test["SO_y_binary_pred"] == 1) & (test["SO_y"] > test["SO_Prop"]) | (test["SO_y_binary_pred"] == 0) & (test["SO_y"] <= test["SO_Prop"])
    ).ifelse("Yes", "No")

    # Save the test set to a CSV file
    test_to_save = test[
        ["Date", "Player", "SO_y", "SO_y_pred", "SO_Prop", "SO_y_binary", "SO_y_binary_pred", "binary_prediction_correct", "p0", "p1"]
    ]
    h2o.export_file(test_to_save, path="./model_output/test_predictions.csv", force=True)

    # View the performance of the leader model on the test data
    perf = aml_classification.leader.model_performance(test)
    print(perf)

    # For each model in the leaderboard (excluding Stacked Ensembles), print the variable importance
    model_ids = list(aml_classification.leaderboard["model_id"].as_data_frame().iloc[:, 0])
    for mid in model_ids:
        if "StackedEnsemble" not in mid:
            m = h2o.get_model(mid)
            print(f"\nModel ID: {mid}")
            print(m.varimp(use_pandas=True))

    # Save the models
    h2o.save_model(aml_regression.leader, path="./model_output/regression")
    h2o.save_model(aml_classification.leader, path="./model_output/classification")


main()
