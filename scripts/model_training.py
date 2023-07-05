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

    # Define the target column
    target = "SO_y"

    # Define the predictors
    predictors = final_data_h2o.columns
    predictors.remove(target)

    # Calculate the train/test split index
    split_index = int(final_data_h2o.shape[0] * 0.7)

    # Split the data into training and testing sets
    train = final_data_h2o[:split_index, :]
    test = final_data_h2o[split_index:, :]

    # Initialize the H2O AutoML model
    aml = H2OAutoML(max_models=3, seed=1, max_runtime_secs=1200)

    # Train the model
    aml.train(x=predictors, y=target, training_frame=train)

    # View the AutoML Leaderboard
    lb = aml.leaderboard
    print(lb.head(rows=lb.nrows))

    # Predict on the test data
    preds = aml.leader.predict(test)

    # View the performance of the leader model on the test data
    perf = aml.leader.model_performance(test)
    print(perf)

    # For each model in the leaderboard (excluding Stacked Ensembles), print the variable importance
    model_ids = list(aml.leaderboard["model_id"].as_data_frame().iloc[:, 0])
    for mid in model_ids:
        if "StackedEnsemble" not in mid:
            m = h2o.get_model(mid)
            print(f"\nModel ID: {mid}")
            print(m.varimp(use_pandas=True))

    # Save the model
    h2o.save_model(aml.leader, path="./model_output")


if __name__ == "__main__":
    main()
