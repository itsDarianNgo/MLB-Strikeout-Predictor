import pandas as pd
import os
import h2o
from data_preprocessing import clean_data, merge_data
from feature_engineering import generate_features, drop_unnecessary_columns

# Initialize the H2O environment
h2o.init()

# Define the path to your model
script_dir = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(script_dir, "../model_output/StackedEnsemble_BestOfFamily_1_AutoML_4_20230705_53623")


def load_new_data():
    # Get the absolute path of the directory the script is in
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Define the directories where the data files are located
    batting_data_dir = os.path.join(script_dir, "../data/new/BattingData/")
    pitching_data_dir = os.path.join(script_dir, "../data/new/PitchingData/")
    game_data_dir = os.path.join(script_dir, "../data/new/GameData/")

    # Load the data
    batting_data = pd.concat(
        [pd.read_csv(os.path.join(batting_data_dir, f), encoding="ISO-8859-1") for f in os.listdir(batting_data_dir)], ignore_index=True
    )
    pitching_data = pd.concat(
        [pd.read_csv(os.path.join(pitching_data_dir, f), encoding="ISO-8859-1") for f in os.listdir(pitching_data_dir)], ignore_index=True
    )
    game_data = pd.concat([pd.read_csv(os.path.join(game_data_dir, f), encoding="ISO-8859-1") for f in os.listdir(game_data_dir)], ignore_index=True)

    # Clean the data
    batting_data, pitching_data, game_data = clean_data(batting_data, pitching_data, game_data)

    # Merge the datasets
    new_data = merge_data(batting_data, pitching_data, game_data)

    return new_data


def main():
    # Load new data
    new_data = load_new_data()

    # Load model
    model = h2o.load_model(MODEL_PATH)

    # Drop unnecessary columns
    new_data = drop_unnecessary_columns(new_data)

    # Generate the features
    new_data = generate_features(new_data)

    # Convert the preprocessed new_data to H2O Frame
    new_data_h2o = h2o.H2OFrame(new_data)

    # Define the predictors
    predictors = new_data_h2o.columns
    predictors.remove("SO_y")

    # Predict on the new data
    preds = model.predict(new_data_h2o)

    # Convert the H2O frame to a pandas dataframe
    preds = preds.as_data_frame()

    # Rename the column of predictions in preds dataframe to 'predictedStrikeouts'
    preds.rename(columns={preds.columns[0]: "predictedStrikeouts"}, inplace=True)

    # Concatenate the actual target variable and the predictions
    final_preds = pd.concat([new_data[["Date", "Player", "SO_y"]].reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    # Define the path for saving the csv file
    predictions_file_path = os.path.join(script_dir, "../data/predictions.csv")

    # Save the dataframe to a csv file
    final_preds.to_csv(predictions_file_path, index=False)
    print(f"Predictions saved to {predictions_file_path}")


if __name__ == "__main__":
    main()
