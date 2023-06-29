import h2o
import pandas as pd
from feature_engineering import generate_features
from data_preprocessing import clean_data, merge_data
import os


def load_data():
    """Load the dataset from CSV files."""
    # Get the absolute path of the directory the script is in
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Define the directories where the data files are located
    batting_data_dir = os.path.join(script_dir, "../data/new/BattingData/")
    pitching_data_dir = os.path.join(script_dir, "../data/new/PitchingData/")
    game_data_dir = os.path.join(script_dir, "../data/new/GameData/")

    # Load and concatenate the data from all CSV files in each directory
    batting_data = pd.concat(
        [pd.read_csv(os.path.join(batting_data_dir, file), encoding="ISO-8859-1") for file in os.listdir(batting_data_dir) if file.endswith(".csv")],
        ignore_index=True,
    )
    pitching_data = pd.concat(
        [
            pd.read_csv(os.path.join(pitching_data_dir, file), encoding="ISO-8859-1")
            for file in os.listdir(pitching_data_dir)
            if file.endswith(".csv")
        ],
        ignore_index=True,
    )
    game_data = pd.concat(
        [pd.read_csv(os.path.join(game_data_dir, file), encoding="ISO-8859-1") for file in os.listdir(game_data_dir) if file.endswith(".csv")],
        ignore_index=True,
    )

    return batting_data, pitching_data, game_data


def load_model():
    """Load the trained H2O model."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, "../model_output/StackedEnsemble_BestOfFamily_1_AutoML_3_20230628_22344")
    return h2o.load_model(model_path)


def prepare_data(data):
    """Prepare the data for prediction."""
    batting_data, pitching_data, game_data = data
    batting_data, pitching_data, game_data = clean_data(batting_data, pitching_data, game_data)
    final_data = merge_data(batting_data, pitching_data, game_data)

    feature_columns = [
        "AB",
        "R_x",
        "H_x",
        "RBI",
        "BB_x",
        "SO_x",
        "PA",
        "BA",
        "OBP",
        "SLG",
        "OPS",
        "Pit_x",
        "Str_x",
        "WPA_x",
        "aLI_x",
        "WPA+",
        "WPA-",
        "cWPA_x",
        "acLI_x",
        "RE24_x",
        "PO",
        "A",
        "IP",
        "H_y",
        "R_y",
        "ER",
        "BB_y",
        "SO_y",
        "HR",
        "ERA",
        "BF",
        "Pit_y",
        "Str_y",
        "Ctct",
        "StS",
        "StL",
        "GB",
        "FB",
        "LD",
        "Unk",
        "GSc",
        "IR",
        "IS",
        "WPA_y",
        "aLI_y",
        "cWPA_y",
        "acLI_y",
        "RE24_y",
    ]
    stats = ["mean"]  # as used in feature_engineering.py

    final_data = generate_features(final_data, feature_columns, stats)

    return h2o.H2OFrame(final_data)


def predict_strikeouts(model, data):
    """Predict the number of strikeouts."""
    predictions = model.predict(data)
    return predictions


def main():
    h2o.init()

    # load the model
    model = load_model()

    # prepare the data
    data = load_data()
    h2o_data = prepare_data(data)
    original_data = h2o_data.as_data_frame()  # save the original data with actual values

    # make predictions
    predictions = predict_strikeouts(model, h2o_data)
    predictions_df = predictions.as_data_frame()  # convert predictions to pandas dataframe

    # Join the original data with the predictions
    results = original_data.join(predictions_df)

    # Save the dataframe to a csv file
    results.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")


if __name__ == "__main__":
    main()
