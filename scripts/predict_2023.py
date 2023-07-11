import pandas as pd
import h2o
from h2o.automl import H2OAutoML

h2o.init()


def load_data(data_file):
    data = pd.read_csv(data_file)
    return data


def apply_model_regression(model, data):
    h2o_frame = h2o.H2OFrame(data)
    predictions = model.predict(h2o_frame)
    h2o_frame["SO_y_pred"] = predictions
    h2o_frame["SO_y_binary"] = (h2o_frame["SO_y_pred"] > h2o_frame["SO_Prop"]).ifelse(1, 0)
    return h2o_frame


def apply_model_classification(model, data):
    predictions = model.predict(data)
    predictions.set_names(["SO_y_binary_pred", "p0", "p1"])
    predictions["SO_y_binary_pred"] = predictions["SO_y_binary_pred"].asnumeric()
    data = data.cbind(predictions)
    return data


def main():
    regression_model = h2o.load_model("./model_output/regression/GLM_1_AutoML_23_20230710_13724")
    classification_model = h2o.load_model("./model_output/classification/GLM_1_AutoML_24_20230710_14002")

    data = load_data("./data/final_data_with_features_2023.csv")

    data_with_regression_preds = apply_model_regression(regression_model, data)
    data_with_all_preds = apply_model_classification(classification_model, data_with_regression_preds)

    data_with_all_preds["binary_prediction_correct"] = (
        (data_with_all_preds["SO_y_binary_pred"] == 1) & (data_with_all_preds["SO_y"] > data_with_all_preds["SO_Prop"])
        | (data_with_all_preds["SO_y_binary_pred"] == 0) & (data_with_all_preds["SO_y"] <= data_with_all_preds["SO_Prop"])
    ).ifelse("Yes", "No")

    data_with_all_preds = data_with_all_preds[
        ["Date", "Player", "SO_y", "SO_y_pred", "SO_Prop", "SO_y_binary", "SO_y_binary_pred", "binary_prediction_correct", "p0", "p1"]
    ]

    data_with_all_preds = data_with_all_preds.as_data_frame()
    data_with_all_preds.to_csv("./model_output/unseen_predictions.csv", index=False)


if __name__ == "__main__":
    main()
