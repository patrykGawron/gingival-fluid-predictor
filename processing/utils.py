import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def perform_processing(
        input_data: pd.DataFrame
) -> pd.DataFrame:
    # NOTE(MF): sample code
    # preprocessed_data = preprocess_data(input_data)
    # models = load_models()  # or load one model
    # please note, that the predicted data should be a proper pd.DataFrame with column names
    # predicted_data = predict(models, preprocessed_data)
    # return predicted_data
    preprocessed_data = preprocess_data(input_data)
    models = load_models()

    # for the simplest approach generate a random DataFrame with proper column names and size
    column_names = ['16-B', '16-P', '11-B', '11-P', '24-B', '24-P', '36-B', '36-P', '31-B', '31-P', '44-B', '44-P']
    predicted_data = pd.DataFrame(
        np.random.randint(low=0, high=100, size=(len(input_data.index), len(column_names))),
        columns=column_names
    )

    for model in models:
        prediction = model["model"].predict(preprocessed_data)
        predicted_data[model["name"]] = prediction


    return predicted_data

def preprocess_data(data):
    booleans = ["zgrzytanie", "zaciskanie", "sztywnosc", "ograniczone otwieranie", "bol miesni", "przygryzanie",
                "cwiczenia", "szyna", "starcie-przednie", "starcie-boczne", "ubytki klinowe", "pekniecia szkliwa",
                "impresje jezyka", "linea alba", "przerost zwaczy", "tkliwosc miesni"]
    interleukinas = ["Unnamed: 0",
                     "Interleukina – 11B",
                     "Interleukina – 11P",
                     "Interleukina – 16B",
                     "Interleukina – 16P",
                     "Interleukina – 24B",
                     "Interleukina – 24P",
                     "Interleukina – 31B",
                     "Interleukina – 31P",
                     "Interleukina – 36B",
                     "Interleukina – 36P",
                     "Interleukina – 44B",
                     "Interleukina – 44P"]

    # change str percentage columns to floats
    data["API"] = data["API"].apply(lambda x: float(x.replace("%", "")) / 100)
    data["SBI"] = data["SBI"].apply(lambda x: float(x.replace("%", "")) / 100)

    data.drop(columns=booleans, axis=1, inplace=True)
    data.drop(columns=interleukinas, axis=1, inplace=True)

    gender_map = {
        "m": 0,
        "k": 1
    }

    # change gender data to numbers
    data["plec"] = data["plec"].apply(lambda x: gender_map[x])
    return data

def load_models():
    y_cols = ["16-B", "16-P", "11-B", "11-P", "24-B", "24-P", "36-B", "36-P", "31-B", "31-P", "44-B", "44-P"]
    models = []
    for target in y_cols:
        model = {
            "model": pickle.load(open(f"./models/{target}_model.pickle", "rb")),
            "name": target
        }
        models.append(model)
    return models
