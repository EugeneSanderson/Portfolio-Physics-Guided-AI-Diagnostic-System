import joblib
import numpy as np
import pandas as pd

def run(csv_name):
    Append_Dataset = True

    print("---------------------------")
    print("Loading Flow Rates model...")

    model = joblib.load("Models/Flow_predictor.pkl")
    scaler_X = joblib.load("Models/Flow_predictorX.pkl")
    scaler_Y = joblib.load("Models/Flow_predictorY.pkl")

    print("Loading dataset...")
    df = pd.read_csv(csv_name)

    features = [
        "R_Noisy",
        "pf_Noisy",
        "Tf_Noisy",
        "steam_factor_Noisy",
        "T_Bottom_Noisy",
        "T_Top_Noisy",
        "X_Feed_Noisy",
        "X_Bottom_Noisy",
        "X_Top_Noisy",
    ]

    X_Input = df[features]

    print("Scaling inputs...")
    X = scaler_X.transform(X_Input)

    print("Predicting flows profile...")
    pred = model.predict(X)
    pred = scaler_Y.inverse_transform(pred)

    if Append_Dataset:
        print("Appending predictions to dataset...")
        col_index = 0
        # Feed / Bottom / Top
        df["Flow_Feed_pred"] = pred[:, col_index]
        col_index += 1
        df["Flow_Top_pred"] = pred[:, col_index]
        col_index += 1
        df["Flow_Bottom_pred"] = pred[:, col_index]
        col_index += 1

    output_file = csv_name

    df.to_csv(output_file, index=False)

    print("\nDataset saved:", output_file)
    print("Script complete.")
