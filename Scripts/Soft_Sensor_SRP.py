import joblib
import numpy as np
import pandas as pd

def run(csv_name):
    Append_Dataset = True

    print("---------------------------")
    print("Loading Steam / Reflux / Pressure model...")

    model = joblib.load("Models/SRP_predictor.pkl")
    scaler_X = joblib.load("Models/SRP_predictorX.pkl")
    scaler_Y = joblib.load("Models/SRP_predictorY.pkl")

    print("Loading dataset...")
    df = pd.read_csv(csv_name)

    features = [
        "Tf_Noisy",
        "T_Bottom_Noisy",
        "T_Top_Noisy",
        "X_Feed_Noisy",
        "X_Bottom_Noisy",
        "X_Top_Noisy",
        "Flow_Feed_Noisy",
        "Flow_Top_Noisy",
        "Flow_Bottom_Noisy"
    ]

    X_Input = df[features]

    print("Scaling inputs...")
    X = scaler_X.transform(X_Input)

    print("Predicting SRP profile...")
    pred = model.predict(X)
    pred = scaler_Y.inverse_transform(pred)

    if Append_Dataset:
        print("Appending predictions to dataset...")
        col_index = 0
        # Feed / Bottom / Top
        df["R_pred"] = pred[:, col_index]
        col_index += 1
        df["pf_pred"] = pred[:, col_index]
        col_index += 1
        df["steam_factor_pred"] = pred[:, col_index]
        col_index += 1

    output_file = csv_name

    df.to_csv(output_file, index=False)

    print("\nDataset saved:", output_file)
    print("Script complete.")
