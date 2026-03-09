import joblib
import numpy as np
import pandas as pd

def run(csv_name):
    Append_Dataset = True

    print("---------------------------")
    print("Loading composition model...")

    model = joblib.load("Models/Composition_predictor.pkl")
    scaler_X = joblib.load("Models/Composition_predictorX.pkl")
    scaler_Y = joblib.load("Models/Composition_predictorY.pkl")

    print("Loading dataset...")
    df = pd.read_csv(csv_name)
    n_trays = df["Number_of_Trays"][0]
    features = [
        "R_Noisy",
        "pf_Noisy",
        "Tf_Noisy",
        "steam_factor_Noisy",
        "T_Bottom_Noisy",
        "T_Top_Noisy",
        "Flow_Feed_Noisy",
        "Flow_Top_Noisy",
        "Flow_Bottom_Noisy"
    ]

    X_Input = df[features]

    print("Scaling inputs...")
    X = scaler_X.transform(X_Input)

    print("Predicting composition profile...")
    pred = model.predict(X)
    pred = scaler_Y.inverse_transform(pred)

    if Append_Dataset:
        print("Appending predictions to dataset...")
        col_index = 0
        # Feed / Bottom / Top
        df["X_Feed_pred"] = pred[:, col_index]
        col_index += 1
        df["X_Bottom_pred"] = pred[:, col_index]
        col_index += 1
        df["X_Top_pred"] = pred[:, col_index]
        col_index += 1
        # Internal trays
        for tray in range(n_trays):
            if col_index >= pred.shape[1]:
                break
            df[f"X_{tray}_pred"] = pred[:, col_index]
            col_index += 1


    output_file = csv_name

    df.to_csv(output_file, index=False)

    print("\nDataset saved:", output_file)
    print("Script complete.")
