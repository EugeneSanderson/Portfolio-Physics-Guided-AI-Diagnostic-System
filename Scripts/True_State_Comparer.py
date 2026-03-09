import joblib
import numpy as np
import pandas as pd

def run(csv_name):
    Append_Dataset = True

    print("---------------------------")
    print("Loading composition model...")

    model = joblib.load("Models/True_State_predictor.pkl")
    scaler_X = joblib.load("Models/True_State_predictorX.pkl")
    scaler_Y = joblib.load("Models/True_State_predictorY.pkl")

    print("Loading dataset...")
    df = pd.read_csv(csv_name)
    n_trays = df["Number_of_Trays"][0]
    features = [
        # Flows
        "Flow_Feed_pred",
        "Flow_Top_pred",
        "Flow_Bottom_pred",
        "Flow_profile1",
        "Flow_profile2",
        "Flow_profile3",
        # Reflux
        "R_pred",
        # Pressure
        "pf_pred",
        # Steam
        "steam_factor_pred",
        # Temperatures
        "Tf_pred",
        "T_Top_pred",
        "T_Bottom_pred",
        "T_profile1",
        "T_profile2",
        "T_profile3",
        # Compositions
        "X_Feed_pred",
        "X_Top_pred",
        "X_Bottom_pred",
        "X_profile1",
        "X_profile2",
        "X_profile3",

    ]

    for i in range(n_trays):
        if f"X_{i}_pred" in df.columns:
            features += [f"X_{i}_pred"]

    for i in range(n_trays):
        if f"T_{i}_pred" in df.columns:
            features += [f"T_{i}_pred"]      

    X_Input = df[features]

    print("Scaling inputs...")
    X = scaler_X.transform(X_Input)

    print("Predicting fault profile...")
    pred = model.predict(X)
    pred = scaler_Y.inverse_transform(pred)

    if Append_Dataset:
        print("Appending predictions to dataset...")
        col_index = 0
        df["Flow_Feed_pred_True_State"] = pred[:, col_index]
        col_index += 1
        df["Flow_Top_pred_True_State"] = pred[:, col_index]
        col_index += 1
        df["Flow_Bottom_pred_True_State"] = pred[:, col_index]
        col_index += 1
        df["R_pred_True_State"] = pred[:, col_index]
        col_index += 1
        df["pf_pred_True_State"] = pred[:, col_index]
        col_index += 1
        df["steam_factor_pred_True_State"] = pred[:, col_index]
        col_index += 1
        df["Tf_pred_True_State"] = pred[:, col_index]
        col_index += 1
        df["T_Top_pred_True_State"] = pred[:, col_index]
        col_index += 1
        df["T_Bottom_pred_True_State"] = pred[:, col_index]
        col_index += 1
        df["X_Feed_pred_True_State"] = pred[:, col_index]
        col_index += 1
        df["X_Top_pred_True_State"] = pred[:, col_index]
        col_index += 1
        df["X_Bottom_pred_True_State"] = pred[:, col_index]
        col_index += 1
        # Internal trays
        for tray in range(n_trays):
            if col_index >= pred.shape[1]:
                break
            df[f"X_{tray}_pred_True_State"] = pred[:, col_index]
            col_index += 1
        for tray in range(n_trays):
            if col_index >= pred.shape[1]:
                break
            df[f"T_{tray}_pred_True_State"] = pred[:, col_index]
            col_index += 1

    output_file = csv_name

    df.to_csv(output_file, index=False)

    print("\nDataset saved:", output_file)
    print("Script complete.")