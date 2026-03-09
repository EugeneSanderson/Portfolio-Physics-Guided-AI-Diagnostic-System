import joblib
import numpy as np
import pandas as pd

def run(csv_name):
    Append_Dataset = False

    print("---------------------------")
    print("Loading composition model...")

    model = joblib.load("Models/Fault_predictor.pkl")
    scaler_X = joblib.load("Models/Fault_predictorX.pkl")
    scaler_Y = joblib.load("Models/Fault_predictorY.pkl")

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
        if f"X_{i}_res" in df.columns:
            features += [f"X_{i}_pred"]

    for i in range(n_trays):
        if f"T_{i}_res" in df.columns:
            features += [f"T_{i}_pred"]        

    X_Input = df[features]

    print("Scaling inputs...")
    X = scaler_X.transform(X_Input)

    print("Predicting fault profile...")
    pred = model.predict(X)
    pred = scaler_Y.inverse_transform(pred)

    fault_class = {
        "0": "Normal",
        "1": "Low Reflux",
        "2": "Low Steam",
        "3": "Fouled Tray",
        "4": "Damaged Tray",
        "5": "Missing Tray",
    }
    fault = fault_class[str(min(round(pred[0][1]),5))]
    if str(max(min(round(pred[0][1]),5),0)) not in fault_class:
        fault = "Unknown"
    if (pred[0][0]) < 0 or (pred[0][0]) > n_trays:
        fault_location = "None"
        if fault == fault_class["3"] or fault == fault_class["4"] or fault == fault_class["5"]:
            fault = "None"
    else:
        if fault == fault_class["0"] or fault == fault_class["1"] or fault == fault_class["2"]:
            fault_location = "None"
        else:
            fault_location = (pred[0][0])
    print(f"Fault Class: {fault}")
    print(f"Fault Location: {fault_location}")
    df["Fault_Class"] = fault
    df["Fault_Location"] = fault_location

    output_file = csv_name

    df.to_csv(output_file, index=False)

    print("\nDataset saved:", output_file)
    print("Script complete.")