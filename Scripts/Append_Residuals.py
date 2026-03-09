import joblib
import numpy as np
import pandas as pd

def run(csv_name):
    Append_Dataset = True

    print("---------------------------")
    print("Compiling residuals")


    print("Loading dataset...")
    df = pd.read_csv(csv_name)

    if Append_Dataset:    
        print("Appending residuals to dataset...")
        # Flows
        df["Flow_profile1"] = df["Flow_Feed_pred"] - df["Flow_Top_pred"]
        df["Flow_profile2"] = df["Flow_Feed_pred"] - df["Flow_Bottom_pred"]
        df["Flow_profile3"] = df["Flow_Top_pred"] - df["Flow_Bottom_pred"]
        # Temperatures
        df["T_profile1"] = df["T_Bottom_pred"] - df["T_Top_pred"]
        df["T_profile2"] = df["T_Bottom_pred"] - df["Tf_pred"]
        df["T_profile3"] = df["Tf_pred"] - df["T_Top_pred"]
        # Compositions
        df["X_profile1"] = df["X_Top_pred"] - df["X_Bottom_pred"]
        df["X_profile2"] = df["X_Top_pred"] - df["X_Feed_pred"]
        df["X_profile3"] = df["X_Feed_pred"] - df["X_Bottom_pred"]



    output_file = csv_name

    df.to_csv(output_file, index=False)

    print("\nDataset saved:", output_file)
    print("Script complete.")
