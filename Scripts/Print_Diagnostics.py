import matplotlib.pyplot as plt
import pandas as pd

def percentage_difference(actual, estimate):
    if actual == 0: return 0 # Avoid division by zero
    return abs((actual - estimate) / actual) * 100

def print_diagnostics(csv_name):
    print("Loading dataset...")

    df = pd.read_csv(csv_name)

    print("==== Flow Feed ====")
    print(f"Sensor feed flow reading:   {df["Flow_Feed_Noisy"][0]:.4f}")
    print(f"Predicted feed flow:        {df["Flow_Feed_pred_True_State"][0]:.4f}")
    print(f"Difference in feed flow:    {df["Flow_Feed_pred_True_State"][0] - df["Flow_Feed_Noisy"][0]:.4f}")
    print("==== Flow Distillate ====")
    print(f"Sensor distillate flow reading: {df["Flow_Top_Noisy"][0]:.4f}")
    print(f"Predicted distillate flow:      {df["Flow_Top_pred_True_State"][0]:.4f}")
    print(f"Difference in distillate flow:  {df["Flow_Top_pred_True_State"][0] - df["Flow_Top_Noisy"][0]:.4f}")
    print("==== Flow Bottom ====")
    print(f"Sensor bottom flow reading: {df["Flow_Bottom_Noisy"][0]:.4f}")
    print(f"Predicted bottom flow:      {df["Flow_Bottom_pred_True_State"][0]:.4f}")
    print(f"Difference in bottom flow:  {df["Flow_Bottom_pred_True_State"][0] - df["Flow_Bottom_Noisy"][0]:.4f}")

    print("\n==== Composition Feed ====")
    print(f"Sensor feed composition reading: {df["X_Feed_Noisy"][0]:.4f}")
    print(f"Predicted composition flow:      {df["X_Feed_pred_True_State"][0]:.4f}")
    print(f"Difference in composition flow:  {df["X_Feed_pred_True_State"][0] - df["X_Feed_Noisy"][0]:.4f}")
    print("==== Composition Top ====")
    print(f"Sensor feed composition reading: {df["X_Top_Noisy"][0]:.4f}")
    print(f"Predicted composition flow:      {df["X_Top_pred_True_State"][0]:.4f}")
    print(f"Difference in composition flow:  {df["X_Top_pred_True_State"][0] - df["X_Top_Noisy"][0]:.4f}")   
    print("==== Composition Bottom ====")
    print(f"Sensor feed composition reading: {df["X_Bottom_Noisy"][0]:.4f}")
    print(f"Predicted composition:           {df["X_Bottom_pred_True_State"][0]:.4f}")
    print(f"Difference in composition:       {df["X_Bottom_pred_True_State"][0] - df["X_Bottom_Noisy"][0]:.4f}")

    print("\n==== Temperature Feed ====")
    print(f"Sensor feed temperature reading: {df["Tf_Noisy"][0]:.4f}")
    print(f"Predicted temperature:           {df["Tf_pred_True_State"][0]:.4f}")
    print(f"Difference in temperature:       {df["Tf_pred_True_State"][0] - df["Tf_Noisy"][0]:.4f}")
    print("==== Temperature Top ====")
    print(f"Sensor top temperature reading: {df["T_Top_Noisy"][0]:.4f}")
    print(f"Predicted temperature:          {df["T_Top_pred_True_State"][0]:.4f}")
    print(f"Difference in temperature:      {df["T_Top_pred_True_State"][0] - df["T_Top_Noisy"][0]:.4f}")
    print("==== Temperature Bottom ====")
    print(f"Sensor bottom temperature reading: {df["T_Bottom_Noisy"][0]:.4f}")
    print(f"Predicted temperature:             {df["T_Bottom_pred_True_State"][0]:.4f}")
    print(f"Difference in temperature:         {df["T_Bottom_pred_True_State"][0] - df["T_Bottom_Noisy"][0]:.4f}")

    print("\n==== Reflux Ratio ====")
    print(f"Sensor reflux reading: {df["R_Noisy"][0]:.4f}")
    print(f"Predicted reflux:      {df["R_pred_True_State"][0]:.4f}")
    print(f"Difference in reflux:  {df["R_pred_True_State"][0] - df["R_Noisy"][0]:.4f}")

    print("\n==== Steam Factor ====")
    print(f"Sensor steam factor reading: {df["steam_factor_Noisy"][0]:.4f}")
    print(f"Predicted steam factor:      {df["steam_factor_pred_True_State"][0]:.4f}")
    print(f"Difference in steam factor:  {df["steam_factor_pred_True_State"][0] - df["steam_factor_Noisy"][0]:.4f}")

    print("\n\n=========== Sensor accuracy in % Difference (Measured vs Predicted) ===========")
    print("==== Flow Sensors ====")
    print(f"Feed sensor mismatch:       {percentage_difference(df["Flow_Feed_pred_True_State"][0], df["Flow_Feed_Noisy"][0]):.4f} %")
    print(f"Distillate sensor mismatch: {percentage_difference(df["Flow_Top_pred_True_State"][0], df["Flow_Top_Noisy"][0]):.4f} %")
    print(f"Bottom sensor mismatch:     {percentage_difference(df["Flow_Bottom_pred_True_State"][0], df["Flow_Bottom_Noisy"][0]):.4f} %")
    print("==== Composition Sensors ====")
    print(f"Feed sensor mismatch:       {percentage_difference(df["X_Feed_pred_True_State"][0], df["X_Feed_Noisy"][0]):.4f} %")
    print(f"Distillate sensor mismatch: {percentage_difference(df["X_Top_pred_True_State"][0], df["X_Top_Noisy"][0]):.4f} %")
    print(f"Bottom sensor mismatch:     {percentage_difference(df["X_Bottom_pred_True_State"][0], df["X_Bottom_Noisy"][0]):.4f} %")
    print("==== Temperature Sensors ====")
    print(f"Feed sensor mismatch:       {percentage_difference(df["Tf_pred_True_State"][0], df["Tf_Noisy"][0]):.4f} %")
    print(f"Distillate sensor mismatch: {percentage_difference(df["T_Top_pred_True_State"][0], df["T_Top_Noisy"][0]):.4f} %")
    print(f"Bottom sensor mismatch:     {percentage_difference(df["T_Bottom_pred_True_State"][0], df["T_Bottom_Noisy"][0]):.4f} %")
    print("==== Reflux Ratio Sensor ====")
    print(f"Sensor mismatch:            {percentage_difference(df["R_pred_True_State"][0], df["R_Noisy"][0]):.4f} %")
    print("==== Steam Factor Sensor ====")
    print(f"Sensor mismatch:            {percentage_difference(df["steam_factor_pred_True_State"][0], df["steam_factor_Noisy"][0]):.4f} %")

    print("\n\n=========== Fault Detection ===========")
    print(f"Fault type:     {df["Fault_Class"][0]}")
    print(f"Fault Location: {df["Fault_Location"][0]}")

def normalize_pair(noisy, pred):

    noisy_n = (noisy/pred)
    pred_n = (1)

    return noisy_n, pred_n

def plot_diagnostics(csv_name):

    df = pd.read_csv(csv_name)
    row = df.iloc[0]

    sensors = [
        "Feed Flow",
        "Top Flow",
        "Bottom Flow",
        "Feed Composition",
        "Top Composition",
        "Bottom Composition",
        "Feed Temp",
        "Top Temp",
        "Bottom Temp",
        "Reflux",
        "Steam"
    ]

    noisy = [
        row["Flow_Feed_Noisy"],
        row["Flow_Top_Noisy"],
        row["Flow_Bottom_Noisy"],
        row["X_Feed_Noisy"],
        row["X_Top_Noisy"],
        row["X_Bottom_Noisy"],
        row["Tf_Noisy"],
        row["T_Top_Noisy"],
        row["T_Bottom_Noisy"],
        row["R_Noisy"],
        row["steam_factor_Noisy"]
    ]

    pred = [
        row["Flow_Feed_pred_True_State"],
        row["Flow_Top_pred_True_State"],
        row["Flow_Bottom_pred_True_State"],
        row["X_Feed_pred_True_State"],
        row["X_Top_pred_True_State"],
        row["X_Bottom_pred_True_State"],
        row["Tf_pred_True_State"],
        row["T_Top_pred_True_State"],
        row["T_Bottom_pred_True_State"],
        row["R_pred_True_State"],
        row["steam_factor_pred_True_State"]
    ]

    noisy_norm = []
    pred_norm = []

    for n, p in zip(noisy, pred):
        n_n, p_n = normalize_pair(n, p)
        noisy_norm.append(n_n)
        pred_norm.append(p_n)


    mismatch = [percentage_difference(p, n) for p, n in zip(pred, noisy)]

    # =====================================
    # Plot 1: Sensor vs Predicted
    # =====================================

    x = range(len(sensors))

    plt.figure(figsize=(12,6))
    plt.bar(x, noisy_norm, width=0.4, label="Measured", align="center")
    plt.bar([i + 0.4 for i in x], pred_norm, width=0.4, label="Predicted")

    plt.xticks([i + 0.2 for i in x], sensors, rotation=45)
    plt.title("Measured vs Predicted Sensors (Normalized) (Eugene Sanderson)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Pictures/Sensor_pred1.png")
    plt.show()
    
    # =====================================
    # Plot 2: Sensor mismatch
    # =====================================

    plt.figure(figsize=(12,6))
    plt.bar(sensors, mismatch)

    plt.xticks(rotation=45)
    plt.ylabel("Mismatch (%)")
    plt.title("Sensor Mismatch (%)")
    plt.tight_layout()
    plt.savefig("Pictures/Sensor_pred2.png")
    plt.show()