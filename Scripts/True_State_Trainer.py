import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

def train(params,csv_name):
    print("Loading dataset...")

    df = pd.read_csv(csv_name)

    # =====================================
    # Detect number of trays automatically
    # =====================================

    n_trays = df["Number_of_Trays"][0]

    # =====================================
    # Inputs (plant sensors)
    # =====================================

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


    X = df[features]

    # =====================================
    # Outputs (true column state)
    # =====================================

    targets = []

    targets += ["Flow_Feed_true"]
    targets += ["Flow_Top_true"]
    targets += ["Flow_Bottom_true"]
    targets += ["R"]
    targets += ["pf"]
    targets += ["steam_factor"]
    targets += ["Tf"]
    targets += ["T_Top_true"]
    targets += ["T_Bottom_true"]
    targets += ["zF"]
    targets += ["X_Top_true"]
    targets += ["X_Bottom_true"]

    for i in range(n_trays):
        if f"X_{i}_true" in df.columns:
            targets += [f"X_{i}_true"]

    for i in range(n_trays):
        if f"T_{i}_true" in df.columns:
            targets += [f"T_{i}_true"]

    Y = df[targets]

    # =====================================
    # Train / Test split
    # =====================================

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    # =====================================
    # Scaling
    # =====================================

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    Y_train = scaler_Y.fit_transform(Y_train)
    Y_test = scaler_Y.transform(Y_test)

    # =====================================
    # Neural Network Regressor
    # =====================================

    print("Training physics state model...")

    model = MLPRegressor(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        max_iter=params["max_iter"],
        learning_rate=params["learning_rate"],
        verbose=params["verbose"],
        tol=params["tol"]
    )

    model.fit(X_train, Y_train)

    # =====================================
    # Evaluation
    # =====================================

    Y_pred = model.predict(X_test)

    Y_pred = scaler_Y.inverse_transform(Y_pred)
    Y_test = scaler_Y.inverse_transform(Y_test)

    mae = mean_absolute_error(Y_test, Y_pred)

    print("\nState reconstruction MAE:", mae)

    # =====================================
    # Save model
    # =====================================

    joblib.dump(model, "Models/True_State_predictor.pkl")
    joblib.dump(scaler_X, "Models/True_State_predictorX.pkl")
    joblib.dump(scaler_Y, "Models/True_State_predictorY.pkl")

    print("State model saved.")

    return mae