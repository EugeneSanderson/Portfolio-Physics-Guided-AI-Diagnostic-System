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
        "R_Noisy",
        "pf_Noisy",
        "steam_factor_Noisy",
        "X_Feed_Noisy",
        "X_Bottom_Noisy",
        "X_Top_Noisy",
        "Flow_Feed_Noisy",
        "Flow_Top_Noisy",
        "Flow_Bottom_Noisy"
    ]

    X = df[features]

    # =====================================
    # Outputs (true column state)
    # =====================================

    targets = []
    targets += ["Tf"]
    targets += ["T_Bottom_true"]
    targets += ["T_Top_true"]
    targets += [f"T_{i}_true" for i in range(n_trays) if f"T_{i}_true" in df.columns]
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

    print("Training temperature state model...")

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

    joblib.dump(model, "Models/Temperature_predictor.pkl")
    joblib.dump(scaler_X, "Models/Temperature_predictorX.pkl")
    joblib.dump(scaler_Y, "Models/Temperature_predictorY.pkl")

    print("State model saved.")

    return mae
