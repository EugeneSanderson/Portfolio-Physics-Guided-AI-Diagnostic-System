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
        "Tf_Noisy",
        "steam_factor_Noisy",
        "T_Bottom_Noisy",
        "T_Top_Noisy",
        "X_Feed_Noisy",
        "X_Bottom_Noisy",
        "X_Top_Noisy",
    ]
    X = df[features]

    # =====================================
    # Outputs (true column state)
    # =====================================

    targets = []
    targets += ["Flow_Feed_true"]
    targets += ["Flow_Top_true"]
    targets += ["Flow_Bottom_true"]
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

    print("Training flow state model...")

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

    joblib.dump(model, "Models/Flow_predictor.pkl")
    joblib.dump(scaler_X, "Models/Flow_predictorX.pkl")
    joblib.dump(scaler_Y, "Models/Flow_predictorY.pkl")

    print("State model saved.")

    return mae