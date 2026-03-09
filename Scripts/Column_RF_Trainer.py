import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train(csv_name):
    # =====================================
    # 1. LOAD DATASET
    # =====================================
    df = pd.read_csv(csv_name)
    print(f"Dataset loaded: {df.shape}")

    # Detect trays dynamically
    n_trays = df["Number_of_Trays"][0]
    # =====================================
    # 2. FEATURE ENGINEERING & SELECTION
    # =====================================
    features = ["R", "pf", "Tf", "steam_factor", "T_Feed_true", "T_Top_true", "T_Bottom_true", "X_Feed_true", "X_Top_true", "X_Bottom_true"]
    #features += [f"T_{i}_true" for i in range(n_trays) if f"T_{i}_true" in df.columns]         #All trays have temp sensors
    #features += [f"X_{i}_true" for i in range(n_trays) if f"X_{i}_true" in df.columns]         #All trays have composition sensors

    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    # =====================================
    # 3. MODEL 1: OPERATIONAL STATUS (Binary)
    # =====================================
    yo = df["operational"]
    X_train_o, X_test_o, yo_train, yo_test = train_test_split(
        X_scaled_df, yo, test_size=0.2, random_state=42, stratify=yo
    )

    op_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=42
    )
    op_model.fit(X_train_o, yo_train)

    print("\n--- OPERATIONAL STATUS REPORT ---")
    print(classification_report(yo_test, op_model.predict(X_test_o)))

    # =====================================
    # 4. MODEL 2: FAULT CLASSIFIER (Multi-class)
    # =====================================
    yf = df["fault_class"]
    X_train_f, X_test_f, yf_train, yf_test = train_test_split(
        X_scaled_df, yf, test_size=0.2, random_state=42, stratify=yf
    )

    fault_model = RandomForestClassifier(
        n_estimators=400, max_depth=20, class_weight="balanced", n_jobs=-1, random_state=42
    )
    fault_model.fit(X_train_f, yf_train)

    print("\n--- FAULT TYPE REPORT ---")
    print(classification_report(yf_test, fault_model.predict(X_test_f)))

    # =====================================
    # 5. MODEL 3: FAULT LOCATION (Specialized)
    # =====================================
    fault_indices = df[df["fault_location"] != -1].index
    X_loc = X_scaled_df.iloc[fault_indices]
    y_loc = df.loc[fault_indices, "fault_location"]

    X_train_l, X_test_l, yl_train, yl_test = train_test_split(
        X_loc, y_loc, test_size=0.2, random_state=42, stratify=y_loc
    )

    loc_model = RandomForestClassifier(
        n_estimators=500, max_depth=25, class_weight="balanced", n_jobs=-1, random_state=42
    )
    loc_model.fit(X_train_l, yl_train)

    print("\n--- FAULT LOCATION REPORT (Active Faults Only) ---")
    print(classification_report(yl_test, loc_model.predict(X_test_l)))

    # =====================================
    # 6. SAVE ARTIFACTS
    # =====================================
    joblib.dump(op_model, "Models/operational_classifier.pkl")
    joblib.dump(fault_model, "Models/fault_classifier.pkl")
    joblib.dump(loc_model, "Models/fault_locator.pkl")
    joblib.dump(scaler, "Models/column_scaler.pkl")
    print("\nAll models and scaler saved successfully.")