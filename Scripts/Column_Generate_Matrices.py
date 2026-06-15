import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def generate_graphs(csv_name):
# Load the artifacts
    scaler = joblib.load("Models/column_scaler.pkl")
    op_model = joblib.load("Models/operational_classifier.pkl")
    fault_model = joblib.load("Models/fault_classifier.pkl")
    loc_model = joblib.load("Models/fault_locator.pkl")

    # Load dataset
    df = pd.read_csv(csv_name)

    # Feature list (must match training)
    n_trays = df["Number_of_Trays"][0]

    features = ["R", "pf", "Tf", "steam_factor", "T_Feed_true", "T_Top_true", "T_Bottom_true", "X_Feed_true", "X_Top_true", "X_Bottom_true"]
    #features += [f"T_{i}_true" for i in range(n_trays) if f"T_{i}_true" in df.columns]    #Temp sensor on each tray
    #features += [f"X_{i}_true" for i in range(n_trays) if f"X_{i}_true" in df.columns]    #Composition sensor on each tray

    X_scaled = scaler.transform(df[features])

    def plot_cm(y_true, y_pred, title, labels, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title(title, fontsize=15)
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved: {filename}")
        plt.close()

    # --- Matrix 1: Operational Status ---
    yo = df["operational"]
    _, X_test_o, _, yo_test = train_test_split(X_scaled, yo, test_size=0.2, random_state=42, stratify=yo)
    yo_pred = op_model.predict(X_test_o)
    plot_cm(yo_test, yo_pred, "Operational Status Confusion Matrix (Eugene Sanderson)", ["Normal (0)", "Fault (1)"], "Pictures/cm_operational.png")

    # --- Matrix 2: Fault Type ---
    yf = df["fault_class"]
    _, X_test_f, _, yf_test = train_test_split(X_scaled, yf, test_size=0.2, random_state=42, stratify=yf)
    yf_pred = fault_model.predict(X_test_f)
    fault_labels = ["Normal", "Low Reflux", "Low Steam", "Fouled Tray", "Damaged Tray", "Missing Tray"] 
    plot_cm(yf_test, yf_pred, "Fault Type Confusion Matrix (Eugene Sanderson)", fault_labels, "Pictures/cm_fault_type.png")

    # --- Matrix 3: Fault Location ---
    df_faults = df[df["fault_location"] != -1]
    X_loc_scaled = scaler.transform(df_faults[features])
    y_loc = df_faults["fault_location"]
    _, X_test_l, _, yl_test = train_test_split(X_loc_scaled, y_loc, test_size=0.2, random_state=42, stratify=y_loc)
    yl_pred = loc_model.predict(X_test_l)
    tray_labels = [f"Tray {i}" for i in range(n_trays)]
    plot_cm(yl_test, yl_pred, "Fault Location Confusion Matrix (By Tray) (Eugene Sanderson)", tray_labels, "Pictures/cm_location.png")
    