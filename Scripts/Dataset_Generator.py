import numpy as np
import pandas as pd
import copy
import random
from Scripts.Column_Model import SetupColumn, RunColumn

def generate(S_params,dataset_ranges,N_SAMPLES,csv_name):
    Xf = S_params["Xf"]
    X1f = S_params["X1f"]
    pf = S_params["pf"]
    Tf = S_params["Tf"]
    Xd = S_params["Xd"]
    X1d = S_params["X1d"]
    Xb = S_params["Xb"]
    X1b = S_params["X1b"]
    R = S_params["R"]
    RectifyingTrays_Setup = S_params["RectifyingTrays_Setup"]
    StrippingTrays_Setup = S_params["StrippingTrays_Setup"]
    SetupColumn(S_params)
    # =========================================
    # CONFIGURATION
    # =========================================

    R_MIN = dataset_ranges["R_MIN"]
    R_MAX = dataset_ranges["R_MAX"]

    STEAM_MIN = dataset_ranges["STEAM_MIN"]
    STEAM_MAX = dataset_ranges["STEAM_MAX"]

    ZF_CENTER = X1f
    ZF_SPAN = X1f * dataset_ranges["Process_Deviation"]   # +- 5%

    NOISE_T_STD = 0.2 
    NOISE_X_STD = 0.002
    NOISE_P_STD = 0.2

    T_FEED_CENTER = Tf
    T_FEED_SPAN = Tf * dataset_ranges["Process_Deviation"] # +- 5%  4.1

    P_FEED_CENTER = pf
    P_FEED_SPAN = pf * dataset_ranges["Process_Deviation"] # +- 5%  5.0

    FLOW_FEED_CENTER = Xf
    FLOW_FEED_SPAN = Xf * dataset_ranges["Process_Deviation"]  # +- 5%  0.05

    # Balanced fault distribution
    FAULT_DISTRIBUTION = {
        0: 0.20,   # Normal
        1: 0.15,   # Low Reflux
        2: 0.15,   # Low Steam
        3: 0.20,   # Fouled Tray
        4: 0.15,   # Damaged Tray
        5: 0.15    # Missing Tray
    }

    # Convert to cumulative for sampling
    fault_classes = list(FAULT_DISTRIBUTION.keys())
    fault_probs = list(FAULT_DISTRIBUTION.values())

    # =========================================
    # Efficiency Generator
    # =========================================

    def generate_efficiencies(n_rect, n_strip):

        base_eff = np.random.uniform(0.65, 0.85)

        E_rect = np.full(n_rect, base_eff)
        E_strip = np.full(n_strip, base_eff)

        return E_rect, E_strip


    # =========================================
    # Fault Injection
    # =========================================

    def inject_fault(fault_class, params, E_rect, E_strip):

        fault_location = -1

        if fault_class == 1:  # Low Reflux
            params["R"] = np.random.uniform(R_MIN, 1.3)

        elif fault_class == 2:  # Low Steam
            params["steam_factor"] = np.random.uniform(STEAM_MIN, 1.05)

        elif fault_class in [3, 4, 5]:

            total_trays = len(E_rect) + len(E_strip)
            fault_location = random.randint(0, total_trays - 1)

            if fault_location < len(E_strip):
                target = E_strip
                idx = fault_location
            else:
                target = E_rect
                idx = fault_location - len(E_strip)

            if fault_class == 3:   # Fouled
                target[idx] *= np.random.uniform(0.3, 0.6)

            elif fault_class == 4: # Damaged
                target[idx] *= np.random.uniform(0.05, 0.2)

            elif fault_class == 5: # Missing
                target[idx] = 0.01 # Pure 0.0 doesn't feel right for some reason

        return params, E_rect, E_strip, fault_location


    # =========================================
    # Feature Extraction
    # =========================================

    def extract_features(Rect, Strip, params,
                        fault_class,
                        fault_location,
                        operational,
                        F,
                        D,
                        B):

        # -----------------------------
        # TRUE COLUMN STATE (physics)
        # -----------------------------
        T_true = np.array(
            [tray["T"] for tray in Strip] +
            [tray["T"] for tray in Rect]
        )

        X_true = np.array(
            [tray["X"] for tray in Strip] +
            [tray["X"] for tray in Rect]
        )

        # -----------------------------
        # MEASURED SENSOR VALUES
        # -----------------------------
        T_profile_Noisy = T_true.copy()
        X_profile_Noisy = X_true.copy()

        # -----------------------------
        # BASE FEATURES - TRUE SENSOR VALUES
        # -----------------------------
        row = {
            "R": params["R"],
            "steam_factor": params["steam_factor"],
            "zF": params["X1f"],
            "pf": params["pf"],
            "Tf": params["Tf"],
            "fault_class": fault_class,
            "fault_location": fault_location,
            "operational": int(operational),
        }

        row["Flow_Feed_true"] = F
        row["Flow_Top_true"] = D
        row["Flow_Bottom_true"] = B

        if not operational:
            return None            #Remove to test for an operational column

        # Store True Values
        # True temperatures
        for i, val in enumerate(T_true):
            if i == 0:
                row["T_Feed_true"] = val
            elif i == (len(Strip)-1):
                row["T_Bottom_true"] = val
            elif i == (len(T_true)-1):
                row["T_Top_true"] = val
            else:
                row[f"T_{i}_true"] = val
        # True compositions
        for i, val in enumerate(X_true):
            if i == 0:
                row["X_Feed_true"] = val
            elif i == (len(Strip)-1):
                row["X_Bottom_true"] = val
            elif i == (len(T_true)-1):
                row["X_Top_true"] = val
            else:
                row[f"X_{i}_true"] = val

        # -----------------------------
        # Noisy Sensors
        # -----------------------------
        T_profile_Noisy = T_profile_Noisy + np.random.normal(0, NOISE_T_STD, len(T_profile_Noisy))
        T_profile_Noisy = np.clip(T_profile_Noisy, 0.0, 200)
        X_profile_Noisy = X_profile_Noisy + np.random.normal(0, NOISE_X_STD, len(X_profile_Noisy))
        X_profile_Noisy = np.clip(X_profile_Noisy, 0.0, 200)
        F_Noisy = F + np.random.normal(0,F*0.05)
        F_Noisy = np.clip(F_Noisy, 0.0, 2.0)
        D_Noisy = D + np.random.normal(0,D*0.05)
        D_Noisy = np.clip(D_Noisy, 0.0, 2.0)
        B_Noisy = B + np.random.normal(0,B*0.05)
        B_Noisy = np.clip(B_Noisy, 0.0, 2.0)



        row["pf_Noisy"] = params["pf"] + np.random.normal(0,NOISE_P_STD)
        row["R_Noisy"] = R + np.random.normal(0,R*0.05)
        row["Tf_Noisy"] = params["Tf"] + np.random.normal(0,params["Tf"]*0.05)
        row["steam_factor_Noisy"] = params["steam_factor"] + np.random.normal(0,params["steam_factor"]*0.05)
        row["Flow_Feed_Noisy"] = F_Noisy
        row["Flow_Top_Noisy"] = D_Noisy
        row["Flow_Bottom_Noisy"] = B_Noisy

        for i, val in enumerate(T_profile_Noisy):
            if i == 0:
                row["T_Feed_Noisy"] = val
            elif i == (len(Strip)-1):
                row["T_Bottom_Noisy"] = val
            elif i == (len(T_profile_Noisy)-1):
                row["T_Top_Noisy"] = val
            else:
                row[f"T_{i}_Noisy"] = val

        for i, val in enumerate(X_profile_Noisy):
            if i == 0:
                row["X_Feed_Noisy"] = val
            elif i == (len(Strip)-1):
                row["X_Bottom_Noisy"] = val
            elif i == (len(T_profile_Noisy)-1):
                row["X_Top_Noisy"] = val
            else:
                row[f"X_{i}_Noisy"] = val

        return row



    # =========================================
    # MAIN GENERATION LOOP
    # =========================================

    dataset = []

    for i in range(N_SAMPLES):

        # ----- Base random parameters -----
        R = np.random.uniform(R_MIN, R_MAX)
        steam = np.random.uniform(STEAM_MIN, STEAM_MAX)
        zF = np.random.uniform(ZF_CENTER - ZF_SPAN,
                            ZF_CENTER + ZF_SPAN)
        Tf = np.random.uniform(T_FEED_CENTER - T_FEED_SPAN,
                            T_FEED_CENTER + T_FEED_SPAN)
        pf = np.random.uniform(P_FEED_CENTER - P_FEED_SPAN,
                            P_FEED_CENTER + P_FEED_SPAN)
        Xf = np.random.uniform(FLOW_FEED_CENTER - FLOW_FEED_SPAN,
                            FLOW_FEED_CENTER + FLOW_FEED_SPAN)
        params = {
            "Xf": Xf,
            "X1f": zF,
            "X1d": 0.8,
            "X1b": 0.05,
            "pf": pf,
            "Tf": Tf,
            "R": R,
            "steam_factor": steam
        }

        Rect_copy = copy.deepcopy(RectifyingTrays_Setup)
        Strip_copy = copy.deepcopy(StrippingTrays_Setup)

        E_rect, E_strip = generate_efficiencies(
            len(Rect_copy),
            len(Strip_copy)
        )

        # ----- Balanced fault selection -----
        fault_class = np.random.choice(
            fault_classes,
            p=fault_probs
        )

        params, E_rect, E_strip, fault_location = \
            inject_fault(fault_class,
                        params,
                        E_rect,
                        E_strip)

        # ----- Run column -----
        Rect, Strip, operational, reason, F, D, B = RunColumn(
            Rect_copy,
            Strip_copy,
            params,
            E_rect,
            E_strip
        )

        row = extract_features(
            Rect,
            Strip,
            params,
            fault_class,
            fault_location,
            operational,
            F,
            D,
            B,
        )
        if row is not None:
            row["Number_of_Trays"] = len(Rect_copy)+len(Strip_copy)
            dataset.append(row)

        if i % 100 == 0:
            print(f"Generated {i} samples")

    # =========================================
    # SAVE
    # =========================================

    df = pd.DataFrame(dataset)
    df.to_csv(csv_name, index=False)

    print("Dataset generation complete.")
