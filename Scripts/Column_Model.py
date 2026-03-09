import Scripts.Thermodynamics as Thermodynamics
import numpy as np
import copy

# --- 1. Initial Conditions ---


def solve_material_balance(F, xF, xD, xB):

    D = F * (xF - xB) / (xD - xB)
    B = F - D
    
    return D, B

def SetupColumn(S_params):
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
    # Calculate q-line and Internal Flows
    _, _, T_bp_f = Thermodynamics.generate_dynamic_vle_from_x(pf, [X1f])
    cp_l_f = Thermodynamics.cp_liquid(Tf + 273.15, X1f) 
    hv_f = Thermodynamics.get_rigorous_latent_heat(T_bp_f[0], X1f, X1f) 
    q = 1 + (cp_l_f * (T_bp_f[0] - Tf) / hv_f)

    L_rect = R * Xd
    V_rect = L_rect + Xd
    L_strip = L_rect + (q * Xf)
    V_strip = V_rect - (1 - q) * Xf

    # --- 2. WORK UPWARD (Feed -> Distillate) ---
    curr_x = X1f
    while curr_x < X1d:
        _, y_sol, t_sol = Thermodynamics.generate_dynamic_vle_from_x(pf, [curr_x])
        curr_y = y_sol[0]
        RectifyingTrays_Setup.append({"X": curr_x, "Y": curr_y, "T": t_sol[0]})
        if curr_y >= X1d: break
        # Operating Line: Find liquid from tray above
        curr_x = (curr_y * V_rect - Xd * X1d) / L_rect

    # --- 3. WORK DOWNWARD (Feed -> Bottoms) ---
    # Start with the vapor rising from the tray below the feed
    curr_y = (L_strip / V_strip) * X1f - (Xb / V_strip) * X1b

    while True:
        # VLE Step: Given vapor y, find liquid x in equilibrium
        _, x_sol, t_sol = Thermodynamics.generate_dynamic_vle_from_y(pf, [curr_y])
        curr_x = x_sol[0]
        StrippingTrays_Setup.append({"X": curr_x, "Y": curr_y, "T": t_sol[0]})
        if curr_x <= X1b: break
        # Operating Line: Find vapor rising from tray below
        curr_y = (L_strip / V_strip) * curr_x - (Xb / V_strip) * X1b
    
    return(RectifyingTrays_Setup , StrippingTrays_Setup)

def RunColumn(Rect_Trays, Strip_Trays, params, E_rect_array, E_strip_array, Iterations=50, tol=1e-7):

    ContinueIteration = True
    operational = True
    reason = "None"
    F   = params["Xf"]
    zF  = params["X1f"]
    xD_guess = params["X1d"]
    xB_guess = params["X1b"]
    pf  = params["pf"]
    Tf  = params["Tf"]
    R   = params["R"]
    steam_factor = params.get("steam_factor", 1.0)
    if len(E_rect_array) != len(Rect_Trays):
        raise ValueError("Length of E_rect_array must equal number of rectifying trays")

    if len(E_strip_array) != len(Strip_Trays):
        raise ValueError("Length of E_strip_array must equal number of stripping trays")

    D, B = solve_material_balance(F, zF, xD_guess, xB_guess)

    for outer in range(Iterations):
        if ContinueIteration:
            D_old = D
            B_old = B

            # =================================
            # 1) Feed thermal condition
            # =================================
            _, _, T_bp_f = Thermodynamics.generate_dynamic_vle_from_x(pf, [zF])
            cp_l_f = Thermodynamics.cp_liquid(Tf + 273.15, zF)
            hv_f   = Thermodynamics.get_rigorous_latent_heat(T_bp_f[0], zF, zF)

            q = 1 + (cp_l_f * (T_bp_f[0] - Tf) / hv_f)

            # =================================
            # 2) Internal flows
            # =================================
            L_rect = R * D
            V_rect = L_rect + D

            L_strip = L_rect + q * F
            V_strip_design = V_rect - (1 - q) * F

            # =================================
            # 3) Steam / Reboiler energy
            # =================================
            T_bottom_guess = Strip_Trays[-1]["T"]
            x_bottom_guess = Strip_Trays[-1]["X"]

            dh_vap_bottom = Thermodynamics.get_backbone_latent_heat(
                T_bottom_guess,
                x_bottom_guess
            )

            Qr_design = V_strip_design * dh_vap_bottom
            Qr = steam_factor * Qr_design

            V_strip = Qr / dh_vap_bottom

            # Hydraulic constraint
            LV_max = 5.0
            if L_strip / V_strip > LV_max:
                V_strip = L_strip / LV_max

            # =================================
            # 4) STRIPPING SECTION
            # =================================

            # Use CURRENT B for operating line
            xB_current = Strip_Trays[-1]["X"]

            # Initial vapor from reboiler
            curr_y = (L_strip / V_strip) * (zF - xB_current)
            curr_y = max(2e-8, min(0.99999, curr_y))

            for j in range(len(Strip_Trays)):
                if ContinueIteration:
                    _, x_sol, t_sol = Thermodynamics.generate_dynamic_vle_from_y(pf, [curr_y])
                    x_eq = x_sol[0]

                    if j == 0:
                        x_in = zF
                    else:
                        x_in = Strip_Trays[j-1]["X"]

                    # Murphree efficiency
                    E_strip = max(0.0, min(1.0, E_strip_array[j]))
                    curr_x = x_in + E_strip * (x_eq - x_in)
                    if curr_y <= 2e-8 or curr_y > 0.999999:
                        operational = False
                        reason = "Stripping broke"
                        ContinueIteration = False
                    curr_x = max(1e-8, min(0.999999, curr_x))

                    Strip_Trays[j] = {
                        "X": curr_x,
                        "Y": curr_y,
                        "T": t_sol[0]
                    }

                    # Stripping operating line
                    curr_y = (L_strip / V_strip) * (curr_x - xB_current)
                    curr_y = max(1e-8, min(0.999999, curr_y))

            # =================================
            # 5) RECTIFYING SECTION
            # =================================

            curr_x = Strip_Trays[0]["X"]

            for j in range(len(Rect_Trays)):
                if ContinueIteration:
                    _, y_sol, t_sol = Thermodynamics.generate_dynamic_vle_from_x(pf, [curr_x])
                    y_eq = y_sol[0]

                    if j == 0:
                        y_in = Strip_Trays[0]["Y"]
                    else:
                        y_in = Rect_Trays[j-1]["Y"]

                    E_rect = max(0.0, min(1.0, E_rect_array[j]))
                    curr_y = y_in + E_rect * (y_eq - y_in)
                    curr_y = max(1e-8, min(0.999999, curr_y))

                    Rect_Trays[j] = {
                        "X": curr_x,
                        "Y": curr_y,
                        "T": t_sol[0]
                    }

                    # Rectifying operating line
                    xD_actual = Rect_Trays[-1]["Y"]
                    if curr_x <= 2e-8 or curr_y > 0.999999:
                        operational = False
                        reason = "Reflux broke"
                        ContinueIteration = False
                    curr_x = ((R + 1) / R) * curr_y - (xD_guess / R)
                    curr_x = max(1e-8, min(0.999999, curr_x))

            # =================================
            # 6) MASS BALANCE CLOSURE
            # =================================
            xD_actual = Rect_Trays[-1]["Y"]
            xB_actual = Strip_Trays[-1]["X"]

            D_new, B_new = solve_material_balance(F, zF, xD_actual, xB_actual)

            alpha = 0.3   # relaxation factor ( This part!!! This part almost broke me!!! Goddamn numerical instability galore without this )

            D = D_old + alpha * (D_new - D_old)
            B = B_old + alpha * (B_new - B_old)

            # Convergence check
            if abs(D - D_old) < tol and abs(B - B_old) < tol:
                break

    return Rect_Trays, Strip_Trays, operational, reason, F, D, B