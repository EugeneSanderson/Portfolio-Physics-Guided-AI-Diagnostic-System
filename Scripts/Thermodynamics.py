import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- NRTL & Antoine Constants from Source ---
# Component 1: Ethanol, Component 2: Water
A12_nrtl = 1.66480 
A21_nrtl = 0.94010 

# Antoine: Log10(P_mmHg) = A - B / (T_C + C)
antoine_eth = {'A': 7.68117, 'B': 1332.04, 'C': 199.200} 
antoine_wat = {'A': 8.07131, 'B': 1730.63, 'C': 233.426} 

def get_psat(T_celsius):
    #Calculates Psat (mmHg) using source constants
    p1 = 10**(antoine_eth['A'] - antoine_eth['B'] / (T_celsius + antoine_eth['C']))
    p2 = 10**(antoine_wat['A'] - antoine_wat['B'] / (T_celsius + antoine_wat['C']))
    return p1, p2

def nrtl_gamma(x1):
    #NRTL activity coefficients from source implementation
    x2 = 1 - x1
    g1 = np.exp(A12_nrtl * (A21_nrtl * x2 / (A12_nrtl * x1 + A21_nrtl * x2))**2)
    g2 = np.exp(A21_nrtl * (A12_nrtl * x1 / (A21_nrtl * x2 + A12_nrtl * x1))**2)
    return g1, g2
#============================
# Physical Property Functions 
#============================

def molar_volume_liquid(T_k, x1):
    #DIPPR 105: Vm = A / B^(1 + (1 - T/C)^D) [m3/mol]
    # Params: [A, B, C, D]
    eth = [1.628, 0.273, 513.92, 0.236]
    wat = [5.459, 0.305, 647.13, 0.081]
    
    def calc_v(p, tk):
        return (p[0] / (p[1]**(1 + (1 - tk/p[2])**p[3])))**-1 / 1000
    return x1 * calc_v(eth, T_k) + (1-x1) * calc_v(wat, T_k)

def cp_liquid(T_k, x1):
    #Liquid Heat Capacity (J/mol-K)
    cp_eth = 112400 # Typical liquid Cp for Ethanol
    cp_wat = 276370 - 2090.1*T_k + 8.125*T_k**2 # Water DIPPR 100
    return (x1 * cp_eth + (1-x1) * cp_wat)/1000

def properties_vapor(T_k, P_pa, x1):
    #Vapor properties using Peng-Robinson EOS for Real Gas
    R = 8.314
    # Simple mixing rules for Tc, Pc, omega
    tc_mix = x1*513.9 + (1-x1)*647.1
    pc_mix = (x1*61.4 + (1-x1)*220.6) * 1e5
    w_mix = x1*0.644 + (1-x1)*0.344
    
    a = 0.45724 * (R**2 * tc_mix**2) / pc_mix
    b = 0.07780 * (R * tc_mix) / pc_mix
    
    # Solve for Vm_vapor: P = RT/(V-b) - a/(V^2 + 2bV - b^2)
    func = lambda V: (R*T_k/(V-b)) - (a/(V**2 + 2*b*V - b**2)) - P_pa
    vm_v = fsolve(func, R*T_k/P_pa)[0]
    
    # Vapor Cp (Ideal Gas + Residual) - simplified
    cp_v = x1*(38.4 + 0.16*T_k) + (1-x1)*(32.2 + 0.0019*T_k) 
    cv_v = cp_v - R
    
    return vm_v, cp_v, cv_v

def calculate_tray_enthalpy(T_k, x1, y1, state='liquid', T_ref=298.15):
    #Determines Molar Enthalpy (J/mol) and Heat Capacities.
    #Returns: (Enthalpy, Cp, Cv)
 
    # 1. Liquid Properties
    cp_l = cp_liquid(T_k, x1)
    h_liquid = cp_l * (T_k - T_ref)
    
    if state == 'liquid':
        return h_liquid, cp_l, cp_l 

    elif state == 'vapor':
        vm_v, cp_v, cv_v = properties_vapor(T_k, 101325, y1) 
        
        # Calculate Latent Heat at the tray temperature
        dh_vap = get_mix_latent_heat(T_k, x1, y1) 
        
        # Total Vapor Enthalpy
        h_v = h_liquid + dh_vap 
        
        return h_v, cp_v, cv_v

def get_rigorous_latent_heat(T_celsius, x1, y1):
    #Calculates Latent Heat (J/mol) including Watson T-scaling 
    #and NRTL Excess Enthalpy (He).

    T_k = T_celsius + 273.15
    R = 8.314
    
    # 1. Temperature-Scaled Pure Latent Heats (Watson)
    # Eth: Tc=513.9K, dH_nb=38560 @ 351.5K
    # Wat: Tc=647.1K, dH_nb=40660 @ 373.15K
    dh_v1 = 38560 * ((513.9 - T_k) / (513.9 - 351.5))**0.38
    dh_v2 = 40660 * ((647.1 - T_k) / (647.1 - 373.15))**0.38
    
    # 2. NRTL Excess Enthalpy (He)
    x2 = 1 - x1
    g1, g2 = nrtl_gamma(x1)
    # Standard approximation for Ethanol-Water heat of mixing
    h_excess = x1 * x2 * (A12_nrtl + A21_nrtl) * R * T_k / 4 
    
    return (y1 * dh_v1 + (1 - y1) * dh_v2) - h_excess

#=======================
# Dynamic VLE Generator
#=======================
def generate_dynamic_vle_from_x(P_kPa=101.325, x_range = np.arange(0, 1.01, 0.01)):
    #Solves Bubble Point T and y at a given Pressure
    P_mmHg = (760 / 101.325) * P_kPa
    y_vals, T_vals = [], []

    for x1 in x_range:
        # Objective: Psat1*x1*g1 + Psat2*x2*g2 = P_total
        def bubble_obj(T_c):
            ps1, ps2 = get_psat(T_c)
            g1, g2 = nrtl_gamma(x1)
            return (x1 * g1 * ps1) + ((1-x1) * g2 * ps2) - P_mmHg
        
        T_bp = fsolve(bubble_obj, 80)[0]
        T_vals.append(T_bp)
        
        # Calculate y1 = (x1*g1*Psat1) / P_total
        ps1, _ = get_psat(T_bp)
        g1, _ = nrtl_gamma(x1)
        y_vals.append((x1 * g1 * ps1) / P_mmHg)
        
    return x_range, y_vals, T_vals

def generate_dynamic_vle_from_y(P_kPa=101.325, y_range=np.arange(0, 1.01, 0.01)):
    #Solves Dew Point T and liquid composition x at a given Pressure.
    #Given vapor composition y, calculates corresponding liquid composition x
    #using a fully coupled nonlinear solution (T and x solved simultaneously).

    P_mmHg = (760 / 101.325) * P_kPa
    x_vals, T_vals = [], []

    for y1 in y_range:

        # Handle edge cases explicitly
        if y1 <= 0.0:
            x_vals.append(0.0)
            T_vals.append(get_psat(80)[1])
            continue

        if y1 >= 1.0:
            x_vals.append(1.0)
            T_vals.append(get_psat(80)[0])
            continue

        def equations(vars):
            T_c, x1 = vars

            ps1, ps2 = get_psat(T_c)
            g1, g2 = nrtl_gamma(x1)

            eq1 = x1 - (y1 * P_mmHg) / (g1 * ps1)
            eq2 = (1 - x1) - ((1 - y1) * P_mmHg) / (g2 * ps2)

            return [eq1, eq2]

        # Initial guesses
        T_guess = 80
        x_guess = y1

        T_dp, x1_sol = fsolve(equations, [T_guess, x_guess])

        T_vals.append(T_dp)
        x_vals.append(x1_sol)

    return y_range, x_vals, T_vals

def get_backbone_latent_heat(T_celsius, y1):
    T_k = T_celsius + 273.15

    dh_v1 = 38560 * ((513.9 - T_k) / (513.9 - 351.5))**0.38
    dh_v2 = 40660 * ((647.1 - T_k) / (647.1 - 373.15))**0.38

    return y1 * dh_v1 + (1 - y1) * dh_v2
