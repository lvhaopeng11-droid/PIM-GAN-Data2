import math

def calculate_theoretical_arrhenius(T_celsius, Q_kj_mol=235.6):
    T_K = T_celsius + 273.15
    R = 8.314e-3
    return math.exp(-Q_kj_mol / (R * T_K))

def calculate_larson_miller(T_celsius, time_min, C=20.0):
    T_K = T_celsius + 273.15
    t_safe = max(time_min, 1.0)
    return (T_K / 1000.0) * (math.log10(t_safe) + C)
