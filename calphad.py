def get_calphad_equilibrium_area_fraction(temperature_celsius):
    # 根据 CALPHAD 标称模拟计算结果返回参考值
    if temperature_celsius >= 1000:
        return 0.010
    elif temperature_celsius >= 980:
        return 0.052
    elif temperature_celsius >= 960:
        return 0.065
    else:
        return 0.084
