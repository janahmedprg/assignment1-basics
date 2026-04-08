import math

def learning_rate_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    Tw: int,
    Tc: int,
):
    if t < Tw:
        return (t / Tw) * alpha_max
    elif Tw <= t and t <= Tc:
        theta = math.pi * (t - Tw) / (Tc - Tw)
        diff = alpha_max - alpha_min
        return alpha_min + 0.5 * (1 + math.cos(theta)) * diff
    else:
        return alpha_min