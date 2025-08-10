import numpy as np
import math

def betas_for_alpha_bar(num_timesteps: int, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i+1) / num_timesteps
        betas.append(min(
            1 - alpha_bar(t2) / alpha_bar(t1), max_beta
        ))
    return np.array(betas)


def get_named_beta_schedule(schedule_name: str, num_timesteps: int):
    if schedule_name == "linear":
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)

    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise ValueError(f"Unknown schedule_name: {schedule_name}")


