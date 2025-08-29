import torch
import numpy as np
import random
import os
from config import load_config
from vae_trainer import train_vae

config = load_config()

# Training mode
_is_vae_training = config.training.is_vae_training
_is_diffu_training = config.training.is_diffu_training
_training_category = config.training.category

_seed = config.general.seed

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    set_seed(42)
    if _is_vae_training:
        print(f"Start VAE training for category: {_training_category}")
        train_vae()

    # if _is_diffu_training:
    #     train_diffusion()
