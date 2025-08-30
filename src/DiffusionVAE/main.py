import torch
import numpy as np
import random
import os
from config import load_config
from vae_trainer import train_vae
from diffusion_trainer import train_diffusion

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
    set_seed(_seed)
    if _is_vae_training:
        print('training vae...')
        train_vae()

    if _is_diffu_training:
        print('training diffusion...')
        train_diffusion()
