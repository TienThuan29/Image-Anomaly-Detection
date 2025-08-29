import torch
from diffusion.model.base_model import BaseModel


class DDPM(BaseModel):
    def __init__(
            self
    ):
        super(DDPM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


