import torch

def get_optimizer(optimizer_name: str, params, **kwargs) -> torch.optim.Optimizer:
    if optimizer_name == "Adam":
        return torch.optim.Adam(params, **kwargs)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(params, **kwargs)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(params, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")