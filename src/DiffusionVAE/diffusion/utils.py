import torch
from diffusion.model.ddpm_model import DDPM

def load_diffusion_model(
    in_channel: int,
    inner_channel: int ,
    out_channel: int,
    norm_groups: int,
    channel_mults: list,
    attn_res: list,
    res_blocks: int,
    dropout_p: float,
    image_size: int,
    channels: int,
    loss_type: str,
    diffusion_model_path: str,
    device
):
    diffusion_model = DDPM(
        in_channel=in_channel,
        out_channel=out_channel,
        inner_channel=inner_channel,
        norm_groups=norm_groups,
        channel_mults=channel_mults,
        attn_res=attn_res,
        res_blocks=res_blocks,
        dropout_p=dropout_p,
        image_size=image_size,
        channels=channels,
        loss_type=loss_type
    )

    checkpoint = torch.load(diffusion_model_path, map_location=device, weights_only=False)
    diffusion_model.netG.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.netG.to(device)
    diffusion_model.netG.eval()

    return diffusion_model

