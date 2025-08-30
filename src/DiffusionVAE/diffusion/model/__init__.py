import logging
logger = logging.getLogger('base')
from config import load_config

config = load_config()
# Unet diffusion config
_in_channel = config.diffusion_model.unet.in_channel
_out_channel = config.diffusion_model.unet.out_channel
_inner_channel = config.diffusion_model.unet.inner_channel
_norm_groups = config.diffusion_model.unet.norm_groups
_channel_mults = config.diffusion_model.unet.channel_mults
_attn_res = config.diffusion_model.unet.attn_res
_res_blocks = config.diffusion_model.unet.res_blocks
_dropout_p = float(config.diffusion_model.unet.dropout)

_image_size = config.diffusion_model.diffusion.image_size
_channels = config.diffusion_model.diffusion.channels
_loss_type = config.diffusion_model.loss_type


from diffusion.model.ddpm_model import DDPM
def create_model():
    model = DDPM(
        in_channel=_in_channel,
        out_channel=_out_channel,
        inner_channel=_inner_channel,
        norm_groups=_norm_groups,
        channel_mults=_channel_mults,
        attn_res=_attn_res,
        res_blocks=_res_blocks,
        dropout_p=_dropout_p,
        image_size=_image_size,
        channels=_channels,
        loss_type=_loss_type
    )
    logger.info('Model [{:s}] is created.'.format(model.__class__.__name__))
    return model