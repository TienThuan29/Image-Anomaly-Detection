import logging
from diffusion.model.ucdir import DY3h
from diffusion.model.diffusion import ResiGaussianGuideDY

logger = logging.getLogger("base")

def define_G(
        in_channel: int,
        out_channel: int,
        inner_channel: int,
        norm_groups: int,
        channel_mults: list,
        attn_res: list,
        res_blocks: int,
        dropout_p: float,
        image_size: int,
        channels: int,
        loss_type: str,
        with_noise_level_emb=True,
        resname='ResnetBlockDY3h'
):
    # Unet denoise
    denoise_fn = DY3h(
        in_channel=in_channel,
        out_channel=out_channel,
        inner_channel=inner_channel,
        norm_groups=norm_groups,
        channel_mults=channel_mults,
        attn_res=attn_res,
        res_blocks=res_blocks,
        dropout_p=dropout_p,
        image_size=image_size,
        with_noise_level_emb=with_noise_level_emb,
        resname=resname
    )

    # diffusion wrapper
    netG = ResiGaussianGuideDY(
        denoise_fn=denoise_fn,
        image_size=image_size,
        channels=channels,
        loss_type=loss_type
    )

    logger.info("**model net G %s %s", denoise_fn.__class__.__name__, netG.__class__.__name__)
    return netG



