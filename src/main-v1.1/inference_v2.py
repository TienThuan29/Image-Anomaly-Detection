import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC
from config import load_config
from utils import load_mvtec_test_dataset, load_vae
from diffusion_model import UNetModel
from diffusion_gaussian import GaussianDiffusion

def to_batch_tensor(x_like, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(x_like, torch.Tensor):
        t = x_like
    elif isinstance(x_like, (list, tuple)):
        if len(x_like) == 0:
            raise ValueError("images_reconstructed is empty.")
        first = x_like[0]
        if isinstance(first, torch.Tensor) and first.dim() == 4:
            # list các bước [K,B,C,H,W] -> average theo bước
            t = torch.stack(x_like, dim=0).mean(dim=0)
        elif isinstance(first, torch.Tensor) and first.dim() == 3:
            # list theo batch [B*[C,H,W]] -> [B,C,H,W]
            t = torch.stack(x_like, dim=0)
        else:
            raise TypeError(f"Unsupported element in list: dim={getattr(first,'dim',lambda:None)()}")
    else:
        raise TypeError(f"Unsupported type for images_reconstructed: {type(x_like)}")
    return t.to(device=ref.device, dtype=ref.dtype)


def compute_anomaly_map(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return (x - x_recon).abs().mean(dim=1) # Tính mean giữa 3 channels


def normalize_maps_global(anomaly_maps: List[torch.Tensor]) -> List[torch.Tensor]:
    stacked = torch.cat(anomaly_maps, dim=0)
    amin = stacked.min()
    amax = stacked.max()
    if (amax - amin) < 1e-8:
        return [torch.zeros_like(m) for m in anomaly_maps]
    return [(m - amin) / (amax - amin) for m in anomaly_maps]


def eval_auroc_image(labels: List[int], scores: List[float]) -> float:
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def eval_auroc_pixel(anomaly_maps: List[torch.Tensor], gt_masks: List[torch.Tensor]) -> float:
    norm_maps = normalize_maps_global(anomaly_maps)
    pred = torch.cat([m.flatten() for m in norm_maps], dim=0).cpu()
    gt = torch.cat([g.flatten() for g in gt_masks], dim=0).cpu().bool()
    metric = AUROC(task="binary")
    return float(metric(pred, gt))


def to_label_list(labels_field) -> List[int]:
    if isinstance(labels_field, list):
        return [int(x) for x in labels_field]
    if torch.is_tensor(labels_field):
        if labels_field.ndim == 0:
            return [int(labels_field.item())]
        return [int(x) for x in labels_field.tolist()]
    return [int(labels_field)]


@torch.no_grad()
def guided_reconstruction(
    x: torch.Tensor,
    y0: torch.Tensor,
    unet: torch.nn.Module,
    gaussian: GaussianDiffusion,
    t_start: int,
    skip: int,
    w: float = 0.5,
    eta: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Conditional DDIM-like reconstruction guided towards condition image y0.
    Returns final x_0 estimate.
    """
    def extract(arr: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        out = arr[t]
        return out.view(-1, *((1,) * (len(shape) - 1))).expand(shape)

    B = x.size(0)
    t0 = torch.full((B,), t_start, dtype=torch.long, device=device)
    a_t0 = extract(gaussian.alphas_cumprod, t0, x.shape)
    x_t = a_t0.sqrt() * x + (1.0 - a_t0).sqrt() * torch.randn_like(x)

    steps = list(range(0, max(t_start, 1), max(1, skip)))
    seq = list(reversed(steps))

    for idx, i in enumerate(seq):
        j = seq[idx + 1] if idx + 1 < len(seq) else -1
        t = torch.full((B,), i, dtype=torch.long, device=device)
        if j >= 0:
            t_next = torch.full((B,), j, dtype=torch.long, device=device)
            a_next = extract(gaussian.alphas_cumprod, t_next, x.shape)
        else:
            a_next = torch.ones_like(a_t0)
        a_t = extract(gaussian.alphas_cumprod, t, x.shape)

        eps = unet(x_t, t)
        y_t = a_t.sqrt() * y0 + (1.0 - a_t).sqrt() * eps
        eps_hat = eps - (1.0 - a_t).sqrt() * w * (y_t - x_t)
        x0_t = (x_t - eps_hat * (1.0 - a_t).sqrt()) / a_t.sqrt()

        c1 = eta * (((1.0 - a_t / a_next) * (1.0 - a_next) / (1.0 - a_t)).clamp(min=0.0)).sqrt()
        c2 = ((1.0 - a_next) - c1 ** 2).clamp(min=0.0).sqrt()
        noise = torch.randn_like(x)
        x_t = a_next.sqrt() * x0_t + c1 * noise + c2 * eps_hat

    return x0_t.clamp(0, 1)


@torch.no_grad()
def run_evaluation():
    config = load_config(os.path.join(os.path.dirname(__file__), 'config.yml'))

    # general
    image_size = config.general.image_size
    batch_size = config.general.batch_size
    input_channels = config.general.input_channels
    output_channels = config.general.output_channels
    cuda_idx = config.general.cuda
    device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")

    # data
    category_name = config.data.category
    mvtec_data_dir = config.data.mvtec_data_dir

    # vae
    vae_name = config.vae_model.name if hasattr(config, 'vae_model') else config.vae.name
    z_dim = config.vae_model.z_dim if hasattr(config, 'vae_model') else config.diffusion_model.z_dim
    dropout_p = config.vae_model.dropout_p if hasattr(config, 'vae_model') else config.diffusion_model.dropout_p
    backbone = getattr(config.vae_model, 'backbone', 'resnet18') if hasattr(config, 'vae_model') else getattr(config.vae, 'backbone', 'resnet18')

    # testing paths
    diff_test_cfg = getattr(config, 'diffusion_testing', None)
    if diff_test_cfg is None:
        raise ValueError("diffusion_testing section not found in config.yml")

    diffusion_model_path = diff_test_cfg.diffusion_model_path
    vae_model_path = diff_test_cfg.diffusion_vae_model_path if hasattr(diff_test_cfg, 'diffusion_vae_model_path') else diff_test_cfg.vae_model_path
    if not diffusion_model_path or not os.path.exists(diffusion_model_path):
        raise FileNotFoundError(f"Invalid diffusion_model_path: {diffusion_model_path}")
    if not vae_model_path or not os.path.exists(vae_model_path):
        raise FileNotFoundError(f"Invalid vae_model_path: {vae_model_path}")

    # diffusion settings
    num_timesteps = config.diffusion_model.num_timesteps
    beta_schedule = config.diffusion_model.beta_schedule

    # v2 guidance hyperparameters (with safe defaults)
    guidance_w = getattr(diff_test_cfg, 'guidance_w', 0.5)
    infer_t_ratio = float(getattr(diff_test_cfg, 'infer_t_ratio', 0.4))
    infer_skip = int(getattr(diff_test_cfg, 'infer_skip', max(1, int(0.05 * num_timesteps))))
    infer_eta = float(getattr(diff_test_cfg, 'infer_eta', 0.0))

    print(f"[INFO] Evaluating (v2 guided) category={category_name} on device={device}")
    print(f"[INFO] VAE: {vae_name}, backbone={backbone}")
    print(f"[INFO] Guidance: w={guidance_w}, t_ratio={infer_t_ratio}, skip={infer_skip}, eta={infer_eta}")

    vae = load_vae_model(
        checkpoint_path=vae_model_path,
        vae_name=vae_name,
        in_channels=input_channels,
        latent_dim=z_dim,
        out_channels=output_channels,
        backbone=backbone,
        dropout_p=dropout_p,
        image_size=image_size,
        device=device,
    )

    unet = load_diffusion_unet(
        checkpoint_path=diffusion_model_path,
        image_size=image_size,
        in_channels=input_channels,
        device=device,
    )
    gaussian = GaussianDiffusion(num_timesteps=num_timesteps, beta_schedule=beta_schedule, device=str(device))

    # dataset
    test_loader = load_mvtec_test_dataset(
        dataset_root_dir=mvtec_data_dir,
        category=category_name,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    # Accumulators
    labels_all: List[int] = []
    img_scores_all: List[float] = []
    maps_all: List[torch.Tensor] = []
    gts_all: List[torch.Tensor] = []

    # t start
    t_start = max(1, int(infer_t_ratio * num_timesteps))

    print("[INFO] Running guided inference v2...")
    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['label']

        recon_vae, _, _ = vae(images)

        recon = guided_reconstruction(
            x=images,
            y0=recon_vae,
            unet=unet,
            gaussian=gaussian,
            t_start=t_start,
            skip=infer_skip,
            w=guidance_w,
            eta=infer_eta,
            device=device,
        )

        B = images.size(0)
        anomaly_map = compute_anomaly_map(images, recon)
        image_scores = anomaly_map.view(B, -1).max(dim=1).values

        labels_all.extend(to_label_list(labels))
        img_scores_all.extend(image_scores.detach().cpu().tolist())
        maps_all.extend(list(anomaly_map.detach().cpu()))
        gts_all.extend(list(masks.detach().cpu().squeeze(1)))

    img_auroc = eval_auroc_image(labels_all, img_scores_all)
    px_auroc = eval_auroc_pixel(maps_all, gts_all)

    print(f"\n[RESULT v2] Image AUROC: {img_auroc:.4f}")
    print(f"[RESULT v2] Pixel AUROC: {px_auroc:.4f}")


if __name__ == '__main__':
    run_evaluation()


#
# def load_vae_model(
#     checkpoint_path: str,
#     vae_name: str,
#     in_channels: int,
#     latent_dim: int,
#     out_channels: int,
#     backbone: str,
#     dropout_p: float,
#     image_size: int,
#     device: torch.device,
# ):
#     if vae_name == 'vae_resnet':
#         model = VAEResNet(
#             image_size=image_size,
#             in_channels=in_channels,
#             latent_dim=latent_dim,
#             out_channels=out_channels,
#             resnet_name=backbone,
#             dropout_p=dropout_p,
#         ).to(device)
#     elif vae_name == 'vae_unet':
#         model = VAEUnet(
#             in_channels=in_channels,
#             latent_dim=latent_dim,
#             out_channels=out_channels,
#         ).to(device)
#     else:
#         raise ValueError(f"Unknown vae model: {vae_name}")
#
#     ckpt = torch.load(checkpoint_path, map_location=device)
#     state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     if missing or unexpected:
#         print(f"[WARN] VAE missing keys: {missing}, unexpected: {unexpected}")
#     model.eval()
#     return model
