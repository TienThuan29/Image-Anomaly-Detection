import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import MvTecDataset
from vae_resnet_model import VAEResNet
from vae_unet_model import VAEUnet
from diffusion_model import UNetModel
import inspect

def _add_numpy_allowlist_for_safe_unpickler():
    """Cho phép một số kiểu numpy phổ biến khi dùng weights_only=True (PyTorch 2.6)."""
    try:
        import numpy as np
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            np.float32, np.float64, np.int32, np.int64,
            np.bool_, np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.uint64
        ])
    except Exception:
        # Không có add_safe_globals ở phiên bản hiện tại hoặc lỗi khác -> bỏ qua
        pass

def _torch_load_compat(path: str, map_location, allow_fallback_to_untrusted: bool = True):
    """
    Ưu tiên load an toàn với weights_only=True (PyTorch 2.6).
    Nếu fail do safe-unpickler (vd numpy dtype), sẽ thử allowlist rồi thử lại.
    Cuối cùng, nếu vẫn fail và bạn cho phép (allow_fallback_to_untrusted=True),
    sẽ fallback sang weights_only=False (CẢNH BÁO: có rủi ro bảo mật nếu file không tin cậy).
    """
    sig = inspect.signature(torch.load)
    supports_weights_only = 'weights_only' in sig.parameters

    # 1) Thử safe load
    try:
        if supports_weights_only:
            return torch.load(path, map_location=map_location, weights_only=True)
        else:
            return torch.load(path, map_location=map_location)
    except Exception as e1:
        # 2) Thử allowlist numpy và thử lại chế độ safe
        _add_numpy_allowlist_for_safe_unpickler()
        try:
            if supports_weights_only:
                return torch.load(path, map_location=map_location, weights_only=True)
            else:
                return torch.load(path, map_location=map_location)
        except Exception as e2:
            # 3) Fallback: chỉ dùng khi bạn TIN TƯỞNG checkpoint
            if supports_weights_only and allow_fallback_to_untrusted:
                print("[WARN] Secure loading failed twice. Falling back to weights_only=False "
                      "(ONLY do this for trusted checkpoints).")
                try:
                    return torch.load(path, map_location=map_location, weights_only=False)
                except Exception as e3:
                    raise RuntimeError(
                        f"Failed to load checkpoint safely and fallback also failed.\n"
                        f"Safe error #1: {e1}\nSafe error #2 (after allowlist): {e2}\n"
                        f"Fallback error: {e3}"
                    )
            # Nếu không cho phép fallback:
            raise RuntimeError(
                f"Failed to load checkpoint with safe mode and fallback disabled.\n"
                f"Safe error #1: {e1}\nSafe error #2 (after allowlist): {e2}"
            )

def _extract_state_dict(ckpt: dict):
    """
    Chấp nhận nhiều dạng checkpoint khác nhau:
    - {'model_state_dict': ...} hoặc {'state_dict': ...} hoặc trực tiếp state_dict
    - Tự bỏ tiền tố 'module.' (nn.DataParallel)
    """
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
            sd = ckpt['model_state_dict']
        elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            sd = ckpt['state_dict']
        else:
            # Có thể ckpt chính là state_dict
            sd = ckpt
    else:
        sd = ckpt

    if isinstance(sd, dict):
        sd = {
            (k.replace('module.', '', 1) if isinstance(k, str) and k.startswith('module.') else k): v
            for k, v in sd.items()
        }
    return sd
# ------------------------------------------------------------


""" Load pre-trained diffusion unet """
@torch.no_grad()
def load_diffusion_unet(
    checkpoint_path: str,
    image_size: int,
    in_channels: int,
    device: torch.device,
):
    model = UNetModel(
        img_size=image_size,
        base_channels=32,
        n_heads=2,
        num_res_blocks=2,
        dropout=0.1,
        attention_resolutions="32,16,8",
        biggan_updown=True,
        in_channels=in_channels,
    ).to(device)

    # DÙNG loader tương thích 2.6
    ckpt = _torch_load_compat(checkpoint_path, map_location=device, allow_fallback_to_untrusted=True)
    state = _extract_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] Diffusion UNet missing keys: {missing}, unexpected: {unexpected}")
    model.eval()
    return model


""" Load pre-trained vae model """
@torch.no_grad()
def load_vae(
        checkpoint_path: str,
        vae_name: str,
        input_channels: int,
        output_channels: int,
        z_dim: int,
        backbone: str,
        dropout_p: float,
        image_size: int,
        device: torch.device
):
    if vae_name == 'vae_resnet':
        print("VAE model name: vae_resnet")
        model = VAEResNet(
            image_size=image_size,
            in_channels=input_channels,
            out_channels=output_channels,
            latent_dim=z_dim,
            resnet_name=backbone,
            dropout_p=dropout_p
        ).to(device)
    elif vae_name == 'vae_unet':
        print("VAE model name: vae_unet")
        model = VAEUnet(
            in_channels=input_channels,
            latent_dim=z_dim,
            out_channels=output_channels
        ).to(device)
    else:
        raise ValueError(f"Unknown vae model: {vae_name}")

    # DÙNG loader tương thích 2.6 (thử safe trước, rồi cảnh báo nếu fallback)
    ckpt = _torch_load_compat(checkpoint_path, map_location=device, allow_fallback_to_untrusted=True)
    state = _extract_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] VAE missing keys: {missing}, Unexpected keys: {unexpected}")

    model.eval()
    return model

# """ Load pre-trained diffusion unet """
# @torch.no_grad()
# def load_diffusion_unet(
#     checkpoint_path: str,
#     image_size: int,
#     in_channels: int,
#     device: torch.device,
# ):
#     model = UNetModel(
#         img_size=image_size,
#         base_channels=32,
#         n_heads=2,
#         num_res_blocks=2,
#         dropout=0.1,
#         attention_resolutions="32,16,8",
#         biggan_updown=True,
#         in_channels=in_channels,
#     ).to(device)
#
#     ckpt = torch.load(checkpoint_path, map_location=device)
#     state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     if missing or unexpected:
#         print(f"[WARN] Diffusion UNet missing keys: {missing}, unexpected: {unexpected}")
#     model.eval()
#     return model
#
#
# """ Load pre-trained vae model """
# @torch.no_grad()
# def load_vae(
#         checkpoint_path: str,
#         vae_name: str,
#         input_channels: int,
#         output_channels: int,
#         z_dim: int,
#         backbone: str,
#         dropout_p: float,
#         image_size: int,
#         device: torch.device
# ):
#     if vae_name == 'vae_resnet':
#         print("VAE model name: vae_resnet")
#         model = VAEResNet(
#             image_size=image_size,
#             in_channels=input_channels,
#             out_channels=output_channels,
#             latent_dim=z_dim,
#             resnet_name=backbone,
#             dropout_p=dropout_p
#         ).to(device)
#     elif vae_name == 'vae_unet':
#         print("VAE model name: vae_unet")
#         model = VAEUnet(
#             in_channels=input_channels,
#             latent_dim=z_dim,
#             out_channels=output_channels
#         ).to(device)
#     else:
#         raise ValueError(f"Unknown vae model: {vae_name}")
#     try:
#         import numpy as np
#         safe_globals = [
#             np.core.multiarray.scalar,
#             np.ndarray,
#             np.dtype,
#             np.float32, np.float64, np.int32, np.int64,
#             np.bool_, np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.uint64
#         ]
#         torch.serialization.add_safe_globals(safe_globals)
#
#         ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
#         print(f"[INFO] Successfully loaded {checkpoint_path} with weights_only=True")
#
#     except Exception as e:
#         # If weights_only=True still fails, use the fallback but warn about security
#         print(f"[WARN] Secure loading failed, using fallback (potential security risk)")
#         print(f"[WARN] Error details: {str(e)[:200]}...")  # Truncate long error messages
#         ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
#
#     # Handle both direct state_dict and nested checkpoint formats
#     if 'model_state_dict' in ckpt:
#         state = ckpt['model_state_dict']
#     elif 'state_dict' in ckpt:
#         state = ckpt['state_dict']
#     else:
#         state = ckpt  # Assume it's a direct state_dict
#
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     if missing or unexpected:
#         print(f"[WARN] Missing keys: {missing}, Unexpected keys: {unexpected}")
#
#     model.eval()
#     return model


""" Dataloader for MVTEC """
def load_mvtec_train_dataset(
        dataset_root_dir: str,
        category: str,
        image_size: int,
        batch_size: int,
        num_workers: int = 2,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False
):
    train_dataset = MvTecDataset(
        dataset_root_dir=dataset_root_dir,
        category=category,
        image_size=image_size,
        is_train=True,
        is_mask=True,
        use_cutout=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return train_loader


def load_mvtec_test_dataset(
        dataset_root_dir: str,
        category: str,
        image_size: int,
        batch_size: int,
        num_workers: int = 2,
        shuffle: bool = False,
        pin_memory: bool = True,
        drop_last: bool = False
):
    test_dataset = MvTecDataset(
        dataset_root_dir=dataset_root_dir,
        category=category,
        image_size=image_size,
        is_train=False,
        is_mask=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return test_loader


def load_mvtec_only_good_test_dataset(
        dataset_root_dir: str,
        category: str,
        image_size: int,
        batch_size: int,
        num_workers: int = 2,
        shuffle: bool = False,
        pin_memory: bool = True,
        drop_last: bool = False
):
    test_dataset = MvTecDataset(
        dataset_root_dir=dataset_root_dir,
        category=category,
        image_size=image_size,
        is_train=False,
        is_mask=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return test_loader



""" Early stopping strategy """
class LossEarlyStopping:
    def __init__(self, patience: int, min_delta: float, smoothing_window: int, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.verbose = verbose
        self.best_loss = None
        self.early_stop = False
        self.loss_history = []
        self.wait = 0
        self.min_delta = min_delta

    def get_smoothed_loss(self, losses):
        if len(losses) < self.smoothing_window:
            return np.mean(losses)
        else:
            return np.mean(losses[-self.smoothing_window:])

    def __call__(self, current_loss: float):
        self.loss_history.append(current_loss)
        smoothed_loss = self.get_smoothed_loss(self.loss_history)

        if self.best_loss is None:
            self.best_loss = smoothed_loss
            if self.verbose:
                print(f"Early stopping baseline set: {smoothed_loss:.4f}")
        elif smoothed_loss > self.best_loss - self.min_delta:
            self.wait += 1
            if self.verbose:
                print(f"No improvement: {smoothed_loss:.4f} vs {self.best_loss:.4f}, patience: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best loss: {self.best_loss:.4f}")
        else:
            # Loss improvement detect
            self.best_loss = smoothed_loss
            self.wait = 0
            if self.verbose:
                print(f"Loss improved to {smoothed_loss:.4f}")
                print(100*"=")

        return self.early_stop


""" Optimizers select """
def get_optimizer(optimizer_name: str, params, **kwargs) -> torch.optim.Optimizer:
    if optimizer_name == "Adam":
        return torch.optim.Adam(params, **kwargs)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(params, **kwargs)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(params, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

