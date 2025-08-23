import torch
from torch import nn, Tensor
from torchvision.models import resnet18, resnet34, resnet50

"""
    VAE Resnet
"""
class ResNetEncoder(nn.Module):
    def __init__(
            self,
            image_size: int,
            in_channels: int,
            latent_dim: int,
            resnet_name: str,
            pool_size: int = 4
    ):
        super(ResNetEncoder, self).__init__()
        self.resnet_name = resnet_name
        if resnet_name == 'resnet18':
            self.resnet = resnet18(weights=None)
            out_channels = 512
        elif resnet_name == 'resnet34':
            self.resnet = resnet34(weights=None)
            out_channels = 512
        elif resnet_name == 'resnet50':
            self.resnet = resnet50(weights=None)
            out_channels = 2048
        else:
            raise ValueError(f"Unsupported ResNet architecture: {resnet_name}")

        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        # if resnet_name == 'resnet18':
        #     self.flatten = nn.Flatten()
        #     self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        #     self.fc_var = nn.Linear(512 * 8 * 8, latent_dim)
        # else:
        #     self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        #     in_feat = out_channels * pool_size * pool_size
        #     self.flatten = nn.Flatten()
        #     self.fc_mu = nn.Linear(in_feat, latent_dim)
        #     self.fc_var = nn.Linear(in_feat, latent_dim)

        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        in_feat = out_channels * pool_size * pool_size
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(in_feat, latent_dim)
        self.fc_var = nn.Linear(in_feat, latent_dim)


    def forward(self, x_input: Tensor) -> tuple[Tensor, Tensor]:
        x = self.features(x_input)  # [ B, 512, 8, 8 ]
        # if self.resnet_name != 'resnet18':
        #     x = self.pool(x)
        x = self.pool(x)
        x = self.flatten(x)         # [ B, 512 * 8 * 8 ]
        mu = self.fc_mu(x)          # [ B, latent_dim ]
        logvar = self.fc_var(x)     # [ B, latent_dim ]
        return mu, logvar


class UpBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, groups: int = 8, dropout_p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(num_groups=groups, num_channels=out_c),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            out_channels: int,
            dropout_p: float,
    ):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        # upsampling 8→16→32→64→128→256
        self.up1 = UpBlock(512, 256, groups=8, dropout_p=dropout_p)  # 8x8 -> 16x16
        self.up2 = UpBlock(256, 128, groups=8, dropout_p=dropout_p)  # 16x16 -> 32x32
        self.up3 = UpBlock(128,  64, groups=8, dropout_p=dropout_p)  # 32x32 -> 64x64
        self.up4 = UpBlock( 64,  32, groups=8, dropout_p=dropout_p)  # 64x64 -> 128x128
        self.up5 = UpBlock( 32,  16, groups=8, dropout_p=dropout_p)  # 128x128 -> 256x256

        # Final layer, get RGB output in [0,1]
        self.final_layer = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)


    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, z: Tensor) -> Tensor:
        x = self.fc(z)                  # [B, 512*8*8]
        x = x.view(-1, 512, 8, 8)      # [B, 512, 8, 8]
        x = self.up1(x)                # [B, 256, 16, 16]
        x = self.up2(x)                # [B, 128, 32, 32]
        x = self.up3(x)                # [B, 64,  64, 64]
        x = self.up4(x)                # [B, 32, 128,128]
        x = self.up5(x)                # [B, 16, 256,256]
        x_recon = self.final_layer(x)  # [B, 3,  256,256]
        return x_recon


class VAEResNet(nn.Module):
    def __init__(
            self,
            image_size: int,
            in_channels: int,
            latent_dim: int,
            out_channels: int,
            resnet_name: str,
            dropout_p: float,
    ):
        super(VAEResNet, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.encoder = ResNetEncoder(in_channels=in_channels, latent_dim=latent_dim, resnet_name=resnet_name, image_size=image_size)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=out_channels, dropout_p=dropout_p)

    def encode(self, x_input: Tensor) -> tuple[Tensor, Tensor]:
        """Encode x_input to latent z"""
        return self.encoder(x_input)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent z -> reconstructed image"""
        return self.decoder(z)

    def forward(self, x_input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through VAE"""
        mu, logvar = self.encode(x_input)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def reconstruct(self, x_input: Tensor) -> Tensor:
        """Reconstruct from input images (B, C, 256, 256)"""
        return self.forward(x_input)[0]

    def get_z(self, x_input: Tensor) -> Tensor:
        """Get latent embedding of input images"""
        mu, logvar = self.encode(x_input)
        z = self.reparameterize(mu, logvar)
        return z

    def generate_from_z(self, z: Tensor) -> Tensor:
        """Generate images from latent embedding z"""
        return self.decode(z)

    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        """Generate samples by sampling from latent space"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)