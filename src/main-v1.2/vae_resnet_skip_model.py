import torch
from torch import nn, Tensor
from torchvision.models import resnet18, resnet34, resnet50

"""
    VAE Resnet + Skip Connections
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

        # Tách các layers để có thể lấy intermediate features
        resnet_layers = list(self.resnet.children())
        self.conv1 = resnet_layers[0]  # 7x7 conv
        self.bn1 = resnet_layers[1]
        self.relu = resnet_layers[2]
        self.maxpool = resnet_layers[3]

        self.layer1 = resnet_layers[4]  # 64 channels
        self.layer2 = resnet_layers[5]  # 128 channels
        self.layer3 = resnet_layers[6]  # 256 channels
        self.layer4 = resnet_layers[7]  # 512 channels

        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        in_feat = out_channels * pool_size * pool_size
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(in_feat, latent_dim)
        self.fc_var = nn.Linear(in_feat, latent_dim)

    def forward(self, x_input: Tensor) -> tuple[Tensor, Tensor, list]:
        # Lưu intermediate features cho skip connections
        skip_features = []

        # Initial layers
        x = self.conv1(x_input)  # [B, 64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        skip_features.append(x)  # Skip 0: [B, 64, 128, 128]

        x = self.maxpool(x)  # [B, 64, 64, 64]

        # ResNet blocks
        x = self.layer1(x)  # [B, 64, 64, 64]
        skip_features.append(x)  # Skip 1: [B, 64, 64, 64]

        x = self.layer2(x)  # [B, 128, 32, 32]
        skip_features.append(x)  # Skip 2: [B, 128, 32, 32]

        x = self.layer3(x)  # [B, 256, 16, 16]
        skip_features.append(x)  # Skip 3: [B, 256, 16, 16]

        x = self.layer4(x)  # [B, 512, 8, 8]
        skip_features.append(x)  # Skip 4: [B, 512, 8, 8]

        # Bottleneck cho VAE
        x = self.pool(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar, skip_features


class UpBlock(nn.Module):
    """Standard upsampling block without skip connection"""

    def __init__(self, in_c: int, out_c: int, groups: int = 8, dropout_p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(num_groups=groups, num_channels=out_c),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UpBlockWithSkip(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int, groups: int = 8, dropout_p: float = 0.1):
        super().__init__()
        self.has_skip = skip_c > 0

        # Projection layer để match skip connection channels
        if self.has_skip:
            self.skip_proj = nn.Conv2d(skip_c, out_c, kernel_size=1) if skip_c != out_c else nn.Identity()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='reflect')
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=out_c)

        # Second conv - channels depend on whether we have skip connection
        conv2_in_channels = out_c * 2 if self.has_skip else out_c
        self.conv2 = nn.Conv2d(conv2_in_channels, out_c, kernel_size=3, padding=1, padding_mode='reflect')
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_c)

        self.activation = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor, skip: Tensor = None) -> Tensor:
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        if skip is not None and self.has_skip:
            skip = self.skip_proj(skip)
            # Đảm bảo spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class DecoderWithSkip(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            out_channels: int,
            dropout_p: float,
    ):
        super(DecoderWithSkip, self).__init__()

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)

        # Upsampling blocks với skip connections
        # 8→16→32→64→128→256
        self.up1 = UpBlockWithSkip(512, 256, 256, groups=8, dropout_p=dropout_p)  # 8x8->16x16, skip từ layer3
        self.up2 = UpBlock(256, 128, groups=8, dropout_p=dropout_p)  # 16x16->32x32, skip từ layer2
        # self.up2 = UpBlock(256, 128, groups=8, dropout_p=dropout_p)
        self.up3 = UpBlock(128, 64, groups=8, dropout_p=dropout_p)  # 32x32->64x64, skip từ layer1
        # self.up4 = UpBlockWithSkip(64, 64, 32, groups=8, dropout_p=dropout_p)  # 64x64->128x128
        self.up4 = UpBlock(64, 32, groups=8, dropout_p=dropout_p) # 64x64->128x128
        self.up5 = UpBlock(32, 16, groups=8, dropout_p=dropout_p)  # 128x128->256x256, no skip

        # Final layer
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

    def forward(self, z: Tensor, skip_features: list) -> Tensor:
        x = self.fc(z)  # [B, 512*8*8]
        x = x.view(-1, 512, 8, 8)  # [B, 512, 8, 8]

        # Upsampling với skip connections (reverse order)
        x = self.up1(x, skip_features[3])  # [B, 256, 16, 16] + skip từ layer3
        x = self.up2(x)  # [B, 128, 32, 32] + skip từ layer2
        x = self.up3(x)  # [B, 64, 64, 64] + skip từ layer1
        x = self.up4(x)  # [B, 32, 128, 128] + skip từ conv1
        x = self.up5(x)  # [B, 16, 256, 256] no skip

        x_recon = self.final_layer(x)  # [B, 3, 256, 256]
        return x_recon


class VAEResNetWithSkip(nn.Module):
    def __init__(
            self,
            image_size: int,
            in_channels: int,
            latent_dim: int,
            out_channels: int,
            resnet_name: str,
            dropout_p: float,
    ):
        super(VAEResNetWithSkip, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.encoder = ResNetEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            resnet_name=resnet_name,
            image_size=image_size
        )
        self.decoder = DecoderWithSkip(
            latent_dim=latent_dim,
            out_channels=out_channels,
            dropout_p=dropout_p
        )

    def encode(self, x_input: Tensor) -> tuple[Tensor, Tensor, list]:
        """Encode x_input to latent z và trả về skip features"""
        return self.encoder(x_input)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor, skip_features: list) -> Tensor:
        """Decode latent z -> reconstructed image với skip connections"""
        return self.decoder(z, skip_features)

    def forward(self, x_input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through VAE"""
        mu, logvar, skip_features = self.encode(x_input)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, skip_features)
        return reconstructed, mu, logvar

    def reconstruct(self, x_input: Tensor) -> Tensor:
        """Reconstruct from input images"""
        return self.forward(x_input)[0]

    def get_z(self, x_input: Tensor) -> Tensor:
        """Get latent embedding"""
        mu, logvar, _ = self.encode(x_input)
        z = self.reparameterize(mu, logvar)
        return z


