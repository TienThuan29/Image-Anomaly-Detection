import torch
from torch import nn, Tensor
import torch.nn.functional as F

"""
    VAE UNet
"""
class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_p=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Calculate number of groups for GroupNorm (must divide channel count)
        def get_groups(channels):
            for groups in [8, 4, 2, 1]:
                if channels % groups == 0:
                    return groups
            return 1
        
        groups1 = get_groups(mid_channels)
        groups2 = get_groups(out_channels)
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(num_groups=groups1, num_channels=mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(num_groups=groups2, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_p=dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_p=0.1):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After concatenation with skip connection, we'll have in_channels + skip_channels
            # For UNet, skip_channels = in_channels//2
            self.conv = DoubleConv(in_channels + in_channels//2, out_channels, dropout_p=dropout_p)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, dropout_p=0.1):
        super(UNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # UNet encoder layers
        self.inc = DoubleConv(in_channels, 64, dropout_p=dropout_p)
        self.down1 = Down(64, 128, dropout_p=dropout_p)
        self.down2 = Down(128, 256, dropout_p=dropout_p)
        self.down3 = Down(256, 512, dropout_p=dropout_p)
        self.down4 = Down(512, 1024, dropout_p=dropout_p)
        
        # Latent space projection
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(1024 * 16 * 16, latent_dim)
        self.fc_var = nn.Linear(1024 * 16 * 16, latent_dim)
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Store intermediate features for skip connections
        x1 = self.inc(x)      # 256x256 -> 256x256
        x2 = self.down1(x1)   # 256x256 -> 128x128
        x3 = self.down2(x2)   # 128x128 -> 64x64
        x4 = self.down3(x3)   # 64x64 -> 32x32
        x5 = self.down4(x4)   # 32x32 -> 16x16
        
        # Flatten and project to latent space
        x_flat = self.flatten(x5)  # [B, 1024*16*16]
        mu = self.fc_mu(x_flat)    # [B, latent_dim]
        logvar = self.fc_var(x_flat)  # [B, latent_dim]
        
        return mu, logvar, (x1, x2, x3, x4, x5)

class UNetDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, dropout_p=0.1):
        super(UNetDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Project latent back to feature space
        self.fc = nn.Linear(latent_dim, 1024 * 16 * 16)
        
        # UNet decoder layers - need to account for skip connection channels
        # up1: 1024 + 512 (skip) = 1536 -> 512
        # up2: 512 + 256 (skip) = 768 -> 256  
        # up3: 256 + 128 (skip) = 384 -> 128
        # up4: 128 + 64 (skip) = 192 -> 64
        self.up1 = Up(1024, 512, bilinear=True, dropout_p=dropout_p)
        self.up2 = Up(512, 256, bilinear=True, dropout_p=dropout_p)
        self.up3 = Up(256, 128, bilinear=True, dropout_p=dropout_p)
        self.up4 = Up(128, 64, bilinear=True, dropout_p=dropout_p)
        
        # Final output layer
        self.outc = OutConv(64, out_channels)
        self.sigmoid = nn.Sigmoid()
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z, skip_features):
        x1, x2, x3, x4, x5 = skip_features
        
        # Project latent back to feature space
        x = self.fc(z)  # [B, 1024*16*16]
        x = x.view(-1, 1024, 16, 16)  # [B, 1024, 16, 16]
        
        # Decode with skip connections
        x = self.up1(x, x4)   # 16x16 -> 32x32
        x = self.up2(x, x3)   # 32x32 -> 64x64
        x = self.up3(x, x2)   # 64x64 -> 128x128
        x = self.up4(x, x1)   # 128x128 -> 256x256
        
        # Final output
        x = self.outc(x)      # [B, out_channels, 256, 256]
        x = self.sigmoid(x)   # Ensure output in [0, 1]
        
        return x

class VAEUNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            latent_dim: int,
            out_channels: int,
            dropout_p: float = 0.1
    ):
        super(VAEUNet, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        self.encoder = UNetEncoder(
            in_channels=in_channels, 
            latent_dim=latent_dim, 
            dropout_p=dropout_p
        )
        self.decoder = UNetDecoder(
            latent_dim=latent_dim, 
            out_channels=out_channels, 
            dropout_p=dropout_p
        )

    def encode(self, x_input: Tensor) -> tuple[Tensor, Tensor, tuple]:
        """Encode x_input to latent z and return skip features"""
        return self.encoder(x_input)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor, skip_features: tuple) -> Tensor:
        """Decode latent z -> reconstructed image using skip features"""
        return self.decoder(z, skip_features)

    def forward(self, x_input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through VAE"""
        mu, logvar, skip_features = self.encode(x_input)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, skip_features)
        return reconstructed, mu, logvar

    def reconstruct(self, x_input: Tensor) -> Tensor:
        """Reconstruct from input images (B, C, 256, 256)"""
        return self.forward(x_input)[0]

    def get_z(self, x_input: Tensor) -> Tensor:
        """Get latent embedding of input images"""
        mu, logvar, _ = self.encode(x_input)
        z = self.reparameterize(mu, logvar)
        return z

    def generate_from_z(self, z: Tensor, skip_features: tuple) -> Tensor:
        """Generate images from latent embedding z using skip features"""
        return self.decode(z, skip_features)

    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        """Generate samples by sampling from latent space"""
        # For sampling, we need to create dummy skip features
        # This is a limitation of the UNet architecture for generation
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Create dummy skip features with appropriate shapes
        dummy_x1 = torch.zeros(num_samples, 64, 256, 256, device=device)
        dummy_x2 = torch.zeros(num_samples, 128, 128, 128, device=device)
        dummy_x3 = torch.zeros(num_samples, 256, 64, 64, device=device)
        dummy_x4 = torch.zeros(num_samples, 512, 32, 32, device=device)
        dummy_x5 = torch.zeros(num_samples, 1024, 16, 16, device=device)
        
        skip_features = (dummy_x1, dummy_x2, dummy_x3, dummy_x4, dummy_x5)
        return self.decode(z, skip_features)

    def get_skip_features(self, x_input: Tensor) -> tuple:
        """Get skip features for a given input (useful for generation)"""
        _, _, skip_features = self.encode(x_input)
        return skip_features
