import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class VAEUnet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            latent_dim: int,
            out_channels: int,
            dropout_p: float = 0.1
    ):
        super(VAEUnet, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.encoder = UnetEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            dropout_p=dropout_p
        )
        self.decoder = UnetDecoder(
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
        z = torch.randn(num_samples, self.latent_dim, device=device)

        # Create dummy skip features with appropriate shapes (ultra small)
        dummy_x1 = torch.zeros(num_samples, 16, 256, 256, device=device)  # 32 -> 16
        dummy_x2 = torch.zeros(num_samples, 32, 128, 128, device=device)  # 64 -> 32
        dummy_x3 = torch.zeros(num_samples, 64, 64, 64, device=device)  # 128 -> 64
        dummy_x4 = torch.zeros(num_samples, 128, 32, 32, device=device)  # 256 -> 128

        skip_features = (dummy_x1, dummy_x2, dummy_x3, dummy_x4)
        return self.decode(z, skip_features)

    def get_skip_features(self, x_input: Tensor) -> tuple:
        """Get skip features for a given input (useful for generation)"""
        _, _, skip_features = self.encode(x_input)
        return skip_features


""" Unet Decoder """
class UnetDecoder(nn.Module):
      def __init__(
            self,
            latent_dim: int,
            out_channels: int,
            dropout_p: float,
      ):
            super(UnetDecoder, self).__init__()
            self.latent_dim = latent_dim
            self.out_channels = out_channels
            self.dropout_p = dropout_p
            
            self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
            # Upsample from 4x4 to 32x32
            self.upsample_to_32 = nn.Sequential(
                  nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=8, stride=8),
                  nn.GroupNorm(8, 128),
                  nn.LeakyReLU(inplace=True)
            ) 
            
            self.up1 = Up(128, 64, bilinear=True, dropout_p=dropout_p)  # 128 -> 64
            self.up2 = Up(64, 32, bilinear=True, dropout_p=dropout_p)   # 64 -> 32
            self.up3 = Up(32, 16, bilinear=True, dropout_p=dropout_p)   # 32 -> 16
            
            # Final output layer
            self.outc = OutConv(in_channels=16, out_channels=out_channels)  # 16 -> out_channels
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
      
      
      def forward(self, z: torch.Tensor, skip_features: tuple):
            x1, x2, x3 = skip_features
            x = self.fc(z)  # [B, 128*4*4]
            x = x.view(-1, 128, 4, 4)     # [B, 128, 4, 4]
            x = self.upsample_to_32(x)  # [B, 128, 32, 32]
            
            # Decode with skip connections
            x = self.up1(x, x3)  # 32x32 -> 64x64
            x = self.up2(x, x2)  # 64x64 -> 128x128
            x = self.up3(x, x1)  # 128x128 -> 256x256
            
            # Final output
            x = self.outc(x)  # [B, out_channels, 256, 256]
            x = self.sigmoid(x)  # Ensure output in [0, 1]

            return x
            
            

class Up(nn.Module):
      def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout_p: float,
            bilinear: bool = True,
      ):
            super().__init__()
            if bilinear:
                  self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                  self.conv = DoubleConv(
                        in_channels=(in_channels + in_channels // 2), 
                        out_channels=out_channels,
                        dropout_p=dropout_p
                  )
            else:
                  self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                  self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)
                  
            
      def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            x1 = self.up(x1)
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


""" Unet Encoder """
class UnetEncoder(nn.Module):
      def __init__(
            self,
            in_channels: int,
            latent_dim: int,
            dropout_p: float,
      ):
            super(UnetEncoder, self).__init__()
            self.in_channels = in_channels
            self.latent_dim = latent_dim
            self.dropout_p = dropout_p
            
            self.inc = DoubleConv(in_channels=self.in_channels, out_channels=16, dropout_p=self.dropout_p)
            self.down1 = Down(in_channels=16, out_channels=32, dropout_p=self.dropout_p)
            self.down2 = Down(in_channels=32, out_channels=64, dropout_p=self.dropout_p)
            self.down3 = Down(in_channels=64, out_channels=128, dropout_p=self.dropout_p)
            self.global_pool = nn.AdaptiveAvgPool2d((4, 4))  # Global average pooling  
            
            self.flatten = nn.Flatten()
            self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)  # Từ 256*32*32 -> 128*4*4
            self.fc_var = nn.Linear(128 * 4 * 4, latent_dim)          


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
                        

      def forward(self, x: torch.Tensor):
            # Store intermediate features for skip connections
            x1 = self.inc(x)        # 256x256 -> 256x256, channels: 16
            x2 = self.down1(x1)     # 256x256 -> 128x128, channels: 32
            x3 = self.down2(x2)     # 128x128 -> 64x64, channels: 64
            x4 = self.down3(x3)     # 64x64 -> 32x32, channels: 128
            x_pooled = self.global_pool(x4)  # 32x32 -> 4x4

            x_flat = self.flatten(x_pooled)     # [B, 128*4*4]
            mu = self.fc_mu(x_flat)             # [B, latent_dim]
            logvar = self.fc_var(x_flat)        # [B, latent_dim]

            return mu, logvar, (x1, x2, x3, x4)
      
      
      
""" Double Convolution """
class DoubleConv(nn.Module):
      
      def get_groups(self, channels: int) -> int:
            for groups in [8, 4, 2, 1]:
                if channels % groups == 0:
                    return groups
            return 1
      
      """(convolution => [BN] => ReLU) * 2"""
      def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: int = None,
            dropout_p: float = 0.1
      ):
            super().__init__()
            if not mid_channels:
                  mid_channels = out_channels

            groups1 = self.get_groups(mid_channels)
            groups2 = self.get_groups(out_channels)

            self.double_conv = nn.Sequential(
                  nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, 
                            kernel_size=3, padding=1, bias=False),
                  nn.GroupNorm(num_groups=groups1, num_channels=mid_channels),
                  nn.LeakyReLU(inplace=True),
                  nn.Dropout(p=dropout_p),
                  
                  nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, 
                            kernel_size=3, padding=1, bias=False),
                  nn.GroupNorm(num_groups=groups2, num_channels=out_channels),
                  nn.LeakyReLU(inplace=True),
                  nn.Dropout(p=dropout_p)
            )
      
      def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.double_conv(x)
      


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # kernel size = 2, stride = 2, reduce size half
            DoubleConv(in_channels, out_channels, dropout_p=dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
 

           
            
            
            
            
            