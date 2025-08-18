import torch
from torch import nn, Tensor
from torchvision.models import resnet18

"""
    VAE Resnet
"""
class ResNetEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            latent_dim: int
    ):
        super(ResNetEncoder, self).__init__()
        self.resnet = resnet18(weights=None)

        # Remove the final fully connected layer and avgpool
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_var = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x_input: Tensor) -> tuple[Tensor, Tensor]:
        x = self.features(x_input) # [ B, 512, 8, 8 ]
        x = self.flatten(x) # [ B, 512 * 8 * 8 ]
        mu = self.fc_mu(x)      # [ B, latent_dim ]
        logvar = self.fc_var(x) # [ B, latent_dim ]
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

class ResNetDecoder(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            out_channels: int = 3,
            dropout_p: float = 0.1,
    ):
        super(ResNetDecoder, self).__init__()

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
            in_channels: int,
            latent_dim: int,
            out_channels: int = 3
    ):
        super(VAEResNet, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.encoder = ResNetEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = ResNetDecoder(latent_dim=latent_dim)

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



# class ResNetDecoder(nn.Module):
#     def __init__(
#             self,
#             latent_dim: int,
#             out_channels: int = 3,
#             kernel_size: int = 4,
#             stride: int = 2,
#             padding: int = 1,
#             dropout_p: float = 0.1,
#     ):
#         super(ResNetDecoder, self).__init__()
#
#         self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
#         self.dropout_p = dropout_p
#
#         self.decoder = nn.Sequential(
#             # 8x8 -> 16x16
#             nn.ConvTranspose2d(512, 256, kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(self.dropout_p),
#
#             # 16x16 -> 32x32
#             nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(self.dropout_p),
#
#             # 32x32 -> 64x64
#             nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Dropout(self.dropout_p),
#
#             # 64x64 -> 128x128
#             nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Dropout(self.dropout_p),
#
#             # 128x128 -> 256x256
#             nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Dropout(self.dropout_p)
#         )
#
#         # Final layer, get RGB output
#         self.final_layer = nn.Sequential(
#             nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, z: Tensor) -> Tensor:
#         x =self.fc(z)
#         x = x.view(-1, 512, 8, 8)
#         x = self.decoder(x)
#         x_recon = self.final_layer(x)
#         return x_recon

"""
    VAE CNN
"""
# class VAE(nn.Module):
#
#     def __init__(
#             self,
#             in_channels: int,
#             latent_dim: int,
#             dropout: float,
#     ):
#         super(VAE, self).__init__()
#         self.in_channels = in_channels
#         self.latent_dim = latent_dim
#         self.dropout_p = dropout
#
#         modules = []
#         self.hidden_dims = [32, 64, 128, 256, 512]
#
#         # Encoder
#         for h_dim in self.hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
#                     nn.BatchNorm2d(h_dim),
#                     nn.LeakyReLU(),
#                     nn.Dropout(p=self.dropout_p),
#                 )
#             )
#             in_channels = h_dim
#         self.encoder = nn.Sequential(*modules)
#         self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
#         self.fc_var = nn.Linear(512 * 8 * 8, latent_dim)
#
#         # Decoder
#         modules = []
#         self.decoder_input = nn.Sequential(
#             nn.Linear(latent_dim, 512 * 8 * 8),
#             nn.LeakyReLU()
#         )
#         self.hidden_dims.reverse()
#         for i in range(len(self.hidden_dims) - 1):
#             modules.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(
#                         self.hidden_dims[i],
#                         self.hidden_dims[i+1],
#                         kernel_size=3,
#                         stride=2,
#                         padding=1,
#                         output_padding=1),
#                     nn.BatchNorm2d(self.hidden_dims[i+1]),
#                     nn.LeakyReLU(),
#                     nn.Dropout(p=self.dropout_p)
#                 )
#             )
#         self.decoder = nn.Sequential(*modules)
#
#         # Final layer to reconstruct origin image
#         self.final_layer = nn.Sequential(
#             nn.ConvTranspose2d(self.hidden_dims[-1],
#                                self.hidden_dims[-1],
#                                kernel_size=3,
#                                stride=2,
#                                padding=1,
#                                output_padding=1),
#             nn.BatchNorm2d(self.hidden_dims[-1]),
#             nn.LeakyReLU(),
#             nn.Conv2d(self.hidden_dims[-1], out_channels=3,
#                       kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#
#
#     def encode(self, x_input: Tensor) -> tuple[Tensor, Tensor]:
#         result = self.encoder(x_input)
#         result = torch.flatten(result, start_dim=1)
#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)
#         return mu, log_var
#
#
#     # Reparameterization trick to sample from N(mu, var) from N(0,1)
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu
#
#
#     def decode(self, z: Tensor) -> Tensor:
#         result = self.decoder_input(z)
#         result = result.view(-1, 512, 8, 8)
#         result = self.decoder(result)
#         result = self.final_layer(result)
#         return result
#
#
#     def forward(self, x_input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
#         mu, logvar = self.encode(x_input)
#         z = self.reparameterize(mu,logvar)
#         return self.decode(z), mu, logvar
#
#
#     # Reconstruct from input images (B, 3, 224, 224)
#     def reconstruct(self, x_input: Tensor) -> Tensor:
#         return self.forward(x_input)[0] # Get value from self.decode(z), ignore mu,logvar
#
#
#     # Get latent embedding of input images
#     def get_z(self, x_input: Tensor) -> Tensor:
#         mu, logvar = self.encode(x_input)
#         z = self.reparameterize(mu, logvar)
#         return z
#
#
#     # Generate images from latent embedding z
#     def generate_from_z(self, z: Tensor) -> Tensor:
#         return self.decode(z)