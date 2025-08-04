from torch import nn, Tensor
import torch

class VAE(nn.Module):

    def __init__(
            self,
            in_channels: int,
            latent_dim: int,
    ):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        modules = []
        self.hidden_dims = [32, 64, 128, 256, 512]

        # Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_var = nn.Linear(512 * 8 * 8, latent_dim)

        # Decoder
        modules = []
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.LeakyReLU()
        )
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i+1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)

        # Final layer to reconstruct origin image
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1],
                               self.hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid()
        )


    def encode(self, x_input: Tensor) -> tuple[Tensor, Tensor]:
        result = self.encoder(x_input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var


    # Reparameterization trick to sample from N(mu, var) from N(0,1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 8, 8)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


    def forward(self, x_input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x_input)
        z = self.reparameterize(mu,logvar)
        return self.decode(z), mu, logvar


    # Reconstruct from input images (B, 3, 224, 224)
    def reconstruct(self, x_input: Tensor) -> Tensor:
        return self.forward(x_input)[0] # Get value from self.decode(z), ignore mu,logvar


    # Get latent embedding of input images
    def get_z(self, x_input: Tensor) -> Tensor:
        mu, logvar = self.encode(x_input)
        z = self.reparameterize(mu, logvar)
        return z


    # Generate images from latent embedding z
    def generate_from_z(self, z: Tensor) -> Tensor:
        return self.decode(z)