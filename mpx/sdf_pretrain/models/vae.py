import jax
import jax.numpy as jnp
import flax.linen as nn

class VAEEncoder(nn.Module):
    latent_dim: int = 90

    @nn.compact
    def __call__(self, x):
        # x shape: (Batch, 40, 40, 1)
        
        # 40x40 -> 20x20
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        # 20x20 -> 10x10
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        # 展平: 10 * 10 * 64 = 6400
        x = x.reshape((x.shape[0], -1)) 
        
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        
        mean_x = nn.Dense(features=self.latent_dim)(x)
        logvar_x = nn.Dense(features=self.latent_dim)(x)
        return mean_x, logvar_x

class HeightmapDecoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(features=512)(z)
        x = nn.relu(x)
        
        # 还原回 10 * 10 * 64 = 6400
        x = nn.Dense(features=6400)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], 10, 10, 64))
        
        # 10x10 -> 20x20
        x = jax.image.resize(x, shape=(x.shape[0], 20, 20, 64), method='nearest')
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        
        # 20x20 -> 40x40
        x = jax.image.resize(x, shape=(x.shape[0], 40, 40, 32), method='nearest')
        x = nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        
        recon_x = nn.sigmoid(x) 
        return recon_x

class HeightmapVAE(nn.Module):
    latent_dim: int = 90
    def setup(self):
        self.encoder = VAEEncoder(latent_dim=self.latent_dim)
        self.decoder = HeightmapDecoder()
    def __call__(self, x, rng):
        mean, logvar = self.encoder(x)
        std = jnp.exp(0.5 * logvar)
        z = mean + jax.random.normal(rng, logvar.shape) * std
        return self.decoder(z), mean, logvar