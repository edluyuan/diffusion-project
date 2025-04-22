import jax
import jaxlib
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn

from nn import MlpBlock
from embeddings import TimestepEmbedder

class InfNetwork(nn.Module):
    latent_dim: int
    hidden_dim: int = 64
    num_layers: int = 5

    @nn.compact
    def __call__(self, x, **kwargs):
        batch_size = x.shape[0]
        h_0 = x.reshape((batch_size, -1))

        z_0 = MlpBlock(
            output_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )(h_0)

        return z_0


class GenNetwork(nn.Module):
    output_dim: int
    hidden_dim: int = 64
    num_layers: int = 5

    @nn.compact
    def __call__(self, z_0, **kwargs):
        x_hat = MlpBlock(
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )(z_0)

        return x_hat


class RevNetwork(nn.Module):
    output_dim: int
    hidden_dim: int = 64
    time_dim: int = hidden_dim // 2
    num_layers: int = 3
    max_timesteps: int = 1000

    @nn.compact
    def __call__(self, z_t, t, **kwargs):

        batch_size = z_t.shape[0]
        h_t = z_t.reshape((batch_size, -1))

        t_emb = TimestepEmbedder(
            dim=self.time_dim,
            max_period=self.max_timesteps
        )(t)

        t_emb = MlpBlock(
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=3
        )(t_emb)

        h_t = MlpBlock(
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=3
        )(h_t)

        h_t = h_t + t_emb

        z_tm1 = MlpBlock(
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )(h_t)

        # Predict noise instead of z_{t-1}
        # epsilon = MlpBlock(
        #    output_dim=self.output_dim,
        #    hidden_dim=self.hidden_dim,
        #    num_layers=self.num_layers
        # )(h_t)
        # return epsilon

        return z_tm1







