import jax
import jaxlib
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn

from nn import MLP
from embeddings import SinusoidalEmbedding

class InfNetwork(nn.Module):
    latent_dim: int
    hidden_dim: int = 64
    num_layers: int = 5

    @nn.compact
    def __call__(self, x, **kwargs):
        batch_size = x.shape[0]
        h = x.reshape((batch_size, -1))

        z_0 = MLP(
            output_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )(h)

        return z_0


class GenNetwork(nn.Module):
    output_dim: int
    hidden_dim: int = 64
    num_layers: int = 5

    @nn.compact
    def __call__(self, z_0, **kwargs):
        x_hat = MLP(
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

    @nn.compact
    def __call__(self, z_t, t, **kwargs):

        batch_size = z_t.shape[0]
        h_t = z_t.reshape((batch_size, -1))

        t_emb = SinusoidalEmbedding(dim=self.time_dim)(t)
        t_emb = MLP(
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=3
        )(t_emb)

        h_t = MLP(
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=3
        )(z_t)

        h_t = h_t + t_emb

        z_t_minus_1 = MLP(
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )(h_t)

        return z_t_minus_1



