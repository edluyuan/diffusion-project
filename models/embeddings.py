import jax.numpy as jnp
import flax.linen as nn

class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal time embedding for reverse models.
    """
    dim: int
    max_period: int = 10000

    @nn.compact
    def __call__(self, t):
        # Create sinusoidal position embeddings
        half_dim = self.dim // 2
        freqs = jnp.exp(-jnp.log(self.max_period) * jnp.arange(half_dim) / half_dim)
        args = t[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)

        return embedding