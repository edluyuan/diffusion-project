import jax
import jaxlib
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn

from .modules import *
from .scheduleders import *


class DiffusionVAE(nn.Module):
    latent_dim: int
    hidden_dim: int
    inf_layers: int
    gen_layers: int
    T: int

    def setup(self) -> None:
        # Networks for the hierarchical model
        self._inf_network = InfNetwork(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.inf_layers
        )

        self._gen_network = GenNetwork(
            output_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.gen_layers
        )

        self._rev_network = RevNetwork(
            output_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            T=self.T
        )

        # Initialize VP schedule parameters
        self._vp_params = vp_schedule(self.T)

    def _get_params(self, t):
        """
        Get diffusion parameters for timestep t.
        """
        # Extract parameters for the given timesteps
        alpha_t = jnp.take(self._vp_params["sqrt_alpha_cumprod"], t)
        sigma_t = jnp.take(self._vp_params["sqrt_one_minus_alpha_cumprod"], t)
        gamma_t = jnp.take(self._vp_params["gamma_t"], t)

        return alpha_t, sigma_t, gamma_t

    def __call__(self, x, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)

        batch_size = x.shape[0]

        # 1. Encode x to z_0 through inference network: q_φ(z_0|x)
        z_0 = self._inf_network(x)

        # 2. Sample a timestep t
        t_key, noise_key = jax.random.split(key)
        t = jax.random.randint(t_key, (batch_size,), 0, self.T)

        # 3. Forward diffusion: q_φ(z_t|z_0) using reparameterization
        alpha_t, sigma_t, gamma_t = self._get_params(t)
        epsilon = jax.random.normal(noise_key, z_0.shape)
        z_t = alpha_t[:, None] * z_0 + sigma_t[:, None] * epsilon  # z_t = α_t·z_0 + σ_t·ε

        # 4. Reverse process: predicting z_{t-1} or z_0 with the reverse network
        z_tm1 = self._rev_network(z_t, t)

        # 5. Compute losses
        # Reconstruction loss
        x_hat = self._gen_network(z_0)
        recon_loss = ((x_hat - x) ** 2).mean()

        # Diffusion loss - using MSE between predicted z_{t-1} and ground truth z_0
        diffusion_loss = ((z_tm1 - z_0) ** 2).mean()
        # Diffusion loss weighted by gamma_t (diagonal case)
        #diffusion_loss = (gamma_t[:, None] * (z_tm1 - z_0) ** 2).mean()

        # Total loss (variational free energy)
        loss = recon_loss + diffusion_loss

        return loss, (recon_loss, diffusion_loss, z_0, z_t)

    def sample(self, batch_size, key=None):
        """
        Sample new data by running the reverse diffusion process.
        Start from z_T ~ N(0, I) and iteratively predict z_{t-1} until z_0.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Sample z_T from standard Gaussian
        z_t = jax.random.normal(key, (batch_size, self.latent_dim))

        # Iteratively sample from p_θ(z_t|z_{t+1}) for t=T-1,...,0
        for t in range(self.T, 0, -1):
            t_batch = jnp.full((batch_size,), t)

            # Predict z_{t-1} using reverse network
            z_t = self._rev_network(z_t, t_batch)

        # Generate x_hat from final z_0
        x_hat = self._gen_network(z_t)

        return x_hat





















