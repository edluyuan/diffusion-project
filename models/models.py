import jax
import jaxlib
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn

from nn import MLP
from modules import *


_inf_network = InfNetwork(latent_dim=2, hidden_dim=64, num_layers=5)
_gen_network = GenNetwork(output_dim=2, hidden_dim=64, num_layers=5)





