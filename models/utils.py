import jax
import jaxlib
import jax.numpy as jnp
import jax.scipy as jsp


def log_gaussian(x, mu, logvar):
    """
    Computes log probability of a Gaussian: log N(x; mu, exp(logvar))

    Args:
        x: JAX array - values to evaluate
        mu: JAX array - mean of the Gaussian
        logvar: JAX array or float - log variance of the Gaussian

    Returns:
        JAX array of log probabilities
    """
    # Convert scalar logvar to array if needed
    if not isinstance(logvar, jnp.ndarray):
        logvar = jnp.array(logvar, dtype=x.dtype)

    # Add dimensions to logvar for broadcasting if needed
    while logvar.ndim < x.ndim:
        logvar = jnp.expand_dims(logvar, axis=0)

    const = jnp.log(2 * jnp.pi)
    return -0.5 * (logvar + jnp.square(x - mu) / jnp.exp(logvar) + const)