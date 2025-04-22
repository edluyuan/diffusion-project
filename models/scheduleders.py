import jax.numpy as jnp


def vp_schedule(T, beta_min=1e-4, beta_max=0.02):
    """
    Variance Preserving (VP) diffusion schedule.
    """
    # Linear variance schedule
    betas = jnp.linspace(beta_min, beta_max, T)

    # Calculate alphas and cumulative products
    alphas = 1.0 - betas
    alpha_cumprod = jnp.cumprod(alphas)

    # Square roots for parameterization
    sqrt_alpha_cumprod = jnp.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = jnp.sqrt(1.0 - alpha_cumprod)

    # Diagonal case for gamma_t
    gamma_t = sqrt_one_minus_alpha_cumprod ** 2 / jnp.clip(1 - alpha_cumprod, 1e-5)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_cumprod": alpha_cumprod,
        "sqrt_alpha_cumprod": sqrt_alpha_cumprod,
        "sqrt_one_minus_alpha_cumprod": sqrt_one_minus_alpha_cumprod,
        "gamma_t": gamma_t
    }

