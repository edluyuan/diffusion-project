#!/usr/bin/env python3
"""
Training script for DiffusionVAE on MOG-40 dataset using JAX/Flax.
"""

import os
import argparse
import logging
import numpy as np
import torch

import jax
import jax.numpy as jnp
from jax import random
import flax
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the model and dataset components
from models import DiffusionVAE
from data.dataloader.datasets import GMM, plot_contours


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiffusionVAE on MOG-40")

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--T", type=int, default=1000)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)

    # Dataset parameters
    parser.add_argument("--loc_scaling", type=float, default=10.0)
    parser.add_argument("--n_mixes", type=int, default=40)

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--eval_every", type=int, default=1000)

    return parser.parse_args()


def create_train_state(rng, model, learning_rate):
    """Create initial training state with initialized model and optimizer"""
    # Initialize model parameters with dummy input
    dummy_input = jnp.ones((1, 2))
    params = model.init(rng, dummy_input)

    # Create optimizer
    tx = optax.adam(learning_rate=learning_rate)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def train_step(state, batch, rng):
    """Execute one training step"""
    rng, step_rng = random.split(rng)

    def loss_fn(params):
        loss, aux = state.apply_fn(params, batch, key=step_rng)
        return loss, aux

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)

    # Update model parameters
    state = state.apply_gradients(grads=grads)

    metrics = {
        "loss": loss,
        "recon_loss": aux[0],
        "diffusion_loss": aux[1],
    }

    return state, metrics, rng


def plot_samples(gmm, samples, file_name=None, title=None):
    """Plot GMM density contours with JAX model samples"""
    # Convert JAX array to PyTorch tensor for visualization
    samples_torch = torch.tensor(np.array(samples))

    if file_name is None:
        plot_contours(gmm.log_prob, samples=samples_torch,
                      bounds=(-20, 20), n_contour_levels=30,
                      grid_width_n_points=200, device="cpu", title=title, plt_show=True)
    else:
        plot_contours(gmm.log_prob, samples=samples_torch,
                      bounds=(-20, 20), n_contour_levels=30,
                      grid_width_n_points=200, device="cpu", title=title, plt_show=False)
        plt.savefig(file_name)
        plt.close()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # Initialize random key
    key = random.PRNGKey(args.seed)
    key, model_key = random.split(key)

    # Create dataset with fixed seed
    gmm = GMM(
        dim=2,
        n_mixes=args.n_mixes,
        loc_scaling=args.loc_scaling,
        log_var_scaling=0.1,
        seed=args.seed,
        n_test_set_samples=10000,
        device="cpu"
    )

    # Create model with specified parameters
    model = DiffusionVAE(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        inf_layers=5,
        gen_layers=5,
        T=args.T
    )

    # Initialize training state
    state = create_train_state(model_key, model, args.learning_rate)

    # Training loop
    logging.info(f"Starting training with {args.steps} steps")
    metrics_history = []

    for step in tqdm(range(args.steps)):
        key, step_key = random.split(key)

        # Sample a batch of data with fixed key=0
        batch_torch = gmm.sample((args.batch_size,))
        # Convert to JAX array
        batch = jnp.array(batch_torch.numpy())

        # Training step
        state, metrics, key = train_step(state, batch, step_key)
        metrics_history.append(metrics)

        # Log metrics periodically
        if (step + 1) % 100 == 0:
            avg_loss = jnp.mean(jnp.array([m["loss"] for m in metrics_history[-100:]]))
            avg_recon = jnp.mean(jnp.array([m["recon_loss"] for m in metrics_history[-100:]]))
            avg_diff = jnp.mean(jnp.array([m["diffusion_loss"] for m in metrics_history[-100:]]))
            logging.info(f"Step {step + 1}: loss={avg_loss:.4f}, recon_loss={avg_recon:.4f}, diff_loss={avg_diff:.4f}")

        # Evaluate and save samples periodically
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            # Generate samples
            key, sample_key = random.split(key)
            samples = model.apply(state.params, args.batch_size, key=sample_key, method=model.sample)

            # Plot samples against ground truth distribution
            plot_path = os.path.join(args.output_dir, f"samples_step_{step + 1}.png")
            plot_samples(gmm, samples, file_name=plot_path, title=f"Step {step + 1}")
            logging.info(f"Saved samples plot to {plot_path}")

            # Save model checkpoint
            checkpoint = {"params": state.params}
            with open(os.path.join(args.output_dir, f"checkpoint_{step + 1}.flax"), "wb") as f:
                f.write(flax.serialization.to_bytes(checkpoint))

    logging.info("Training completed!")


if __name__ == "__main__":
    main()