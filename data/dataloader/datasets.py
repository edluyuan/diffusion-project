#!/usr/bin/env python3
r"""
Data Generation Module.

For now it provides a 2D Gaussian Mixture Model (GMM) for generating synthetic data and utilities
for plotting density contours. The GMM is used to model the empirical data distribution
\\(\hat{p}(x)\\).

The GMM is defined as:
    \\(p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x; \mu_k, \Sigma_k)\\)
where the mixture components are randomly initialized.
"""

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GMM(nn.Module):
    """
    2D Gaussian Mixture Model (GMM) for data generation.

    This model generates data from a mixture of Gaussians and can compute
    the log-probability of a sample.
    """

    def __init__(self, dim, n_mixes, loc_scaling, log_var_scaling=0.1,
                 seed=0, n_test_set_samples=1000, device="cpu"):
        """
        Initialize the GMM.

        Args:
            dim (int): Data dimensionality.
            n_mixes (int): Number of mixture components.
            loc_scaling (float): Scale factor for component means.
            log_var_scaling (float): Scaling factor for log variance.
            seed (int): Random seed for reproducibility.
            n_test_set_samples (int): Number of test samples.
            device (str or torch.device): Device for computations.
        """
        super(GMM, self).__init__()
        self.seed = seed
        torch.manual_seed(seed)
        self.n_mixes = n_mixes
        self.dim = dim
        self.n_test_set_samples = n_test_set_samples

        # Randomly initialize component means scaled by loc_scaling.
        mean = (torch.rand((n_mixes, dim)) - 0.5) * 2 * loc_scaling
        # Fixed log variance for each component.
        log_var = torch.ones((n_mixes, dim)) * log_var_scaling

        # Uniform mixture weights.
        self.register_buffer("cat_probs", torch.ones(n_mixes))
        self.register_buffer("locs", mean)
        # Use softplus to ensure positive scale parameters.
        self.register_buffer("scale_trils",
                             torch.diag_embed(F.softplus(log_var)))
        self.device = device
        # Move the model's buffers to the specified device.
        self.to(self.device)

    def to(self, device):
        """
        Override the to() method to move all buffers to the specified device.

        Args:
            device (torch.device or str): The device to move the model to.

        Returns:
            self: The model on the new device.
        """
        super().to(device)
        self.device = device
        return self

    @property
    def distribution(self):
        """
        Return the MixtureSameFamily distribution representing the GMM.

        All parameters are explicitly moved to the current device.
        """
        # Ensure device is a torch.device
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        cat_probs = self.cat_probs.to(device)
        locs = self.locs.to(device)
        scale_trils = self.scale_trils.to(device)
        mix = torch.distributions.Categorical(cat_probs)
        comp = torch.distributions.MultivariateNormal(
            locs, scale_tril=scale_trils, validate_args=False
        )
        return torch.distributions.MixtureSameFamily(
            mixture_distribution=mix,
            component_distribution=comp,
            validate_args=False
        )

    @property
    def test_set(self):
        """
        Generate a test set of samples from the GMM.
        """
        return self.sample((self.n_test_set_samples,))

    def log_prob(self, x: torch.Tensor, **kwargs):
        """
        Compute the log-probability of x under the GMM.

        Ensures that x is moved to the correct device before computation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Log-probabilities.
        """
        x = x.to(self.device)
        log_prob = self.distribution.log_prob(x)
        # Clip very low log-probabilities for numerical stability.
        mask = torch.zeros_like(log_prob)
        mask[log_prob < -1e4] = -float("inf")
        return log_prob + mask

    def sample(self, shape=(1,)):
        """
        Sample from the GMM.

        Args:
            shape (tuple): Desired shape for the samples.

        Returns:
            torch.Tensor: Generated samples.
        """
        return self.distribution.sample(shape)


def plot_contours(log_prob_func, samples=None, ax=None,
                  bounds=(-25.0, 25.0), grid_width_n_points=100,
                  n_contour_levels=None, log_prob_min=-1000.0,
                  device='cpu', plot_marginal_dims=[0, 1],
                  s=2, alpha=0.6, title=None, plt_show=True, xy_tick=True):
    r"""
    Plot contours of a log-probability function over a 2D grid.

    Useful for visualizing the true data density \(\hat{p}(x)\) alongside generated samples.

    Args:
        log_prob_func (callable): Function computing log probability.
        samples (torch.Tensor, optional): Samples to overlay.
        ax (matplotlib.axes.Axes, optional): Plot axes.
        bounds (tuple): (min, max) bounds for each axis.
        grid_width_n_points (int): Number of grid points per axis.
        n_contour_levels (int, optional): Number of contour levels.
        log_prob_min (float): Minimum log probability value.
        device (str): Device for computation.
        plot_marginal_dims (list): Dimensions to plot.
        s (int): Marker size.
        alpha (float): Marker transparency.
        title (str, optional): Plot title.
        plt_show (bool): Whether to display the plot.
        xy_tick (bool): Whether to set custom ticks.
    """
    if ax is None:
        fig, ax = plt.subplots(1)

    x_points = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    grid_points = torch.tensor(list(itertools.product(x_points, x_points)),
                               device=device)
    log_p_x = log_prob_func(grid_points).cpu().detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))

    x1 = grid_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).cpu().numpy()
    x2 = grid_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).cpu().numpy()

    if n_contour_levels:
        ax.contour(x1, x2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x1, x2, log_p_x)

    if samples is not None:
        samples = np.clip(samples.detach().cpu(), bounds[0], bounds[1])
        ax.scatter(samples[:, plot_marginal_dims[0]],
                   samples[:, plot_marginal_dims[1]],
                   s=s, alpha=alpha)
        if xy_tick:
            ax.set_xticks([bounds[0], 0, bounds[1]])
            ax.set_yticks([bounds[0], 0, bounds[1]])
        ax.tick_params(axis='both', which='major', labelsize=15)

    if title:
        ax.set_title(title, fontsize=15)
    if plt_show:
        plt.show()


def plot_MoG40(log_prob_function, samples, file_name=None, title=None):
    """
    Plot GMM density contours with overlaid generated samples.

    Args:
        log_prob_function (callable): Function computing log probability.
        samples (torch.Tensor): Samples from the model.
        file_name (str, optional): Path to save the plot.
        title (str, optional): Plot title.
    """
    if file_name is None:
        plot_contours(log_prob_function, samples=samples.detach().cpu(),
                      bounds=(-20, 20), n_contour_levels=30,
                      grid_width_n_points=200, device="cpu", title=title, plt_show=True)
    else:
        plot_contours(log_prob_function, samples=samples.detach().cpu(),
                      bounds=(-20, 20), n_contour_levels=30,
                      grid_width_n_points=200, device="cpu", title=title, plt_show=False)
        plt.savefig(file_name)
        plt.close()
