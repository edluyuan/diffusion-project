import flax.linen as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with SiLU activations
    """
    output_dim: int
    hidden_dim: int = 64
    num_layers: int = 2

    @nn.compact
    def __call__(self, x):
        # First layer: input to hidden
        h = nn.Dense(features=self.hidden_dim)(x)
        h = nn.silu(h)

        # Hidden layers
        for _ in range(self.num_layers - 1):
            h = nn.Dense(features=self.hidden_dim)(h)
            h = nn.silu(h)

        # Output layer
        return nn.Dense(features=self.output_dim)(h)