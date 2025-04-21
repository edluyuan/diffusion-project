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
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.silu(x)

        # Hidden layers
        for _ in range(self.num_layers - 1):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.silu(x)

        # Output layer
        return nn.Dense(features=self.output_dim)(x)