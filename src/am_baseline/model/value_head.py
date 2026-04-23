import torch
from torch import nn


class ValueHead(nn.Module):
    """
    Auxiliary value head for Stage 1.

    Maps per-step decoder glimpse vectors to scalar value estimates of the
    normalized cost-to-go from the corresponding partial tour state. Trained
    via MSE against realized cost-to-go; does NOT enter the policy gradient.
    """

    def __init__(self, embedding_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, glimpses):
        # glimpses: (batch, N, embedding_dim)  -> values: (batch, N)
        # glimpses: (batch, embedding_dim)     -> values: (batch,)
        return self.mlp(glimpses).squeeze(-1)
