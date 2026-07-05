import torch
from torch import nn


class ValueTrunk(nn.Module):
    """Separate value head with its OWN attention over the unvisited nodes.

    Stage 5 §H.7. Decoupled from the policy glimpse: the query is built from the
    current + first node embeddings (a learned placeholder at step 0), and the
    keys/values are the node embeddings with **visited nodes masked out**, so the
    trunk attends over the remaining sub-tour — the geometry that actually
    determines cost-to-go. Output is a scalar RAW cost-to-go estimate
    (paired with value_target_norm='none').

    Unlike `ValueHead` (an MLP on the policy's pre-logit glimpse), this gives the
    value function an input the policy representation does not bottleneck.
    """

    def __init__(self, embedding_dim, n_heads=8, hidden_dim=None):
        super().__init__()
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"
        d = embedding_dim
        hidden_dim = hidden_dim if hidden_dim is not None else d
        self.n_heads = n_heads
        self.head_dim = d // n_heads

        self.query_proj = nn.Linear(2 * d, d, bias=False)   # [h_current ; h_first] -> d
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_out = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Learned query input for step 0 (no current/first node selected yet).
        self.W_placeholder = nn.Parameter(torch.empty(2 * d))
        nn.init.uniform_(self.W_placeholder, -1.0, 1.0)

    def forward(self, node_embeddings, visited_mask, current_idx, first_idx, step_zero):
        """
        node_embeddings : (B, N, d)
        visited_mask    : (B, N) bool, True = visited/illegal (masked OUT of attention)
        current_idx     : (B,) long  — current (last-placed) node
        first_idx       : (B,) long  — tour start node
        step_zero       : (B,) bool  — True where the state is the empty step-0 root
        returns         : (B,) raw cost-to-go estimate
        """
        B, N, d = node_embeddings.shape
        H, dk = self.n_heads, self.head_dim
        h = node_embeddings

        hc = h.gather(1, current_idx.view(B, 1, 1).expand(B, 1, d)).squeeze(1)  # (B, d)
        hf = h.gather(1, first_idx.view(B, 1, 1).expand(B, 1, d)).squeeze(1)    # (B, d)
        ctx_in = torch.cat([hc, hf], dim=-1)                                    # (B, 2d)
        ctx_in = torch.where(
            step_zero.view(B, 1),
            self.W_placeholder.unsqueeze(0).expand(B, -1),
            ctx_in,
        )
        q = self.query_proj(ctx_in)                                            # (B, d)

        Q = self.W_q(q).view(B, H, dk)                                         # (B, H, dk)
        K = self.W_k(h).view(B, N, H, dk).permute(0, 2, 1, 3)                  # (B, H, N, dk)
        V = self.W_v(h).view(B, N, H, dk).permute(0, 2, 1, 3)                  # (B, H, N, dk)

        scores = (Q.unsqueeze(2) * K).sum(-1) / (dk ** 0.5)                    # (B, H, N)
        # Attend over UNVISITED nodes only (mask out visited).
        scores = scores.masked_fill(visited_mask.view(B, 1, N), float("-inf"))
        attn = torch.softmax(scores, dim=-1)                                  # (B, H, N)
        ctx = (attn.unsqueeze(-1) * V).sum(dim=2).reshape(B, d)               # (B, d)
        ctx = self.W_out(ctx)
        return self.mlp(ctx).squeeze(-1)                                      # (B,)
