import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from am_baseline.model.encoder import GraphAttentionEncoder
from am_baseline.model.decoder import Decoder
from am_baseline.model.value_head import ValueHead
from am_baseline.model.value_trunk import ValueTrunk
from am_baseline.problem.tsp import TSP
from am_baseline.utils.tensor_ops import sample_many


def set_decode_type(model, decode_type):
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModel(nn.Module):

    def __init__(self, config):
        super(AttentionModel, self).__init__()

        self.embedding_dim = config.embedding_dim
        self.n_encode_layers = config.n_encode_layers
        self.n_heads = config.n_heads

        # Node embedding: (x, y) -> embedding_dim
        self.init_embed = nn.Linear(2, config.embedding_dim)

        # Encoder
        self.embedder = GraphAttentionEncoder(
            n_heads=config.n_heads,
            embed_dim=config.embedding_dim,
            n_layers=config.n_encode_layers,
            normalization=config.normalization,
            feed_forward_hidden=getattr(config, 'feed_forward_hidden', 512),
        )

        # Decoder
        self.decoder = Decoder(
            embedding_dim=config.embedding_dim,
            n_heads=config.n_heads,
            tanh_clipping=config.tanh_clipping,
        )

        # Value head (Stage 1: auxiliary, does not enter the policy gradient).
        # value_head_type: 'glimpse_mlp' (default, MLP on the policy glimpse) or
        # 'separate_trunk' (Stage 5 §H.7 — own attention over unvisited nodes).
        self.value_enabled = getattr(config, 'value_enabled', True)
        self.value_head_type = getattr(config, 'value_head_type', 'glimpse_mlp')
        # When True (and separate_trunk), the value path gets its OWN encoder
        # over coords, so value training never touches the policy encoder and
        # the policy (hence the E[z|s] reference) is unchanged (Stage 5 §H.7 0b).
        self.value_own_encoder = getattr(config, 'value_own_encoder', False)
        self.value_init_embed = None
        self.value_embedder = None
        if self.value_enabled:
            self.value_head = ValueHead(
                embedding_dim=config.embedding_dim,
                hidden_dim=getattr(config, 'value_hidden_dim', config.embedding_dim),
            )
            if self.value_head_type == 'separate_trunk':
                self.value_trunk = ValueTrunk(
                    embedding_dim=config.embedding_dim,
                    n_heads=config.n_heads,
                    hidden_dim=getattr(config, 'value_hidden_dim', config.embedding_dim),
                )
                if self.value_own_encoder:
                    self.value_init_embed = nn.Linear(2, config.embedding_dim)
                    self.value_embedder = GraphAttentionEncoder(
                        n_heads=config.n_heads,
                        embed_dim=config.embedding_dim,
                        n_layers=config.n_encode_layers,
                        normalization=config.normalization,
                        feed_forward_hidden=getattr(config, 'feed_forward_hidden', 512),
                    )
            else:
                self.value_trunk = None
        else:
            self.value_head = None
            self.value_trunk = None

        self.problem = TSP

    def set_decode_type(self, decode_type, temp=None):
        self.decoder.set_decode_type(decode_type, temp)

    def forward(self, input, return_pi=False, compute_values=False):
        """
        :param input: (batch_size, graph_size, 2) node coordinates
        :param return_pi: if True, also return the sampled/greedy tour tensor pi
        :param compute_values: if True, also return per-step value predictions
                               (requires value_enabled=True). Values shape: (batch, N).
        Returns one of:
            (cost, ll)                      [default]
            (cost, ll, pi)                  [return_pi=True]
            (cost, ll, values)              [compute_values=True]
            (cost, ll, pi, values)          [both flags True]
        """
        embeddings = self.encode(input)

        if compute_values:
            assert self.value_head is not None, \
                "compute_values=True but value head is disabled (value_enabled=False)"
            _log_p, pi, glimpses = self.decoder.decode(
                input, embeddings, self.problem, compute_values=True
            )
            values = self.value_head(glimpses)  # (batch, N)
        else:
            _log_p, pi = self.decoder.decode(input, embeddings, self.problem)

        cost, mask = self.problem.get_costs(input, pi)
        ll = self._calc_log_likelihood(_log_p, pi, mask)

        if compute_values and return_pi:
            return cost, ll, pi, values
        if compute_values:
            return cost, ll, values
        if return_pi:
            return cost, ll, pi
        return cost, ll

    def encode(self, input):
        """Encode input graph. Can be called once and reused for MCTS."""
        return self.embedder(self.init_embed(input))[0]

    def precompute_decoder(self, embeddings):
        """Precompute fixed decoder context from embeddings. For MCTS."""
        return self.decoder.precompute(embeddings)

    def decode_step(self, fixed, state, return_glimpse=False):
        """Single decoding step. For MCTS."""
        return self.decoder.decode_step(fixed, state, return_glimpse=return_glimpse)

    def value_from_state(self, fixed, state, detach_encoder=True):
        """Separate-value-trunk leaf value from a (fixed, state) pair.

        Used when value_head_type='separate_trunk'. Returns a raw cost-to-go
        estimate (B,). `detach_encoder=True` stops the value loss from
        perturbing the shared policy encoder (the lv0-compatible decoupling).
        """
        assert self.value_trunk is not None, \
            "value_from_state requires value_head_type='separate_trunk'"
        if self.value_embedder is not None:
            # Fully separate value encoder: encode coords independently of the
            # policy. `fixed` is ignored; the policy path is untouched.
            h = self.value_embedder(self.value_init_embed(state.loc))[0]
        else:
            h = fixed.node_embeddings
            if detach_encoder:
                h = h.detach()
        bsz, n_nodes, _ = h.shape
        mask = state.get_mask().reshape(bsz, n_nodes)        # (B, N) True = visited
        current = state.get_current_node().reshape(bsz)      # (B,)
        first = state.first_a.reshape(bsz)                   # (B,)
        step_zero = (state.i.reshape(-1) == 0)
        if step_zero.numel() == 1:
            step_zero = step_zero.expand(bsz)
        return self.value_trunk(h, mask, current, first, step_zero)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        return sample_many(
            lambda input: self.decoder.decode(input[0], input[1], self.problem),
            lambda input, pi: self.problem.get_costs(input[0], pi),
            (input, self.encode(input)),
            batch_rep, iter_rep
        )

    def _calc_log_likelihood(self, _log_p, a, mask):
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        return log_p.sum(1)
