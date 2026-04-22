import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from am_baseline.model.encoder import GraphAttentionEncoder
from am_baseline.model.decoder import Decoder
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

        self.problem = TSP

    def set_decode_type(self, decode_type, temp=None):
        self.decoder.set_decode_type(decode_type, temp)

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, 2) node coordinates
        :return: (cost, log_likelihood) or (cost, log_likelihood, pi) if return_pi
        """
        embeddings = self.encode(input)
        _log_p, pi = self.decoder.decode(input, embeddings, self.problem)

        cost, mask = self.problem.get_costs(input, pi)
        ll = self._calc_log_likelihood(_log_p, pi, mask)

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
