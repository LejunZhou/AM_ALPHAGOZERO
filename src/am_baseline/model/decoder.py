import torch
from torch import nn
import math
from typing import NamedTuple


class AttentionModelFixed(NamedTuple):
    """
    Precomputed context for the decoder, fixed during decoding.
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],
            glimpse_val=self.glimpse_val[:, key],
            logit_key=self.logit_key[key],
        )


class Decoder(nn.Module):
    """
    Decoder for TSP using multi-head attention glimpse mechanism.
    Exposes decode_step() for single-step decoding (used by MCTS in Stage 2).
    """

    def __init__(self, embedding_dim, n_heads, tanh_clipping=10.0):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping

        step_context_dim = 2 * embedding_dim  # TSP: first + last node embeddings

        # Learned placeholder for the first decoding step (no node visited yet)
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)

        # Projection layers
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Decode type
        self.decode_type = None
        self.temp = 1.0

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def precompute(self, embeddings, num_steps=1):
        """Precompute fixed attention data from node embeddings. Called once per instance."""
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous(),
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def decode(self, input, embeddings, problem):
        """Full autoregressive decoding loop. Returns (log_p, sequences)."""
        outputs = []
        sequences = []

        state = problem.make_state(input)
        fixed = self.precompute(embeddings)

        while not state.all_finished():
            log_p, mask = self.decode_step(fixed, state)
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])
            state = state.update(selected)
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def decode_step(self, fixed, state, return_glimpse=False):
        """
        Single decoding step. Returns (log_p, mask), optionally glimpse vector.
        MCTS can call this directly for step-by-step tree search.
        """
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_step_context(fixed.node_embeddings, state))

        glimpse_K, glimpse_V, logit_K = fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key
        mask = state.get_mask()

        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        if return_glimpse:
            return log_p, mask, glimpse.squeeze(-2)
        return log_p, mask

    def _get_step_context(self, embeddings, state):
        """Get context for current decoding step (TSP: first + current node embeddings)."""
        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if num_steps == 1:
            if state.i.item() == 0:
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                return embeddings.gather(
                    1,
                    torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                ).view(batch_size, 1, -1)
        # Multi-step (parallel) context
        embeddings_per_step = embeddings.gather(
            1,
            current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
        )
        return torch.cat((
            self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
            torch.cat((
                embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                embeddings_per_step
            ), 2)
        ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )
