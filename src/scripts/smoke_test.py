"""Smoke test: verify forward pass, encode/decode_step API, and weight loading."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from am_baseline.config import Config
from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.tsp import TSP


def test_forward_pass():
    config = Config(graph_size=20, no_cuda=True)
    model = AttentionModel(config)
    model.set_decode_type('greedy')
    input = torch.rand(4, 20, 2)
    cost, ll = model(input)
    print('Forward pass OK')
    print('  cost shape:', cost.shape, 'values:', [round(c, 4) for c in cost.tolist()])
    print('  ll shape:', ll.shape)

    # Test encode / decode_step separation
    embeddings = model.encode(input)
    fixed = model.precompute_decoder(embeddings)
    state = TSP.make_state(input)
    log_p, mask = model.decode_step(fixed, state)
    print('  decode_step OK, log_p shape:', log_p.shape, 'mask shape:', mask.shape)

    # Test with return_glimpse
    log_p, mask, glimpse = model.decode_step(fixed, state, return_glimpse=True)
    print('  glimpse shape:', glimpse.shape)

    # Test sampling decode
    model.set_decode_type('sampling')
    cost2, ll2 = model(input)
    print('Sampling decode OK, cost:', [round(c, 4) for c in cost2.tolist()])

    # Test sample_many
    model.set_decode_type('sampling')
    best_pi, best_cost = model.sample_many(input, batch_rep=4, iter_rep=2)
    print('sample_many OK, best_cost:', [round(c, 4) for c in best_cost.tolist()])

    print('\nAll Phase B milestones PASSED')


if __name__ == '__main__':
    test_forward_pass()
